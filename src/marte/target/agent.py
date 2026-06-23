from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from marte.llm.errors import NonRetryableLLMResponseError
from marte.llm.phase_retry import PhaseRetryPolicy, retry_llm_phase
from marte.prompts.loader import load_prompt
from marte.target.guardrail import GuardrailDecision, GuardrailEngine, GuardrailLevel
from marte.target.tool_call_parser import _ms, _parse_tool_call
from marte.target.tool_schema import _build_tool_definitions
from marte.target.model_adapter import (
    LLMBackend,
    ModelAdapter,
)
from marte.target.sandbox import SandboxBoundary
from marte.target.tool_broker import DispatchResult, ToolBroker
from marte.target.tools import FileTool, SQLTool, ToolCall

_TARGET_SYSTEM_PROMPT = load_prompt("target/target_system.txt")
_TOOL_FEEDBACK_USER_PROMPT = load_prompt("target/tool_feedback_user.txt")

_ZERO_TIMINGS: dict[str, float] = {
    "llm_ms": 0.0,
    "guardrail_ms": 0.0,
    "tool_ms": 0.0,
}


@dataclass
class StepResult:
    """Structured output of one TargetAgent step, returned to TargetExecutor.

    Always fully populated — even on errors or denied requests.

    Invariant: len(policy_decisions) == len(tool_trace).
    policy_decisions[i] is the ToolBroker decision for tool_trace[i].

    Input- and output-guardrail verdicts are kept in separate fields so they
    cannot corrupt the per-tool-call audit record.
    """

    response_text: str
    tool_trace: list[ToolCall] = field(default_factory=list)
    policy_decisions: list[str] = field(default_factory=list)
    input_decision: str = "allow"  # verdict of check_input()
    output_decision: str = "allow"  # verdict of check_output()
    input_reason: str = ""
    output_reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)


class TargetAgent:
    """Facade coordinating ModelAdapter, GuardrailEngine, and ToolBroker.

    Lifecycle per run:
        agent.init_config(guardrail_level, tool_allowlist, model_cfg)
        result = agent.execute_step(prompt)
    """

    def __init__(
        self,
        sandbox: SandboxBoundary,
        model_backend: LLMBackend,
        phase_retry_policy: PhaseRetryPolicy | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._model_backend = model_backend
        self._guardrail: GuardrailEngine | None = None
        self._broker: ToolBroker | None = None
        self._model_cfg: dict[str, Any] = {}
        self._model_adapter: ModelAdapter | None = None
        self._tool_feedback_loop = False
        self._tool_definitions: list[dict[str, Any]] = []
        self._phase_retry_policy = (
            phase_retry_policy or PhaseRetryPolicy.from_env()
        )

    def init_config(self, guardrail_level: GuardrailLevel, tool_allowlist: list[str],
                    model_cfg: dict[str, Any] | None = None, tool_feedback_loop: bool = False) -> None:
        """Set run-specific config for guardrail, tools, and optional tool feedback.

        Must be called after sandbox.reset_sandbox() so that the DB and FS
        already reflect the seed state when the system prompt is built.
        """
        self._model_cfg = model_cfg or {}
        self._model_adapter = ModelAdapter(self._model_backend, self._model_cfg)
        self._tool_feedback_loop = tool_feedback_loop
        self._guardrail = GuardrailEngine(guardrail_level)
        self._broker = ToolBroker(guardrail=self._guardrail,
                                  sql_tool=SQLTool(self._sandbox.db_path) if "sql" in tool_allowlist else None,
                                  file_tool=FileTool(self._sandbox.fs_path) if "file" in tool_allowlist else None)
        # Build tool definitions after sandbox reset so schema/FS are current.
        # The DB schema and FS layout live in the tool descriptions, following
        # the OpenAI function-calling pattern used in production agentic systems.
        self._tool_definitions = _build_tool_definitions(self._sandbox, list(tool_allowlist))

    def execute_step(self, prompt: str) -> StepResult:
        """Run one attack step.

        Default flow:
        1. Input guardrail check
        2. Initial LLM response
        3. Optional tool dispatch from response JSON
        4. Output guardrail check

        If `tool_feedback_loop` is enabled and step 3 executed a tool call,
        the tool result is sent back to the model for a second (final) response
        before step 4.

        Structured provider failures from the initial model call are propagated
        so RunOrchestrator can retry this phase without consuming a step.
        """
        if self._guardrail is None or self._broker is None:
            return StepResult(
                response_text="[ERROR] init_config() not called",
                input_decision="deny",
                input_reason="init_config() not called",
                timings=dict(_ZERO_TIMINGS),
            )

        tool_trace: list[ToolCall] = []
        policy_decisions: list[str] = []
        timings: dict[str, float] = dict(_ZERO_TIMINGS)

        # 1. input guardrail
        in_dec = self._timed_check_input(prompt, timings)

        if in_dec.action != "allow":
            return StepResult(
                response_text=f"[BLOCKED] {in_dec.reason}",
                tool_trace=tool_trace,
                policy_decisions=policy_decisions,
                input_decision=in_dec.action,
                input_reason=in_dec.reason,
                timings=timings,
            )

        # 2. initial LLM call
        t0 = time.monotonic()
        try:
            response_text = self._complete_with_model(prompt)
        except NonRetryableLLMResponseError:
            timings["llm_ms"] += _ms(t0)
            raise
        except Exception as exc:  # noqa: BLE001
            timings["llm_ms"] += _ms(t0)
            # No tool ran, policy_decisions stays empty.
            return StepResult(
                response_text=f"[LLM ERROR] {exc}",
                tool_trace=tool_trace,
                policy_decisions=policy_decisions,
                input_decision=in_dec.action,
                input_reason=in_dec.reason,
                timings=timings,
            )
        timings["llm_ms"] += _ms(t0)

        # 3. tool dispatch
        tool_call = self._record_tool_dispatch(response_text=response_text, tool_trace=tool_trace,
                                               policy_decisions=policy_decisions, timings=timings)
        # 3b. optional tool-feedback ReAct call
        if self._tool_feedback_loop and tool_call is not None:
            t0 = time.monotonic()
            feedback_retry_count = 0

            def record_feedback_retry(
                retry_number: int,
                exc: NonRetryableLLMResponseError,
            ) -> None:
                nonlocal feedback_retry_count
                feedback_retry_count = retry_number

            try:
                response_text = retry_llm_phase(
                    lambda: self._complete_with_tool_feedback(
                        prompt=prompt,
                        initial_response=response_text,
                        tool_call=tool_call,
                    ),
                    phase="target_tool_feedback",
                    policy=self._phase_retry_policy,
                    on_retry=record_feedback_retry,
                )
            except NonRetryableLLMResponseError:
                timings["llm_ms"] += _ms(t0)
                raise
            except Exception as exc:  # noqa: BLE001
                timings["llm_ms"] += _ms(t0)
                return StepResult(
                    response_text=f"[LLM ERROR] {exc}",
                    tool_trace=tool_trace,
                    policy_decisions=policy_decisions,
                    input_decision=in_dec.action,
                    input_reason=in_dec.reason,
                    timings=timings,
                )
            timings["llm_ms"] += _ms(t0)
            if feedback_retry_count:
                timings["target_tool_feedback_retries"] = float(
                    feedback_retry_count
                )

        # 4. output guardrail
        # Verdict goes into output_decision — never overwrites policy_decisions.
        out_dec = self._timed_check_output(response_text, timings)

        if out_dec.action != "allow":
            response_text = f"[OUTPUT BLOCKED] {out_dec.reason}"

        return StepResult(
            response_text=response_text,
            tool_trace=tool_trace,
            policy_decisions=policy_decisions,
            input_decision=in_dec.action,
            output_decision=out_dec.action,
            input_reason=in_dec.reason,
            output_reason=out_dec.reason,
            timings=timings,
        )

    def _timed_check_input(self, prompt: str, timings: dict[str, float]) -> GuardrailDecision:
        guardrail = self._guardrail
        assert guardrail is not None
        t0 = time.monotonic()
        decision = guardrail.check_input(prompt)
        timings["guardrail_ms"] += _ms(t0)
        return decision

    def _timed_check_output(self, response_text: str, timings: dict[str, float]) -> GuardrailDecision:
        guardrail = self._guardrail
        assert guardrail is not None
        t0 = time.monotonic()
        decision = guardrail.check_output(response_text)
        timings["guardrail_ms"] += _ms(t0)
        return decision

    def _complete_with_model(self, prompt: str) -> str:
        assert self._model_adapter is not None
        return self._model_adapter.complete(
            [
                {"role": "system", "content": _TARGET_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=self._tool_definitions or None,
        )

    def _record_tool_dispatch(self, *, response_text: str, tool_trace: list[ToolCall], policy_decisions: list[str],
                              timings: dict[str, float]) -> ToolCall | None:
        t0 = time.monotonic()
        spec = _parse_tool_call(response_text)
        if spec is not None:
            dispatch_result = self._dispatch(spec)
            tool_trace.append(dispatch_result.tool_call)
            policy_decisions.append(dispatch_result.policy_decision.action)
            timings["tool_ms"] += _ms(t0)
            return dispatch_result.tool_call
        timings["tool_ms"] += _ms(t0)
        return None

    def _complete_with_tool_feedback(self, *, prompt: str, initial_response: str, tool_call: ToolCall) -> str:
        assert self._model_adapter is not None
        tool_result = json.dumps(
            {
                "tool": tool_call.tool,
                "operation": tool_call.operation,
                "target": tool_call.target,
                "output": tool_call.output,
                "error": tool_call.error,
            }
        )
        feedback_prompt = _TOOL_FEEDBACK_USER_PROMPT.format(
            original_prompt=prompt,
            initial_response=initial_response,
            tool_result=tool_result
        )
        return self._model_adapter.complete(
            [
                {"role": "system", "content": _TARGET_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": initial_response},
                {"role": "user", "content": feedback_prompt},
            ]
        )

    def _dispatch(self, spec: dict[str, Any]) -> DispatchResult:
        broker = self._broker
        assert broker is not None
        tool = spec.get("tool", "")
        if tool == "sql":
            return broker.dispatch_sql(spec.get("input", ""))
        if tool == "file_read":
            return broker.dispatch_file_read(spec.get("path", ""))
        if tool == "file_write":
            return broker.dispatch_file_write(
                spec.get("path", ""), spec.get("content", "")
            )
        reason = f"unknown tool: {tool!r}"
        return DispatchResult(
            tool_call=ToolCall(tool=tool or "unknown", operation="DENIED", target="", raw_input=str(spec), output="",
                               error=f"[DENIED] {reason}"),
            policy_decision=GuardrailDecision(action="deny", reason=reason),
        )
