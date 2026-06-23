from __future__ import annotations

import pytest

from marte.attack.target_executor import ExecutionResult, TargetExecutor
from marte.llm.errors import LLMCompletionTruncatedError
from marte.target.agent import StepResult
from marte.target.tools.tool_call import ToolCall


class DummyTargetAgent:
    def __init__(self, result: StepResult) -> None:
        self._result = result
        self.prompts: list[str] = []

    def execute_step(self, prompt: str) -> StepResult:
        self.prompts.append(prompt)
        return self._result


class FailingTargetAgent:
    def execute_step(self, prompt: str) -> StepResult:  # noqa: ARG002
        raise RuntimeError("target boom")


class TruncatedTargetAgent:
    def execute_step(self, prompt: str) -> StepResult:  # noqa: ARG002
        raise LLMCompletionTruncatedError(
            role="target",
            model="gpt-oss-120b",
            finish_reason="length",
            output_limit=4096,
            prompt_tokens=100,
            completion_tokens=4096,
            reasoning_tokens=4000,
        )


def _sample_step_result() -> StepResult:
    return StepResult(
        response_text="ok",
        tool_trace=[
            ToolCall(
                tool="sql",
                operation="SELECT",
                target="users",
                raw_input="SELECT name FROM users",
                output="[]",
                error=None,
            )
        ],
        policy_decisions=["allow"],
        input_decision="allow",
        output_decision="allow",
        timings={"llm_ms": 1.0, "guardrail_ms": 2.0, "tool_ms": 3.0},
    )


def test_execute_returns_bundled_execution_result() -> None:
    target = DummyTargetAgent(_sample_step_result())
    executor = TargetExecutor(target_agent=target)

    result = executor.execute("probe users")

    assert isinstance(result, ExecutionResult)
    assert result.prompt == "probe users"
    assert result.step_result.response_text == "ok"
    assert result.error is None
    assert target.prompts == ["probe users"]


def test_execute_handles_target_exception_without_crash() -> None:
    executor = TargetExecutor(target_agent=FailingTargetAgent())

    result = executor.execute("anything")

    assert result.step_result.response_text.startswith("[TARGET ERROR]")
    assert result.step_result.tool_trace == []
    assert result.step_result.policy_decisions == []
    assert result.step_result.timings == {"llm_ms": 0.0, "guardrail_ms": 0.0, "tool_ms": 0.0}
    assert result.error is not None
    assert "target execution failed" in result.error


def test_execute_propagates_truncated_completion_for_phase_retry() -> None:
    executor = TargetExecutor(target_agent=TruncatedTargetAgent())

    with pytest.raises(LLMCompletionTruncatedError):
        executor.execute("anything")
