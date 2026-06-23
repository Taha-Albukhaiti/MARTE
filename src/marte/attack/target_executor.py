from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from marte.llm.errors import NonRetryableLLMResponseError
from marte.target.agent import StepResult, _ZERO_TIMINGS


@dataclass
class ExecutionResult:
    prompt: str
    step_result: Any
    error: str | None = None


def make_failed_execution_result(prompt: str, error: str) -> ExecutionResult:
    return ExecutionResult(
        prompt=prompt,
        step_result=StepResult(
            response_text=f"[TARGET ERROR] {error}",
            tool_trace=[],
            policy_decisions=[],
            input_decision="deny",
            output_decision="allow",
            input_reason=f"target execution failed: {error}",
            timings=dict(_ZERO_TIMINGS),
        ),
        error=f"target execution failed: {error}",
    )


class TargetStepAPI(Protocol):
    def execute_step(self, prompt: str) -> StepResult:
        ...


class TargetExecutor:
    """Technical execution layer: prompt in, StepResult out."""

    def __init__(self, target_agent: TargetStepAPI) -> None:
        self._target_agent = target_agent

    def execute(self, prompt: str) -> ExecutionResult:
        try:
            step_result = self._target_agent.execute_step(prompt)
            return ExecutionResult(prompt=prompt, step_result=step_result, error=None)
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            return make_failed_execution_result(prompt, str(exc))
