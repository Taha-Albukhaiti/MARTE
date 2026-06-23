from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marte.attack.planner import AttackPlan
    from marte.attack.target_executor import ExecutionResult


def update_run_state_after_step(
    *,
    run_state: dict[str, Any],
    step_idx: int,
    plan: AttackPlan,
    prompt: str,
    execution_result: ExecutionResult,
    oracle_verdict: bool,
    judge_verdict: str,
    errors: list[Any],
) -> None:
    """Store the latest step outcome for the next Planner/Mutator turn."""
    run_state.update(
        {
            "last_strategy": plan.strategy,
            "last_prompt": prompt,
            "last_response_text": execution_result.step_result.response_text,
            "last_policy_decisions": list(execution_result.step_result.policy_decisions),
            "last_tool_trace": [tc.model_dump() for tc in execution_result.step_result.tool_trace],
            "last_input_decision": execution_result.step_result.input_decision,
            "last_output_decision": execution_result.step_result.output_decision,
            "last_input_reason": execution_result.step_result.input_reason,
            "last_output_reason": execution_result.step_result.output_reason,
            "last_oracle_verdict": oracle_verdict,
            "last_judge_verdict": judge_verdict,
            "last_errors": list(errors),
            "steps_executed": step_idx,
        }
    )
