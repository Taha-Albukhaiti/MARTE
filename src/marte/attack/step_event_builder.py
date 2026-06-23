from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from marte.attack.helpers.orchestrator import (
    _debug_attr,
    _elapsed_ms,
    _json_safe,
    _serialize_plan,
)
from marte.attack.target_executor import ExecutionResult

if TYPE_CHECKING:
    from marte.attack.interfaces import JudgeProtocol, MutatorProtocol, PlannerProtocol
    from marte.attack.planner import AttackPlan
    from marte.target.sandbox import SandboxSnapshot


def build_step_event(
    *,
    step_idx: int,
    run_started_at: float,
    memory_context: Mapping[str, Any],
    retrieval_trace: Mapping[str, Any],
    plan: AttackPlan,
    prompt: str,
    execution_result: ExecutionResult,
    pre_snapshot: SandboxSnapshot,
    post_snapshot: SandboxSnapshot,
    oracle_verdict: bool,
    judge_verdict: str,
    phase_errors: tuple[str | None, ...],
    phase_retry_counts: dict[str, int],
    planner: PlannerProtocol,
    mutator: MutatorProtocol,
    judge: JudgeProtocol,
) -> dict[str, Any]:
    """Build the raw step event dict persisted by ComparativeRunner."""
    timings = dict(execution_result.step_result.timings)
    if phase_retry_counts:
        timings["technical_retries"] = dict(phase_retry_counts)

    return {
        "step_index": step_idx,
        "timestamp_ms": _elapsed_ms(run_started_at),
        "memory_context": _json_safe(memory_context),
        "prompt": prompt,
        "response_text": execution_result.step_result.response_text,
        "plan": _serialize_plan(plan),
        "planner_raw_response": _debug_attr(planner, "last_raw_response"),
        "planner_error": _debug_attr(planner, "last_error"),
        "planner_selected_strategy": _debug_attr(planner, "last_selected_strategy"),
        "mutator_raw_response": _debug_attr(mutator, "last_raw_response"),
        "mutator_sanitized_response": _debug_attr(mutator, "last_sanitized_response"),
        "mutator_sanitization_actions": _debug_attr(mutator, "last_sanitization_actions"),
        "mutator_used_fallback": _debug_attr(mutator, "last_used_fallback"),
        "mutator_fallback_reason": _debug_attr(mutator, "last_fallback_reason"),
        "mutator_error": _debug_attr(mutator, "last_error"),
        "judge_raw_response": _debug_attr(judge, "last_raw_response"),
        "judge_error": _debug_attr(judge, "last_error"),
        "judge_used_fallback": _debug_attr(judge, "last_used_fallback"),
        "judge_rationale": _debug_attr(judge, "last_rationale"),
        "judge_memory_summary": _debug_attr(judge, "last_memory_summary"),
        "judge_strategy_feedback": _debug_attr(judge, "last_strategy_feedback"),
        "judge_confidence": _debug_attr(judge, "last_confidence"),
        "judge_conflict": _debug_attr(judge, "last_conflict"),
        "oracle_verdict": oracle_verdict,
        "judge_verdict": judge_verdict,
        "pre_snapshot": pre_snapshot.model_dump(),
        "post_snapshot": post_snapshot.model_dump(),
        "tool_trace": [tc.model_dump() for tc in execution_result.step_result.tool_trace],
        "policy_decisions": list(execution_result.step_result.policy_decisions),
        "input_decision": execution_result.step_result.input_decision,
        "output_decision": execution_result.step_result.output_decision,
        "input_reason": execution_result.step_result.input_reason,
        "output_reason": execution_result.step_result.output_reason,
        "retrieval_trace": _json_safe(retrieval_trace),
        "timings": timings,
        "errors": [err for err in phase_errors if err],
    }
