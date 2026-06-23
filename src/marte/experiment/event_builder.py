from __future__ import annotations

import json
from typing import Any

from marte.persistence.schemas import StepEvent


def build_step_event(raw_ev: dict[str, Any], *, experiment_id: str, run_id: int, condition: str) -> StepEvent:
    """Convert an orchestrator raw event into the EventStore schema."""
    return StepEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        condition=condition,
        step_t=int(raw_ev.get("step_index", 0)),
        timestamp_ms=int(raw_ev.get("timestamp_ms", 0)),
        prompt=str(raw_ev.get("prompt", "")),
        response=str(raw_ev.get("response_text", "")),
        tool_trace=_as_list(raw_ev.get("tool_trace", [])),
        policy_decisions=_as_list(raw_ev.get("policy_decisions", [])),
        oracle_verdict=bool(raw_ev.get("oracle_verdict", False)),
        judge_verdict=str(raw_ev.get("judge_verdict", "fail")),
        memory_context=_as_dict(raw_ev.get("memory_context", {})),
        plan=_as_dict(raw_ev.get("plan", {})),
        pre_snapshot=_as_dict(raw_ev.get("pre_snapshot", {})),
        post_snapshot=_as_dict(raw_ev.get("post_snapshot", {})),
        input_decision=_stringify(raw_ev.get("input_decision", "")),
        output_decision=_stringify(raw_ev.get("output_decision", "")),
        input_reason=_stringify(raw_ev.get("input_reason", "")),
        output_reason=_stringify(raw_ev.get("output_reason", "")),
        retrieval_trace=_as_dict(raw_ev.get("retrieval_trace", {})),
        timings=_as_dict(raw_ev.get("timings", {})),
        errors=_as_list(raw_ev.get("errors", [])),
        planner_debug=_planner_debug(raw_ev),
        mutator_debug=_mutator_debug(raw_ev),
        judge_debug=_judge_debug(raw_ev),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, default=str)


def _planner_debug(raw_ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_response": raw_ev.get("planner_raw_response", raw_ev.get("planner_raw_output", "")),
        "error": raw_ev.get("planner_error"),
        "selected_strategy": raw_ev.get("planner_selected_strategy"),
    }


def _mutator_debug(raw_ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_response": raw_ev.get("mutator_raw_response", raw_ev.get("mutator_raw_output", "")),
        "sanitized_response": raw_ev.get("mutator_sanitized_response", ""),
        "sanitization_actions": _as_list(raw_ev.get("mutator_sanitization_actions", [])),
        "used_fallback": raw_ev.get("mutator_used_fallback"),
        "fallback_reason": raw_ev.get("mutator_fallback_reason"),
        "error": raw_ev.get("mutator_error"),
    }


def _judge_debug(raw_ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_response": raw_ev.get("judge_raw_response", raw_ev.get("judge_raw_output", "")),
        "error": raw_ev.get("judge_error"),
        "used_fallback": raw_ev.get("judge_used_fallback"),
        "rationale": raw_ev.get("judge_rationale"),
        "memory_summary": raw_ev.get("judge_memory_summary"),
        "strategy_feedback": raw_ev.get("judge_strategy_feedback"),
        "confidence": raw_ev.get("judge_confidence"),
        "oracle_gate_conflict": raw_ev.get("judge_conflict"),
    }
