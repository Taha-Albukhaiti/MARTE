from __future__ import annotations

from typing import Any


def _classify_ltm_episode(
    episode_dict: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Classify an episode for LTM or return None for STM-only storage."""
    if bool(episode_dict.get("oracle_verdict")):
        return "positive", "", ""

    label = str(episode_dict.get("judge_verdict", "")).strip().lower()
    if label == "partial":
        return "positive", "", ""

    input_decision = str(episode_dict.get("input_decision", "")).strip().lower()
    if input_decision in {"deny", "sanitize"}:
        reason = str(episode_dict.get("input_reason", "")).strip()
        if _is_infrastructure_policy_reason(reason):
            return None
        return "blocked", "input", reason or f"input_{input_decision}"

    decisions = episode_dict.get("policy_decisions", [])
    tool_trace = episode_dict.get("tool_trace", [])
    if isinstance(decisions, list):
        for index, raw_decision in enumerate(decisions):
            decision = str(raw_decision).strip().lower()
            if decision not in {"deny", "sanitize"}:
                continue
            tool_call = (
                tool_trace[index]
                if isinstance(tool_trace, list) and index < len(tool_trace)
                else None
            )
            reason = _tool_error(tool_call)
            if _is_infrastructure_policy_reason(reason):
                continue
            return "blocked", "tool", reason or f"tool_{decision}"

    return None


def _should_store_in_ltm(episode_dict: dict[str, Any]) -> bool:
    """Return whether an episode belongs in a run-spanning LTM channel."""
    return _classify_ltm_episode(episode_dict) is not None


def _summarize_tool_effect(tool_trace: list[Any]) -> str:
    """Produce a compact text summary of the tool calls in a step."""
    parts: list[str] = []
    for call in tool_trace:
        if isinstance(call, dict):
            op = call.get("operation", "")
            target = call.get("target", "")
        else:
            op = getattr(call, "operation", "")
            target = getattr(call, "target", "")
        if op:
            parts.append(f"{op}:{target}" if target else str(op))
    return ", ".join(parts[:5])


def _tool_error(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("error", "") or "").strip()
    return str(getattr(tool_call, "error", "") or "").strip()


def _is_infrastructure_policy_reason(reason: str) -> bool:
    lowered = reason.lower()
    return (
        lowered.startswith("toolbroker:")
        or "not configured" in lowered
        or "init_config() not called" in lowered
        or lowered.startswith("target execution failed:")
    )
