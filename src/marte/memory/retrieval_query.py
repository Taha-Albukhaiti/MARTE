from __future__ import annotations

from typing import Any, Mapping


def _build_where_clause(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` clause from metadata filters."""
    if not filters:
        return None
    items = []
    for key, value in filters.items():
        if value in ("", None):
            continue
        if isinstance(value, dict):
            items.append({key: value})
        else:
            items.append({key: {"$eq": value}})
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {"$and": items}


def _build_retrieval_query(run_state: Mapping[str, Any]) -> str:
    """Compose a situation-aware planner query from the latest run state."""
    parts = [
        _labeled("scenario", run_state.get("scenario")),
        _labeled("goal", run_state.get("goal")),
        _labeled("last_strategy", run_state.get("last_strategy")),
        _labeled("last_prompt", run_state.get("last_prompt"), limit=300),
        _labeled("last_judge", run_state.get("last_judge_verdict")),
        _labeled("input_block", run_state.get("last_input_reason"), limit=200),
    ]
    parts.extend(_tool_query_parts(run_state.get("last_tool_trace")))
    return " | ".join(part for part in parts if part) or "attack"


def _build_mutator_retrieval_query(
    strategy: str,
    objective: str,
    run_state: Mapping[str, Any],
) -> str:
    """Compose the strategy-specific query used after planner selection."""
    parts = [
        _labeled("strategy", strategy),
        _labeled("objective", objective),
        _labeled("scenario", run_state.get("scenario")),
        _labeled("last_prompt", run_state.get("last_prompt"), limit=300),
    ]
    parts.extend(_tool_query_parts(run_state.get("last_tool_trace")))
    return " | ".join(part for part in parts if part) or f"strategy: {strategy}"


def _build_retrieval_filters(
    run_state: Mapping[str, Any],
    *,
    memory_kind: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Extract constrained metadata filters from run state."""
    filters: dict[str, Any] = {}
    if scenario := str(run_state.get("scenario", "")).strip():
        filters["scenario"] = scenario
    if level := str(run_state.get("guardrail_level", "")).strip():
        filters["guardrail_level"] = level
    if run_id := _current_memory_run_id(run_state):
        filters["run_id"] = {"$ne": run_id}
    if memory_kind:
        filters["memory_kind"] = memory_kind
    if strategy:
        filters["strategy"] = strategy
    return filters


def _tool_query_parts(raw_trace: Any) -> list[str]:
    if not isinstance(raw_trace, list):
        return []
    parts: list[str] = []
    for item in raw_trace[-2:]:
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation", "")).strip()
        target = str(item.get("target", "")).strip()
        error = str(item.get("error", "") or "").strip()
        if operation or target:
            parts.append(f"tool: {operation}:{target}".rstrip(":"))
        if error:
            parts.append(f"tool_block: {error[:200]}")
    return parts


def _labeled(label: str, value: Any, *, limit: int = 200) -> str:
    text = str(value or "").strip()
    return f"{label}: {text[:limit]}" if text else ""


def _current_memory_run_id(run_state: Mapping[str, Any]) -> int | None:
    return _coerce_positive_int(run_state.get("memory_run_id")) or _coerce_positive_int(
        run_state.get("run_id")
    )


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
