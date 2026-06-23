"""Serialization and timing helpers for RunOrchestrator."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from marte.attack.planner import AttackPlan


def _serialize_plan(plan: AttackPlan) -> dict[str, Any] | str:
    if is_dataclass(plan):
        return asdict(plan)
    return str(plan)


def _debug_attr(obj: object, name: str) -> Any | None:
    value = getattr(obj, name, None)
    if callable(value):
        return None
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)
