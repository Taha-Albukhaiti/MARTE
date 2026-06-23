from __future__ import annotations

from collections import defaultdict
from typing import Any


def _group_stops_by_condition(stops: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for event in stops:
        grouped[event.condition].append(event)
    return dict(grouped)


def _format_values(values: list[int], limit: int = 12) -> str:
    if not values:
        return "none"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(str(value) for value in shown) + suffix
