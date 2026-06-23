from __future__ import annotations

import datetime
import time
from typing import Any, Callable

from marte.attack.orchestrator import RunResult
from marte.config.snapshot import ConfigSnapshot

from marte.experiment.event_builder import build_step_event
from marte.experiment.debug_trace import safe_append_jsonl, safe_touch_debug_file
from marte.experiment.memory_metrics import (
    collection_token,
    ltm_hits_count,
    ltm_scope_name,
    memory_context_length,
    set_ltm_scope,
)

# 2×2×2 = 8 conditions in insertion order
_SCENARIOS = ("sql", "file")
_GUARDRAIL_LEVELS = ("moderate", "enterprise")
_MEMORY_MODES = ("stm_only", "rag_ltm")

_CONDITIONS: tuple[tuple[str, str, str], ...] = tuple(
    (scenario, guardrail, memory)
    for scenario in _SCENARIOS
    for guardrail in _GUARDRAIL_LEVELS
    for memory in _MEMORY_MODES
)

CONDITIONS: tuple[tuple[str, str, str], ...] = _CONDITIONS

StepEventSink = Callable[[dict[str, Any]], None]
RunCallback = Callable[[ConfigSnapshot, StepEventSink | None], RunResult]
ExecutionItem = tuple[str, str, str, int]


def format_condition(scenario: str, guardrail_level: str, memory_mode: str) -> str:
    return f"{scenario}_{guardrail_level}_{memory_mode}"


def all_conditions() -> tuple[tuple[str, str, str], ...]:
    return CONDITIONS


def build_execution_schedule(
    conditions: tuple[tuple[str, str, str], ...],
    runs_per_condition: int,
) -> tuple[ExecutionItem, ...]:
    """Build a seed-wise counterbalanced execution schedule.

    When both memory modes are present for a scenario/guardrail pair, odd
    seeds execute ``stm_only`` first and even seeds execute ``rag_ltm`` first.
    """
    if runs_per_condition < 1:
        raise ValueError("runs_per_condition must be at least 1")

    pair_order: list[tuple[str, str]] = []
    modes_by_pair: dict[tuple[str, str], list[str]] = {}
    for scenario, guardrail_level, memory_mode in conditions:
        pair = (scenario, guardrail_level)
        if pair not in modes_by_pair:
            pair_order.append(pair)
            modes_by_pair[pair] = []
        if memory_mode not in modes_by_pair[pair]:
            modes_by_pair[pair].append(memory_mode)

    schedule: list[ExecutionItem] = []
    for scenario, guardrail_level in pair_order:
        configured_modes = modes_by_pair[(scenario, guardrail_level)]
        has_both_memory_modes = all(mode in configured_modes for mode in _MEMORY_MODES)

        for seed in range(1, runs_per_condition + 1):
            if has_both_memory_modes:
                preferred_order = _MEMORY_MODES if seed % 2 == 1 else tuple(reversed(_MEMORY_MODES))
                ordered_modes = [mode for mode in preferred_order if mode in configured_modes]
                ordered_modes.extend(mode for mode in configured_modes if mode not in ordered_modes)
            else:
                ordered_modes = configured_modes

            schedule.extend(
                (scenario, guardrail_level, memory_mode, seed)
                for memory_mode in ordered_modes
            )

    return tuple(schedule)


def generate_experiment_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("exp_%Y-%m-%d_%H-%M-%S-%f")


def wall_time_ms() -> int:
    return int(time.time() * 1000)


__all__ = [
    "CONDITIONS",
    "ExecutionItem",
    "RunCallback",
    "StepEventSink",
    "all_conditions",
    "build_execution_schedule",
    "build_step_event",
    "collection_token",
    "format_condition",
    "generate_experiment_id",
    "ltm_hits_count",
    "ltm_scope_name",
    "memory_context_length",
    "safe_append_jsonl",
    "safe_touch_debug_file",
    "set_ltm_scope",
    "wall_time_ms",
]
