"""Console reporting helpers for CLI-launched MARTE-thesis-worktree experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from marte.evaluation.evaluation_validator import CONDITION_PAIRS
from marte.evaluation.metrics import MetricsService
from marte.evaluation.stats import StatsService, holm_adjust_p_values
from marte.persistence.event_store import EventStore


def _print_experiment_summary(runs_per_condition: int, total_runs: int, max_budget: int, num_conditions: int) -> None:
    print(
        f"-------------------------------------------------------------"
        f"\nStarting MARTE-thesis-worktree experiment: {runs_per_condition} runs × {num_conditions} conditions "
        f"= {total_runs} runs, max_budget={max_budget}"
    )


def _handle_experiment_results(event_store: EventStore, experiment_id: str, db_path: Path) -> None:
    metrics = MetricsService(event_store)
    stats = StatsService(metrics)
    significance_report = _build_significance_report(stats, experiment_id)

    print("Demo significance_report:")
    print(json.dumps(_json_sanitize(significance_report), indent=2, allow_nan=False))
    print(f"Experiment {experiment_id} complete. Results in {db_path}")


def _build_significance_report(stats: StatsService, experiment_id: str) -> dict[str, Any]:
    rows: list[tuple[str, Any]] = []
    for condition_a, condition_b in CONDITION_PAIRS:
        result = stats.compare_conditions(experiment_id, condition_a, condition_b)
        rows.append((f"{condition_a}__vs__{condition_b}", result))

    asr_holm = holm_adjust_p_values([result.asr_p_value for _, result in rows])
    q2b_holm = holm_adjust_p_values([result.q2b_p_value for _, result in rows])
    t2b_holm = holm_adjust_p_values([result.t2b_p_value for _, result in rows])

    report: dict[str, Any] = {}
    for (key, result), asr_p, q2b_p, t2b_p in zip(rows, asr_holm, q2b_holm, t2b_holm):
        report[key] = {
            "asr_p_value": result.asr_p_value,
            "asr_p_holm": asr_p,
            "asr_odds_ratio": result.asr_odds_ratio,
            "q2b_p_value": result.q2b_p_value,
            "q2b_p_holm": q2b_p,
            "t2b_p_value": result.t2b_p_value,
            "t2b_p_holm": t2b_p,
        }
    return report


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    return value
