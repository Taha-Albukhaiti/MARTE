"""Console and JSON reporting for MARTE-thesis-worktree evaluation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from lifelines import KaplanMeierFitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_p(value: float) -> str:
    """Format a p-value for display: NaN, <0.001, or 3 decimals."""
    if math.isnan(value):
        return "   n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _fmt_float(value: float, decimals: int = 3) -> str:
    if math.isnan(value):
        return "    n/a"
    if math.isinf(value):
        return "    inf" if value > 0 else "   -inf"
    return f"{value:.{decimals}f}"


def _fmt_asr(value: float | None) -> str:
    """Format ASR so empty conditions are visible missing data."""
    if value is None:
        return "No data"
    return f"{value:.3f}"


def _km_median_q2b(dist: list[tuple[int, bool]]) -> int | float | None:
    """Kaplan-Meier median Q2B; None means the median was not reached."""
    if not dist:
        return None

    durations = [q2b for q2b, _ in dist]
    observed = [not censored for _, censored in dist]
    if not any(observed):
        return None

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=observed)
    median = float(kmf.median_survival_time_)
    if not math.isfinite(median):
        return None
    if median.is_integer():
        return int(median)
    return median


def _fmt_median_q2b(value: int | float | None) -> str:
    if value is None:
        return "not reached"
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"


def _list_experiment_ids(store) -> list[str]:
    """Return unique experiment IDs found in evaluation-relevant tables."""
    with store._connect() as conn:
        conn.row_factory = None
        cursor = conn.execute(
            """
            SELECT experiment_id FROM stop_events
            UNION
            SELECT experiment_id FROM run_events
            ORDER BY experiment_id
            """
        )
        return [row[0] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_condition_summary(metrics, experiment_id: str, conditions: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        asr = metrics.compute_asr(experiment_id, condition)
        q2b_dist = metrics.get_q2b_distribution(experiment_id, condition)
        n_total = len(q2b_dist)
        n_breach = sum(1 for _, censored in q2b_dist if not censored)
        rows.append(
            {
                "condition": condition,
                "n_total": n_total,
                "n_breach": n_breach,
                "asr": asr,
                "median_q2b": _km_median_q2b(q2b_dist),
            }
        )
    return rows


def _build_comparison_table(stats, experiment_id: str) -> list[dict[str, Any]]:
    from marte.evaluation.evaluation_validator import CONDITION_PAIRS
    from marte.evaluation.stats import holm_adjust_p_values

    rows: list[dict[str, Any]] = []
    for condition_a, condition_b in CONDITION_PAIRS:
        result = stats.compare_conditions(experiment_id, condition_a, condition_b)
        rows.append(
            {
                "pair": f"{condition_a}  vs  {condition_b}",
                "asr_p": result.asr_p_value,
                "asr_or": result.asr_odds_ratio,
                "q2b_p": result.q2b_p_value,
                "t2b_p": result.t2b_p_value,
            }
        )

    asr_holm = holm_adjust_p_values([r["asr_p"] for r in rows])
    q2b_holm = holm_adjust_p_values([r["q2b_p"] for r in rows])
    t2b_holm = holm_adjust_p_values([r["t2b_p"] for r in rows])
    for row, asr_p, q2b_p, t2b_p in zip(rows, asr_holm, q2b_holm, t2b_holm):
        row["asr_p_holm"] = asr_p
        row["q2b_p_holm"] = q2b_p
        row["t2b_p_holm"] = t2b_p
    return rows


def _holm_family_sizes(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return planned and effective Holm family sizes for report transparency.

    ``holm_adjust_p_values`` preserves unavailable p-values as NaN and adjusts
    only finite values. The effective family size therefore may be smaller
    than the planned number of condition pairs; reporting both values prevents
    silent under-reporting in partial or degenerate experiments.
    """
    families = {
        "asr": "asr_p",
        "q2b": "q2b_p",
        "t2b": "t2b_p",
    }
    sizes: dict[str, dict[str, int]] = {}
    for family, key in families.items():
        effective_m = 0
        for row in rows:
            try:
                value = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                effective_m += 1
        sizes[family] = {"planned_m": len(rows), "effective_m": effective_m}
    return sizes


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------


def _print_condition_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print("  Per-condition summary")
    print(f"  {'Condition':<36}  {'N':>4}  {'Breach':>6}  {'ASR':>7}  {'Median Q2B':>12}")
    print(f"  {'-'*36}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*12}")
    for r in rows:
        print(
            f"  {r['condition']:<36}  {r['n_total']:>4}  {r['n_breach']:>6}  "
            f"{_fmt_asr(r['asr']):>7}  {_fmt_median_q2b(r['median_q2b']):>12}"
        )


def _print_comparison_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("  Condition comparisons  (stm_only  vs  rag_ltm)")
    print(
        f"  {'Pair':<56}  {'ASR p':>7}  {'ASR Holm':>8}  {'ASR OR':>8}  "
        f"{'Q2B p':>7}  {'Q2B Holm':>9}  {'T2B p':>7}  {'T2B Holm':>9}"
    )
    print(
        f"  {'-'*56}  {'-'*7}  {'-'*8}  {'-'*8}  "
        f"{'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}"
    )
    for r in rows:
        print(
            f"  {r['pair']:<56}  {_fmt_p(r['asr_p']):>7}  "
            f"{_fmt_p(r['asr_p_holm']):>8}  {_fmt_float(r['asr_or']):>8}  "
            f"{_fmt_p(r['q2b_p']):>7}  {_fmt_p(r['q2b_p_holm']):>9}  "
            f"{_fmt_p(r['t2b_p']):>7}  {_fmt_p(r['t2b_p_holm']):>9}"
        )
    print()
    print("  Significance threshold α = 0.05; Holm columns are family-wise adjusted")
    sizes = _holm_family_sizes(rows)
    print(
        "  Holm effective m: "
        f"ASR {sizes['asr']['effective_m']}/{sizes['asr']['planned_m']}; "
        f"Q2B {sizes['q2b']['effective_m']}/{sizes['q2b']['planned_m']}; "
        f"T2B {sizes['t2b']['effective_m']}/{sizes['t2b']['planned_m']}"
    )
    print("  n/a = insufficient data (< 2 observations per arm or degenerate table)")


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _to_json_safe(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _asr_json_value(value: float | None) -> float | str:
    if value is None:
        return "n/a"
    return round(value, 6)


def _odds_ratio_json_value(value: float | None) -> float | str | None:
    if value is None or math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _validation_json(violations: list[Any]) -> dict[str, Any]:
    errors = sum(1 for violation in violations if violation.level == "error")
    warnings = sum(1 for violation in violations if violation.level == "warning")
    if errors:
        status = "errors"
    elif warnings:
        status = "warnings"
    else:
        status = "passed"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "violations": [
            {
                "level": violation.level,
                "code": violation.code,
                "message": violation.message,
                "condition": violation.condition,
                "run_id": violation.run_id,
            }
            for violation in violations
        ],
    }


def _build_json_report(
    experiment_id: str,
    condition_summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    violations: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "validation": _validation_json(violations or []),
        "holm": _holm_family_sizes(comparisons),
        "conditions": [
            {
                "condition": r["condition"],
                "n_total": r["n_total"],
                "n_breach": r["n_breach"],
                "asr": _asr_json_value(r["asr"]),
                "median_q2b": r["median_q2b"],
            }
            for r in condition_summary
        ],
        "comparisons": [
            {
                "condition_a": pair.split("  vs  ")[0].strip(),
                "condition_b": pair.split("  vs  ")[1].strip(),
                "asr_p_value": _to_json_safe(r["asr_p"]),
                "asr_p_holm": _to_json_safe(r["asr_p_holm"]),
                "asr_odds_ratio": _odds_ratio_json_value(r["asr_or"]),
                "q2b_p_value": _to_json_safe(r["q2b_p"]),
                "q2b_p_holm": _to_json_safe(r["q2b_p_holm"]),
                "t2b_p_value": _to_json_safe(r["t2b_p"]),
                "t2b_p_holm": _to_json_safe(r["t2b_p_holm"]),
            }
            for r, pair in ((r, r["pair"]) for r in comparisons)
        ],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _evaluate_experiment(
    store,
    experiment_id: str,
    output_dir: Path | None,
    expected_runs_per_condition: int,
    max_budget: int,
    strict: bool,
) -> None:
    from marte.evaluation.evaluation_validator import EXPECTED_CONDITIONS, EvaluationValidator
    from marte.evaluation.metrics import MetricsService
    from marte.evaluation.stats import StatsService

    validator = EvaluationValidator(
        store,
        experiment_id=experiment_id,
        expected_runs_per_condition=expected_runs_per_condition,
        max_budget=max_budget,
    )
    violations = validator.check_all()
    _print_validation(violations, strict=strict)
    if strict and any(violation.level == "error" for violation in violations):
        raise SystemExit(1)

    excluded_run_ids = {
        v.run_id
        for v in violations
        if v.level == "error" and v.run_id is not None
    }
    metrics = MetricsService(store, excluded_run_ids=excluded_run_ids if excluded_run_ids else None)
    stats = StatsService(metrics)

    print(f"\n{'=' * 70}")
    print(f"  Experiment: {experiment_id}")
    print(f"{'=' * 70}")

    condition_summary = _build_condition_summary(metrics, experiment_id, EXPECTED_CONDITIONS)
    comparison_rows = _build_comparison_table(stats, experiment_id)

    _print_condition_summary(condition_summary)
    print()
    _print_comparison_table(comparison_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = _build_json_report(
            experiment_id,
            condition_summary,
            comparison_rows,
            violations=violations,
        )
        safe_id = experiment_id.replace(":", "-").replace("/", "-")
        out_path = output_dir / f"{safe_id}_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  JSON report written to: {out_path}")


def _print_validation(violations: list[Any], *, strict: bool) -> None:
    """Print validation status before statistics are computed."""
    if not violations:
        print("Validation: passed.")
        return

    error_count = sum(1 for violation in violations if violation.level == "error")
    warning_count = sum(1 for violation in violations if violation.level == "warning")
    if strict and error_count:
        print("Validation failed:", file=sys.stderr)
    else:
        print(f"Validation issues: {error_count} error(s), {warning_count} warning(s).", file=sys.stderr)
    for violation in violations:
        print(f"  {violation.level.upper()}: {violation.message}", file=sys.stderr)
