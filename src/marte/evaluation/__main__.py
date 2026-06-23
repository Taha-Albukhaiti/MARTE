"""
Generic post-hoc statistical evaluation CLI for MARTE experiments.

This command evaluates one EventStore database and is intended for standard
experiment checks and reusable text/JSON reports. The final thesis artifact
pipeline lives in ``scripts/build_thesis_evaluation.py`` because it combines
the four frozen result sources and writes report-ready tables, figures, audits,
and provenance metadata.

Usage
-----
    python -m marte.evaluation --db-path results/events.db
    python -m marte.evaluation --db-path results/events.db --experiment-id exp_2026-04-15_...
    python -m marte.evaluation --db-path results/events.db --output-dir results/reports

What it produces
----------------
For every experiment found in the database (or the one specified via
--experiment-id) it prints a human-readable table and optionally writes
a machine-readable JSON report to --output-dir.

Outputs per experiment
----------------------
  1. Per-condition summary: ASR, N runs, N breaches, median Q2B
  2. Four condition comparisons (stm_only vs rag_ltm per scenario/guardrail):
       - Fisher's Exact Test p-value and odds-ratio (ASR)
       - Log-Rank Test p-value (Q2B survival)
       - Log-Rank Test p-value (Time-to-Breach survival)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marte.evaluation.reporting import (
    _build_comparison_table,
    _build_condition_summary,
    _build_json_report,
    _evaluate_experiment,
    _fmt_asr,
    _fmt_float,
    _fmt_median_q2b,
    _fmt_p,
    _holm_family_sizes,
    _km_median_q2b,
    _list_experiment_ids,
    _print_comparison_table,
    _print_condition_summary,
    _print_validation,
)

__all__ = [
    "main",
    "_build_comparison_table",
    "_build_condition_summary",
    "_build_json_report",
    "_evaluate_experiment",
    "_fmt_asr",
    "_fmt_float",
    "_fmt_median_q2b",
    "_fmt_p",
    "_holm_family_sizes",
    "_km_median_q2b",
    "_list_experiment_ids",
    "_print_comparison_table",
    "_print_condition_summary",
    "_print_validation",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m marte.evaluation",
        description=(
            "Generic post-hoc statistical evaluation for one MARTE EventStore "
            "database. Use scripts/build_thesis_evaluation.py for the final "
            "four-source thesis artifacts."
        ),
    )
    parser.add_argument(
        "--db-path",
        required=True,
        metavar="PATH",
        help="Path to the SQLite EventStore database (e.g. results/events.db).",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        metavar="ID",
        help=(
            "Experiment ID to evaluate. "
            "If omitted, all experiments in the database are evaluated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write JSON report files into. "
            "If omitted, only console output is produced."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort with exit code 1 when validation reports any error.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=30,
        metavar="N",
        help="Expected valid stop_events per condition (default: 30).",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=50,
        metavar="B",
        help="Expected max_budget used by budget_exhausted runs (default: 50).",
    )
    args = parser.parse_args(argv)

    if args.expected_runs <= 0:
        print("ERROR: --expected-runs must be positive.", file=sys.stderr)
        raise SystemExit(1)
    if args.max_budget <= 0:
        print("ERROR: --max-budget must be positive.", file=sys.stderr)
        raise SystemExit(1)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    from marte.persistence.event_store import EventStore

    store = EventStore(db_path)

    if args.experiment_id:
        experiment_ids = [args.experiment_id]
    else:
        experiment_ids = _list_experiment_ids(store)
        if not experiment_ids:
            print("No experiments found in the database.", file=sys.stderr)
            raise SystemExit(1)
        print(f"Found {len(experiment_ids)} experiment(s) in {db_path}.", flush=True)

    for experiment_id in experiment_ids:
        _evaluate_experiment(
            store,
            experiment_id,
            output_dir,
            expected_runs_per_condition=args.expected_runs,
            max_budget=args.max_budget,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
