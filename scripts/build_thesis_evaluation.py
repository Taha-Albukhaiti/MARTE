#!/usr/bin/env python3
"""Build the final read-only thesis evaluation artifacts for MARTE-thesis-worktree.

The script intentionally lives outside ``src/marte`` so it can be added after
the experiments without changing the protocol fingerprint of the experiment
implementation. It reads the final EventStore SQLite databases, validates the
multi-source experiment matrix, computes condition metrics and pairwise tests,
audits RAG retrieval/LTM state, and writes report-ready artifacts.

For generic one-database evaluation, use ``python -m marte.evaluation``. This
script is the thesis-specific artifact builder for the frozen four-source
result set, including LaTeX tables, PDF figures, LTM audits, and provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402
from scipy import stats  # noqa: E402

from marte.attack.oracle import BreachOracle, SandboxPolicy  # noqa: E402
from marte.memory.episode_codec import _blocked_document_id  # noqa: E402
from marte.memory.episode_policy import _classify_ltm_episode  # noqa: E402
from marte.target.sandbox import SandboxSnapshot  # noqa: E402
from marte.target.tools.tool_call import ToolCall  # noqa: E402

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "sql_moderate": {
        "db": "experiment-results/events_sql_moderate.db",
        "experiment_id": "exp_2026-06-19_20-32-33-421921",
        "stm_condition": "sql_moderate_stm_only",
        "rag_condition": "sql_moderate_rag_ltm",
        "expected_runs": 30,
        "max_budget": 50,
    },
    "sql_enterprise": {
        "db": "experiment-results/events_sql_enterprise.db",
        "experiment_id": "exp_2026-06-21_05-35-31-469253",
        "stm_condition": "sql_enterprise_stm_only",
        "rag_condition": "sql_enterprise_rag_ltm",
        "expected_runs": 30,
        "max_budget": 50,
    },
    "file_moderate": {
        "db": "experiment-results/events_file_moderate.db",
        "experiment_id": "exp_2026-06-20_04-30-04-231642",
        "stm_condition": "file_moderate_stm_only",
        "rag_condition": "file_moderate_rag_ltm",
        "expected_runs": 30,
        "max_budget": 50,
    },
    "file_enterprise": {
        "db": "experiment-results/events_file_enterprise.db",
        "experiment_id": "exp_2026-06-20_13-24-10-978642",
        "stm_condition": "file_enterprise_stm_only",
        "rag_condition": "file_enterprise_rag_ltm",
        "expected_runs": 30,
        "max_budget": 50,
    },
}

PAIR_ORDER = ("sql_moderate", "sql_enterprise", "file_moderate", "file_enterprise")
RETRIEVAL_PHASES = ("planner_positive", "planner_blocked", "mutator_positive")
VALID_STOP_REASONS = {"breach_stop", "budget_exhausted", "error"}
Z_95 = 1.959963984540054
BLOCK_LABELS = {
    "sql_moderate": "SQL Moderate",
    "sql_enterprise": "SQL Enterprise",
    "file_moderate": "Datei Moderate",
    "file_enterprise": "Datei Enterprise",
}
PHASE_LABELS = {
    "planner_positive": "Planner: positive Episoden",
    "planner_blocked": "Planner: blockierte Muster",
    "mutator_positive": "Mutator: positive Episoden",
}
BLOCK_LABELS_EN: dict[str, str] = {
    "sql_moderate": "SQL Moderate",
    "sql_enterprise": "SQL Enterprise",
    "file_moderate": "File Moderate",
    "file_enterprise": "File Enterprise",
}
PHASE_LABELS_EN: dict[str, str] = {
    "planner_positive": "Planner: positive episodes",
    "planner_blocked": "Planner: blocked patterns",
    "mutator_positive": "Mutator: positive episodes",
}
_PLOT_LABELS: dict[str, dict[str, Any]] = {
    "de": {
        "block": BLOCK_LABELS,
        "phase": PHASE_LABELS,
        "risk_diff_xlabel": "Risikodifferenz RAG − STM (Prozentpunkte)",
        "step_xlabel": "Schritt $t$",
        "no_breach_ylabel": "$P(\\mathrm{kein\\ Breach\\ bis\\ }t)$",
        "all_censored": "alle Läufe zensiert",
        "q2b_budget_ylabel": "Q2B / Budgetgrenze",
        "censored_label": "zensiert",
        "retrieval_ylabel": "Retrievalabfragen mit Treffer (%)",
    },
    "en": {
        "block": BLOCK_LABELS_EN,
        "phase": PHASE_LABELS_EN,
        "risk_diff_xlabel": "Risk Difference RAG − STM (percentage points)",
        "step_xlabel": "Step $t$",
        "no_breach_ylabel": "$P(\\mathrm{no\\ breach\\ until\\ }t)$",
        "all_censored": "all runs censored",
        "q2b_budget_ylabel": "Q2B / Budget Limit",
        "censored_label": "censored",
        "retrieval_ylabel": "Retrieval queries with hit (%)",
    },
}
STM_COLOR = "#1f77b4"
RAG_COLOR = "#d55e00"
PHASE_COLORS = {
    "planner_positive": "#4c78a8",
    "planner_blocked": "#72b7b2",
    "mutator_positive": "#f58518",
}


@dataclass(frozen=True)
class SourceSpec:
    block: str
    db_path: Path
    experiment_id: str
    stm_condition: str
    rag_condition: str
    expected_runs: int = 30
    max_budget: int = 50

    @property
    def conditions(self) -> tuple[str, str]:
        return (self.stm_condition, self.rag_condition)

    @property
    def scenario(self) -> str:
        return self.block.split("_", 1)[0]

    @property
    def guardrail_level(self) -> str:
        return self.block.split("_", 1)[1]


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    block: str | None = None
    condition: str | None = None
    run_id: int | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "block": self.block,
            "condition": self.condition,
            "run_id": self.run_id,
            "seed": self.seed,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build MARTE-thesis-worktree thesis evaluation artifacts from final EventStore DBs."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for JSON, CSV, LaTeX tables, plots, and provenance.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("runtime/chroma"),
        help="Persistent Chroma path used for RAG LTM audit.",
    )
    parser.add_argument(
        "--copy-chroma-for-audit",
        action="store_true",
        help=(
            "Copy the Chroma directory to a writable temporary location before "
            "opening it. Recommended for final read-only audits."
        ),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Optional JSON source config overriding the built-in final DB mapping.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when validation contains any error.",
    )
    args = parser.parse_args(argv)

    sources = load_sources(args.source_config)
    chroma_path = _resolve_path(args.chroma_path)
    if args.copy_chroma_for_audit:
        with tempfile.TemporaryDirectory(prefix="marte-chroma-audit-") as tmpdir:
            audit_chroma_path = Path(tmpdir) / "chroma"
            copy_chroma_tree(chroma_path, audit_chroma_path)
            result = build_evaluation(
                sources=sources,
                output_dir=args.output_dir,
                chroma_path=chroma_path,
                audit_chroma_path=audit_chroma_path,
                chroma_audit_mode="temporary_copy",
            )
    else:
        result = build_evaluation(
            sources=sources,
            output_dir=args.output_dir,
            chroma_path=chroma_path,
            audit_chroma_path=chroma_path,
            chroma_audit_mode="direct",
        )
    _print_summary(result)

    if args.strict and result["validation"]["errors"] > 0:
        return 1
    return 0


def load_sources(config_path: Path | None = None) -> list[SourceSpec]:
    raw = DEFAULT_SOURCES if config_path is None else json.loads(config_path.read_text(encoding="utf-8"))
    sources: list[SourceSpec] = []
    for block in PAIR_ORDER:
        if block not in raw:
            raise ValueError(f"Source config is missing block {block!r}.")
        entry = raw[block]
        sources.append(
            SourceSpec(
                block=block,
                db_path=_resolve_path(Path(entry["db"])),
                experiment_id=str(entry["experiment_id"]),
                stm_condition=str(entry["stm_condition"]),
                rag_condition=str(entry["rag_condition"]),
                expected_runs=int(entry.get("expected_runs", 30)),
                max_budget=int(entry.get("max_budget", 50)),
            )
        )
    return sources


def build_evaluation(
        *,
        sources: list[SourceSpec],
        output_dir: Path,
        chroma_path: Path,
        audit_chroma_path: Path,
        chroma_audit_mode: str,
) -> dict[str, Any]:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_issues: list[ValidationIssue] = []
    manifest_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    ltm_rows: list[dict[str, Any]] = []
    oracle_replay_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    block_payloads: dict[str, dict[str, Any]] = {}

    for source in sources:
        payload = load_block_payload(source)
        block_payloads[source.block] = payload
        all_issues.extend(validate_block(source, payload))
        replay_rows, replay_issues = build_oracle_replay_rows(source, payload)
        oracle_replay_rows.extend(replay_rows)
        all_issues.extend(replay_issues)
        manifest_rows.append(build_manifest_row(source, payload))
        for condition in source.conditions:
            condition_rows.append(build_condition_metrics(source, payload, condition))
            fallback_rows.append(build_fallback_diagnostics(source, payload, condition))
        comparison_rows.append(build_pairwise_comparison(source, payload))
        retrieval_rows.extend(build_retrieval_rows(source, payload))
        ltm_audit_row, ltm_issues = audit_chroma_ltm(source, payload, audit_chroma_path)
        ltm_rows.append(ltm_audit_row)
        all_issues.extend(ltm_issues)

    all_issues.extend(validate_global_sources(sources, manifest_rows, ltm_rows))
    _apply_holm_adjustments(comparison_rows)

    validation = build_validation_report(all_issues)
    manifest = build_output_manifest(
        sources,
        manifest_rows,
        chroma_path,
        audit_chroma_path,
        chroma_audit_mode,
    )
    provenance = build_provenance(sources, manifest_rows, ltm_rows, chroma_audit_mode)

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "validation_report.json", validation)
    write_json(output_dir / "condition_metrics.json", condition_rows)
    write_json(output_dir / "pairwise_comparisons.json", comparison_rows)
    write_json(output_dir / "retrieval_mechanism.json", retrieval_rows)
    write_json(output_dir / "ltm_audit.json", ltm_rows)
    write_json(output_dir / "oracle_replay.json", oracle_replay_rows)
    write_json(output_dir / "fallback_diagnostics.json", fallback_rows)

    write_csv(output_dir / "condition_metrics.csv", condition_rows)
    write_csv(output_dir / "pairwise_comparisons.csv", comparison_rows)
    write_csv(output_dir / "retrieval_mechanism.csv", retrieval_rows)
    write_csv(output_dir / "ltm_audit.csv", ltm_rows)
    write_csv(output_dir / "source_manifests.csv", manifest_rows)
    write_csv(output_dir / "oracle_replay.csv", oracle_replay_rows)
    write_csv(output_dir / "fallback_diagnostics.csv", fallback_rows)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    write_condition_latex(tables_dir / "condition_metrics.tex", condition_rows)
    write_comparison_latex(tables_dir / "pairwise_comparisons.tex", comparison_rows)
    write_source_manifest_latex(tables_dir / "source_manifests.tex", manifest_rows)

    plots_dir = output_dir / "figures"
    plots_dir.mkdir(exist_ok=True)
    for lang in ("de", "en"):
        lang_plots_dir = plots_dir / lang
        lang_plots_dir.mkdir(exist_ok=True)
        plot_issues = write_plots(
            plots_dir=lang_plots_dir,
            block_payloads=block_payloads,
            condition_rows=condition_rows,
            comparison_rows=comparison_rows,
            retrieval_rows=retrieval_rows,
            lang=lang,
        )
        if plot_issues:
            all_issues.extend(plot_issues)
    validation = build_validation_report(all_issues)
    write_json(output_dir / "validation_report.json", validation)

    return {
        "validation": validation,
        "manifest": manifest,
        "condition_metrics": condition_rows,
        "pairwise_comparisons": comparison_rows,
        "retrieval_mechanism": retrieval_rows,
        "ltm_audit": ltm_rows,
        "oracle_replay": oracle_replay_rows,
        "fallback_diagnostics": fallback_rows,
    }


def load_block_payload(source: SourceSpec) -> dict[str, Any]:
    if not source.db_path.exists():
        raise FileNotFoundError(f"Database not found for {source.block}: {source.db_path}")

    with connect(source.db_path) as conn:
        manifest = conn.execute(
            """
            SELECT experiment_id, config_json, created_at_ms
            FROM experiment_manifests
            WHERE experiment_id = ?
            """,
            (source.experiment_id,),
        ).fetchone()
        all_stop_conditions = rows_to_dicts(
            conn.execute(
                """
                SELECT condition, COUNT(*) AS n
                FROM stop_events
                WHERE experiment_id = ?
                GROUP BY condition
                ORDER BY condition
                """,
                (source.experiment_id,),
            ).fetchall()
        )
        stops = rows_to_dicts(
            conn.execute(
                """
                SELECT *
                FROM stop_events
                WHERE experiment_id = ?
                  AND condition IN (?, ?)
                ORDER BY condition, seed, run_id
                """,
                (source.experiment_id, source.stm_condition, source.rag_condition),
            ).fetchall()
        )
        stop_run_ids = [int(row["run_id"]) for row in stops]
        run_events = rows_to_dicts(
            conn.execute(
                """
                SELECT *
                FROM run_events
                WHERE experiment_id = ?
                  AND condition IN (?, ?)
                ORDER BY event_id
                """,
                (source.experiment_id, source.stm_condition, source.rag_condition),
            ).fetchall()
        )
        step_events = rows_to_dicts(
            conn.execute(
                """
                SELECT *
                FROM step_events
                WHERE experiment_id = ?
                  AND condition IN (?, ?)
                ORDER BY run_id, step_t, event_id
                """,
                (source.experiment_id, source.stm_condition, source.rag_condition),
            ).fetchall()
        )

    manifest_dict = dict(manifest) if manifest is not None else None
    manifest_config = safe_json_loads(manifest_dict["config_json"], {}) if manifest_dict else {}
    return {
        "source": source,
        "manifest": manifest_dict,
        "manifest_config": manifest_config,
        "db_sha256": sha256_file(source.db_path),
        "db_size_bytes": source.db_path.stat().st_size,
        "all_stop_conditions": all_stop_conditions,
        "stops": stops,
        "run_events": run_events,
        "step_events": step_events,
        "stop_run_ids": stop_run_ids,
    }


def validate_block(source: SourceSpec, payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if payload["manifest"] is None:
        issues.append(issue("error", "missing_manifest", source, "Missing experiment manifest."))

    present_conditions = {row["condition"] for row in payload["all_stop_conditions"]}
    expected_conditions = set(source.conditions)
    for condition in sorted(expected_conditions - present_conditions):
        issues.append(
            issue("error", "missing_condition", source, f"Missing expected condition {condition}.", condition=condition)
        )
    for condition in sorted(present_conditions - expected_conditions):
        issues.append(
            issue(
                "error",
                "unexpected_condition",
                source,
                f"Unexpected condition {condition} in block DB.",
                condition=condition,
            )
        )

    stops = payload["stops"]
    valid_stops = [row for row in stops if row["stop_reason"] != "error"]
    for condition in source.conditions:
        condition_stops = [row for row in valid_stops if row["condition"] == condition]
        if len(condition_stops) != source.expected_runs:
            issues.append(
                issue(
                    "error",
                    "run_count",
                    source,
                    f"{condition} has {len(condition_stops)} valid stop_events; expected {source.expected_runs}.",
                    condition=condition,
                )
            )
        seeds = [int(row["seed"]) for row in condition_stops]
        seed_counts = count_values(seeds)
        expected_seed_set = set(range(1, source.expected_runs + 1))
        for seed, count in sorted(seed_counts.items()):
            if count > 1:
                issues.append(
                    issue(
                        "error",
                        "duplicate_seed",
                        source,
                        f"{condition} has duplicate seed {seed}.",
                        condition=condition,
                        seed=seed,
                    )
                )
        for seed in sorted(expected_seed_set - set(seeds)):
            issues.append(
                issue(
                    "error",
                    "missing_seed",
                    source,
                    f"{condition} is missing seed {seed}.",
                    condition=condition,
                    seed=seed,
                )
            )
        for seed in sorted(set(seeds) - expected_seed_set):
            issues.append(
                issue(
                    "error",
                    "unexpected_seed",
                    source,
                    f"{condition} has seed outside 1..{source.expected_runs}: {seed}.",
                    condition=condition,
                    seed=seed,
                )
            )

    issues.extend(validate_seed_pairing_and_order(source, valid_stops))
    issues.extend(validate_run_lifecycle(source, stops, payload["run_events"]))
    issues.extend(validate_stop_invariants(source, stops))
    issues.extend(validate_stop_step_consistency(source, stops, payload["step_events"]))
    issues.extend(validate_duplicate_stops(source, stops))
    issues.extend(validate_completed_step_failures(source, valid_stops, payload["step_events"]))
    issues.extend(validate_same_seed_retrieval(source, valid_stops, payload["step_events"]))
    return issues


def validate_seed_pairing_and_order(source: SourceSpec, valid_stops: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_condition_seed = {
        (row["condition"], int(row["seed"])): row
        for row in valid_stops
    }
    stm_seeds = {int(row["seed"]) for row in valid_stops if row["condition"] == source.stm_condition}
    rag_seeds = {int(row["seed"]) for row in valid_stops if row["condition"] == source.rag_condition}
    if stm_seeds != rag_seeds:
        issues.append(
            issue(
                "error",
                "seed_pairing",
                source,
                f"Seed sets differ between {source.stm_condition} and {source.rag_condition}.",
            )
        )

    successful_order = {
        int(row["run_id"]): index
        for index, row in enumerate(sorted(valid_stops, key=lambda item: int(item["run_id"])))
    }
    for seed in range(1, source.expected_runs + 1):
        stm = by_condition_seed.get((source.stm_condition, seed))
        rag = by_condition_seed.get((source.rag_condition, seed))
        if stm is None or rag is None:
            continue
        stm_run_id = int(stm["run_id"])
        rag_run_id = int(rag["run_id"])
        expected_first = source.stm_condition if seed % 2 == 1 else source.rag_condition
        actual_first = source.stm_condition if stm_run_id < rag_run_id else source.rag_condition
        if actual_first != expected_first:
            issues.append(
                issue(
                    "error",
                    "execution_order",
                    source,
                    f"Seed {seed} starts with {actual_first}; expected {expected_first}.",
                    condition=actual_first,
                    run_id=min(stm_run_id, rag_run_id),
                    seed=seed,
                )
            )
        if abs(successful_order[stm_run_id] - successful_order[rag_run_id]) != 1:
            issues.append(
                issue(
                    "error",
                    "execution_adjacency",
                    source,
                    f"Seed {seed} pair was not executed in adjacent successful runs.",
                    run_id=min(stm_run_id, rag_run_id),
                    seed=seed,
                )
            )
    return issues


def validate_run_lifecycle(
        source: SourceSpec,
        stops: list[dict[str, Any]],
        run_events: list[dict[str, Any]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not run_events:
        return [issue("warning", "missing_run_events", source, "No run_events found.")]

    stops_by_run: dict[int, list[dict[str, Any]]] = {}
    valid_keys = {
        (row["condition"], int(row["seed"]))
        for row in stops
        if row["stop_reason"] != "error"
    }
    for row in stops:
        stops_by_run.setdefault(int(row["run_id"]), []).append(row)

    latest_by_run: dict[int, dict[str, Any]] = {}
    for row in run_events:
        latest_by_run[int(row["run_id"])] = row

    for run_id, row in sorted(latest_by_run.items()):
        status = str(row["status"])
        condition = str(row["condition"])
        seed = int(row["seed"])
        if status == "started":
            level = "warning" if run_id in stops_by_run else "error"
            code = "open_lifecycle_with_stop" if run_id in stops_by_run else "open_run"
            issues.append(
                issue(level, code, source, f"Run {run_id} is still open.", condition=condition, run_id=run_id,
                      seed=seed)
            )
        elif status in {"aborted", "retry_aborted"}:
            recovered = (condition, seed) in valid_keys
            issues.append(
                issue(
                    "warning" if recovered else "error",
                    "recovered_run_attempt" if recovered else "unrecovered_run_attempt",
                    source,
                    (
                        f"Run attempt {run_id} ended as {status}"
                        f"{' and was superseded by a valid retry' if recovered else ''}: "
                        f"{row.get('error') or 'no error message'}"
                    ),
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )
        elif status == "completed" and run_id not in stops_by_run:
            issues.append(
                issue(
                    "error",
                    "completed_without_stop",
                    source,
                    f"Run {run_id} has completed lifecycle but no stop_event.",
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )
        elif status not in {"completed", "started", "aborted", "retry_aborted"}:
            issues.append(
                issue(
                    "error",
                    "unknown_run_status",
                    source,
                    f"Run {run_id} has unknown status {status!r}.",
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )

    for row in stops:
        run_id = int(row["run_id"])
        if run_id not in latest_by_run:
            issues.append(
                issue(
                    "warning",
                    "stop_without_run_event",
                    source,
                    f"Run {run_id} has stop_event but no lifecycle record.",
                    condition=row["condition"],
                    run_id=run_id,
                    seed=int(row["seed"]),
                )
            )
    return issues


def validate_stop_invariants(source: SourceSpec, stops: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for row in stops:
        stop_reason = str(row["stop_reason"])
        condition = str(row["condition"])
        run_id = int(row["run_id"])
        seed = int(row["seed"])
        q2b = int(row["q2b"])
        censored = bool(row["censored_flag"])
        if stop_reason not in VALID_STOP_REASONS:
            issues.append(
                issue("error", "unknown_stop_reason", source, f"Unknown stop_reason {stop_reason!r}.",
                      condition=condition, run_id=run_id, seed=seed)
            )
            continue
        if stop_reason == "error":
            issues.append(
                issue("error", "error_stop", source, "stop_reason='error' is not a valid evaluation run.",
                      condition=condition, run_id=run_id, seed=seed)
            )
            continue
        if q2b < 1:
            issues.append(
                issue("error", "invalid_q2b", source, f"q2b={q2b}; expected >= 1.", condition=condition, run_id=run_id,
                      seed=seed))
        if int(row["timestamp_ms"]) <= 0:
            issues.append(
                issue("error", "invalid_t2b_timestamp", source,
                      "timestamp_ms must be a positive run-relative duration.", condition=condition, run_id=run_id,
                      seed=seed)
            )
        if stop_reason == "breach_stop":
            if censored:
                issues.append(issue("error", "breach_censored", source, "breach_stop has censored_flag=True.",
                                    condition=condition, run_id=run_id, seed=seed))
            if q2b > source.max_budget:
                issues.append(issue("error", "breach_q2b_over_budget", source,
                                    f"q2b={q2b} exceeds max budget {source.max_budget}.", condition=condition,
                                    run_id=run_id, seed=seed))
        if stop_reason == "budget_exhausted":
            if not censored:
                issues.append(issue("error", "budget_not_censored", source, "budget_exhausted has censored_flag=False.",
                                    condition=condition, run_id=run_id, seed=seed))
            if q2b != source.max_budget:
                issues.append(issue("error", "budget_q2b_mismatch", source,
                                    f"budget_exhausted q2b={q2b}; expected {source.max_budget}.", condition=condition,
                                    run_id=run_id, seed=seed))
    return issues


def validate_stop_step_consistency(
        source: SourceSpec,
        stops: list[dict[str, Any]],
        step_events: list[dict[str, Any]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    steps_by_run: dict[int, list[dict[str, Any]]] = {}
    for row in step_events:
        steps_by_run.setdefault(int(row["run_id"]), []).append(row)

    for stop in stops:
        if stop["stop_reason"] == "error":
            continue
        run_id = int(stop["run_id"])
        condition = str(stop["condition"])
        seed = int(stop["seed"])
        q2b = int(stop["q2b"])
        steps = sorted(steps_by_run.get(run_id, []),
                       key=lambda item: (int(item["step_t"]), int(item.get("event_id") or 0)))
        if not steps:
            issues.append(issue("error", "missing_step_events", source, "Valid stop_event has no step_events.",
                                condition=condition, run_id=run_id, seed=seed))
            continue
        step_indices = [int(row["step_t"]) for row in steps]
        expected = list(range(1, q2b + 1)) if q2b >= 1 else []
        if step_indices != expected:
            issues.append(
                issue(
                    "error",
                    "step_sequence_mismatch",
                    source,
                    f"Step indices {format_values(step_indices)} do not match q2b={q2b}.",
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )
        mismatches = sorted({row["condition"] for row in steps if row["condition"] != condition})
        if mismatches:
            issues.append(
                issue("error", "step_condition_mismatch", source, f"Step condition mismatch: {', '.join(mismatches)}.",
                      condition=condition, run_id=run_id, seed=seed))
        last_step_ms = max(int(row["timestamp_ms"]) for row in steps)
        if int(stop["timestamp_ms"]) < last_step_ms:
            issues.append(
                issue("error", "stop_before_last_step", source, "Stop duration is before last step timestamp.",
                      condition=condition, run_id=run_id, seed=seed))
        oracle_steps = [int(row["step_t"]) for row in steps if bool(row["oracle_verdict"])]
        if stop["stop_reason"] == "breach_stop" and oracle_steps != [q2b]:
            issues.append(
                issue(
                    "error",
                    "breach_oracle_mismatch",
                    source,
                    f"breach_stop at q2b={q2b}, but positive Oracle steps are {format_values(oracle_steps)}.",
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )
        if stop["stop_reason"] == "budget_exhausted" and oracle_steps:
            issues.append(
                issue(
                    "error",
                    "censored_run_has_breach",
                    source,
                    f"Right-censored run has positive Oracle steps {format_values(oracle_steps)}.",
                    condition=condition,
                    run_id=run_id,
                    seed=seed,
                )
            )
    return issues


def validate_duplicate_stops(source: SourceSpec, stops: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    run_counts = count_values(int(row["run_id"]) for row in stops)
    for run_id, count in sorted(run_counts.items()):
        if count > 1:
            issues.append(
                issue("error", "duplicate_stop_run", source, f"Run {run_id} has {count} stop_events.", run_id=run_id))
    key_counts = count_values((str(row["condition"]), int(row["seed"])) for row in stops)
    for (condition, seed), count in sorted(key_counts.items()):
        if count > 1:
            issues.append(
                issue(
                    "error",
                    "duplicate_condition_seed",
                    source,
                    f"{condition} seed {seed} has {count} stop_events.",
                    condition=condition,
                    seed=seed,
                )
            )
    return issues


def validate_completed_step_failures(
        source: SourceSpec,
        valid_stops: list[dict[str, Any]],
        step_events: list[dict[str, Any]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    valid_run_ids = {int(row["run_id"]) for row in valid_stops}
    reported_backend: set[int] = set()
    reported_critical: set[int] = set()
    critical_prefixes = (
        "executor failed:",
        "target execution failed:",
        "pre snapshot failed:",
        "post snapshot failed:",
        "oracle failed:",
    )
    for row in step_events:
        run_id = int(row["run_id"])
        if run_id not in valid_run_ids:
            continue
        condition = str(row["condition"])
        errors = safe_json_loads(row.get("errors"), [])
        planner_debug = safe_json_loads(row.get("planner_debug"), {})
        judge_debug = safe_json_loads(row.get("judge_debug"), {})
        evidence = [
            str(row.get("response") or ""),
            str(planner_debug.get("error", "")) if isinstance(planner_debug, dict) else "",
            str(judge_debug.get("error", "")) if isinstance(judge_debug, dict) else "",
            *(str(item) for item in errors if isinstance(errors, list)),
        ]
        if run_id not in reported_backend:
            match = next((text for text in evidence if looks_like_backend_failure(text)), None)
            if match:
                reported_backend.add(run_id)
                issues.append(
                    issue(
                        "error",
                        "completed_backend_failure",
                        source,
                        f"Completed run contains backend failure evidence: {match[:180]}",
                        condition=condition,
                        run_id=run_id,
                    )
                )
        if run_id not in reported_critical:
            critical = next(
                (
                    str(item)
                    for item in errors
                    if str(item).strip().lower().startswith(critical_prefixes)
                ),
                None,
            )
            if critical:
                reported_critical.add(run_id)
                issues.append(
                    issue(
                        "error",
                        "completed_critical_phase_failure",
                        source,
                        f"Completed run contains critical phase failure: {critical[:180]}",
                        condition=condition,
                        run_id=run_id,
                    )
                )
    return issues


def validate_same_seed_retrieval(
        source: SourceSpec,
        valid_stops: list[dict[str, Any]],
        step_events: list[dict[str, Any]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seed_by_run = {int(row["run_id"]): int(row["seed"]) for row in valid_stops}
    reported = 0
    for row in step_events:
        if row["condition"] != source.rag_condition:
            continue
        run_id = int(row["run_id"])
        current_seed = seed_by_run.get(run_id)
        if current_seed is None:
            continue
        trace = safe_json_loads(row.get("retrieval_trace"), {})
        if not isinstance(trace, dict):
            continue
        for phase, batch in trace.items():
            if not isinstance(batch, dict):
                continue
            for hit in batch.get("hits", []) or []:
                episode = hit.get("episode", {}) if isinstance(hit, dict) else {}
                if not isinstance(episode, dict):
                    continue
                if int_or_none(episode.get("run_id")) == run_id:
                    reported += 1
                    if reported <= 10:
                        issues.append(
                            issue(
                                "error",
                                "same_seed_retrieval",
                                source,
                                f"RAG retrieved the current run in phase {phase}.",
                                condition=source.rag_condition,
                                run_id=run_id,
                                seed=current_seed,
                            )
                        )
    return issues


def validate_global_sources(
        sources: list[SourceSpec],
        manifest_rows: list[dict[str, Any]],
        ltm_rows: list[dict[str, Any]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    conditions: list[str] = []
    for source in sources:
        conditions.extend(source.conditions)
    duplicates = [condition for condition, count in count_values(conditions).items() if count > 1]
    if duplicates:
        issues.append(ValidationIssue("error", "duplicate_global_condition",
                                      f"Conditions appear multiple times: {', '.join(sorted(duplicates))}."))
    expected = {
        "sql_moderate_stm_only",
        "sql_moderate_rag_ltm",
        "sql_enterprise_stm_only",
        "sql_enterprise_rag_ltm",
        "file_moderate_stm_only",
        "file_moderate_rag_ltm",
        "file_enterprise_stm_only",
        "file_enterprise_rag_ltm",
    }
    missing = sorted(expected - set(conditions))
    if missing:
        issues.append(
            ValidationIssue("error", "missing_global_condition", f"Global matrix is missing: {', '.join(missing)}."))
    model_keys = {
        (
            row.get("attack_model"),
            row.get("judge_model"),
            row.get("target_model"),
            row.get("runs_per_condition"),
            row.get("max_budget"),
        )
        for row in manifest_rows
    }
    if len(model_keys) > 1:
        issues.append(ValidationIssue("warning", "manifest_model_or_budget_drift",
                                      "Manifest model or budget fields differ across sources."))
    fingerprints = {row.get("protocol_fingerprint") for row in manifest_rows if row.get("protocol_fingerprint")}
    if len(fingerprints) > 1:
        issues.append(ValidationIssue("warning", "protocol_fingerprint_drift",
                                      "Protocol fingerprints differ across blocks and must be disclosed."))
    for row in ltm_rows:
        if row.get("status") != "passed":
            issues.append(
                ValidationIssue(
                    "error",
                    "ltm_audit_failed",
                    f"LTM audit failed for {row.get('block')}: {row.get('status')}.",
                    block=str(row.get("block")),
                    condition=str(row.get("condition")),
                )
            )
    return issues


def build_condition_metrics(source: SourceSpec, payload: dict[str, Any], condition: str) -> dict[str, Any]:
    stops = sorted(
        [row for row in payload["stops"] if row["condition"] == condition and row["stop_reason"] != "error"],
        key=lambda row: int(row["seed"]),
    )
    n_total = len(stops)
    n_breach = sum(1 for row in stops if row["stop_reason"] == "breach_stop")
    n_censored = sum(1 for row in stops if bool(row["censored_flag"]))
    asr = safe_div(n_breach, n_total)
    ci_low, ci_high = wilson_ci(n_breach, n_total)
    q2b_dist = [(int(row["q2b"]), bool(row["censored_flag"])) for row in stops]
    t2b_dist = [(int(row["timestamp_ms"]), bool(row["censored_flag"])) for row in stops]
    retry_total = technical_retry_count(payload["step_events"], condition)
    return {
        "block": source.block,
        "condition": condition,
        "scenario": source.scenario,
        "guardrail_level": source.guardrail_level,
        "memory_mode": "rag_ltm" if condition == source.rag_condition else "stm_only",
        "n_total": n_total,
        "n_breach": n_breach,
        "n_censored": n_censored,
        "censored_rate": safe_div(n_censored, n_total),
        "asr": asr,
        "asr_ci_low": ci_low,
        "asr_ci_high": ci_high,
        "km_median_q2b": km_median(q2b_dist),
        "breach_prob_step_5": breach_probability_by(q2b_dist, 5),
        "breach_prob_step_10": breach_probability_by(q2b_dist, 10),
        "breach_prob_step_20": breach_probability_by(q2b_dist, 20),
        "breach_prob_step_50": breach_probability_by(q2b_dist, 50),
        "rmst_q2b_50": restricted_mean_survival(q2b_dist, 50),
        "km_median_t2b_minutes": ms_to_minutes(km_median(t2b_dist)),
        "technical_retries_total": retry_total,
    }


def build_pairwise_comparison(source: SourceSpec, payload: dict[str, Any]) -> dict[str, Any]:
    stm_stops = sorted(
        [row for row in payload["stops"] if row["condition"] == source.stm_condition and row["stop_reason"] != "error"],
        key=lambda row: int(row["seed"]),
    )
    rag_stops = sorted(
        [row for row in payload["stops"] if row["condition"] == source.rag_condition and row["stop_reason"] != "error"],
        key=lambda row: int(row["seed"]),
    )
    stm_breach = sum(1 for row in stm_stops if row["stop_reason"] == "breach_stop")
    rag_breach = sum(1 for row in rag_stops if row["stop_reason"] == "breach_stop")
    stm_n = len(stm_stops)
    rag_n = len(rag_stops)
    stm_asr = safe_div(stm_breach, stm_n)
    rag_asr = safe_div(rag_breach, rag_n)
    rd = None if stm_asr is None or rag_asr is None else rag_asr - stm_asr
    rd_low, rd_high = newcombe_risk_difference_ci(rag_breach, rag_n, stm_breach, stm_n)
    asr_or, asr_p = fisher_rag_vs_stm(rag_breach, rag_n, stm_breach, stm_n)
    q2b_p = logrank_p(
        [(int(row["q2b"]), bool(row["censored_flag"])) for row in stm_stops],
        [(int(row["q2b"]), bool(row["censored_flag"])) for row in rag_stops],
    )
    t2b_p = logrank_p(
        [(int(row["timestamp_ms"]), bool(row["censored_flag"])) for row in stm_stops],
        [(int(row["timestamp_ms"]), bool(row["censored_flag"])) for row in rag_stops],
    )
    return {
        "block": source.block,
        "stm_condition": source.stm_condition,
        "rag_condition": source.rag_condition,
        "stm_n": stm_n,
        "rag_n": rag_n,
        "stm_breach": stm_breach,
        "rag_breach": rag_breach,
        "stm_asr": stm_asr,
        "rag_asr": rag_asr,
        "risk_difference_rag_minus_stm": rd,
        "risk_difference_ci_low": rd_low,
        "risk_difference_ci_high": rd_high,
        "asr_odds_ratio_rag_vs_stm": asr_or,
        "asr_fisher_p": asr_p,
        "q2b_logrank_p": q2b_p,
        "t2b_logrank_p": t2b_p,
    }


def build_retrieval_rows(source: SourceSpec, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not source.rag_condition:
        return []
    stops = [row for row in payload["stops"] if
             row["condition"] == source.rag_condition and row["stop_reason"] != "error"]
    seed_by_run = {int(row["run_id"]): int(row["seed"]) for row in stops}
    accumulator = {
        phase: {
            "queries": 0,
            "with_candidates": 0,
            "with_hits": 0,
            "total_hits": 0,
            "positive_hits": 0,
            "blocked_hits": 0,
            "similarities": [],
            "same_seed_hits": 0,
        }
        for phase in RETRIEVAL_PHASES
    }
    for step in payload["step_events"]:
        if step["condition"] != source.rag_condition:
            continue
        current_seed = seed_by_run.get(int(step["run_id"]))
        trace = safe_json_loads(step.get("retrieval_trace"), {})
        if not isinstance(trace, dict):
            continue
        for phase in RETRIEVAL_PHASES:
            batch = trace.get(phase)
            if not isinstance(batch, dict):
                continue
            acc = accumulator[phase]
            hits = batch.get("hits", []) or []
            acc["queries"] += 1
            if int_or_none(batch.get("candidate_count")) and int(batch.get("candidate_count")) > 0:
                acc["with_candidates"] += 1
            if hits:
                acc["with_hits"] += 1
            acc["total_hits"] += len(hits)
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                similarity = float_or_none(hit.get("similarity"))
                if similarity is not None:
                    acc["similarities"].append(similarity)
                episode = hit.get("episode", {})
                if isinstance(episode, dict):
                    kind = str(episode.get("memory_kind", ""))
                    if kind == "positive":
                        acc["positive_hits"] += 1
                    elif kind == "blocked":
                        acc["blocked_hits"] += 1
                    if current_seed is not None and int_or_none(episode.get("run_id")) == int(step["run_id"]):
                        acc["same_seed_hits"] += 1

    rows: list[dict[str, Any]] = []
    for phase, acc in accumulator.items():
        similarities = acc.pop("similarities")
        queries = int(acc["queries"])
        total_hits = int(acc["total_hits"])
        rows.append(
            {
                "block": source.block,
                "condition": source.rag_condition,
                "phase": phase,
                "queries": queries,
                "queries_with_candidates": acc["with_candidates"],
                "queries_with_hits": acc["with_hits"],
                "candidate_query_rate": safe_div(acc["with_candidates"], queries),
                "hit_query_rate": safe_div(acc["with_hits"], queries),
                "total_hits": total_hits,
                "mean_hits_per_query": safe_div(total_hits, queries),
                "positive_hits": acc["positive_hits"],
                "blocked_hits": acc["blocked_hits"],
                "mean_similarity": mean_or_none(similarities),
                "min_similarity": min(similarities) if similarities else None,
                "max_similarity": max(similarities) if similarities else None,
                "same_seed_hits": acc["same_seed_hits"],
            }
        )
    return rows


def build_oracle_replay_rows(
        source: SourceSpec,
        payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    oracle = BreachOracle()
    policy = sandbox_policy_for_source(source)
    seed_by_run = seed_map(payload)
    rows: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    for step in payload["step_events"]:
        run_id = int(step["run_id"])
        step_t = int(step["step_t"])
        stored = bool(step["oracle_verdict"])
        row = {
            "block": source.block,
            "condition": step["condition"],
            "run_id": run_id,
            "seed": seed_by_run.get(run_id),
            "step_t": step_t,
            "event_id": step.get("event_id"),
            "stored_oracle_verdict": stored,
            "replayed_oracle_verdict": None,
            "tool_call_count": 0,
            "status": "error",
            "error": "",
        }
        try:
            pre_snapshot = SandboxSnapshot(**safe_json_loads(step.get("pre_snapshot"), {}))
            post_snapshot = SandboxSnapshot(**safe_json_loads(step.get("post_snapshot"), {}))
            tool_trace = [
                ToolCall.model_validate(item)
                for item in safe_json_loads(step.get("tool_trace"), [])
            ]
            replayed = bool(oracle.check(pre_snapshot, post_snapshot, tool_trace, policy))
            row["replayed_oracle_verdict"] = replayed
            row["tool_call_count"] = len(tool_trace)
            row["status"] = "passed" if replayed == stored else "mismatch"
        except Exception as exc:  # noqa: BLE001 - validation must report malformed persisted events.
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)
        if row["status"] != "passed":
            issues.append(
                issue(
                    "error",
                    "oracle_replay_mismatch" if row["status"] == "mismatch" else "oracle_replay_error",
                    source,
                    (
                        "Oracle replay did not match persisted step verdict"
                        if row["status"] == "mismatch"
                        else f"Oracle replay failed: {row['error']}"
                    ),
                    condition=str(step["condition"]),
                    run_id=run_id,
                    seed=seed_by_run.get(run_id),
                )
            )
    return rows, issues


def build_fallback_diagnostics(source: SourceSpec, payload: dict[str, Any], condition: str) -> dict[str, Any]:
    valid_run_ids = {
        int(row["run_id"])
        for row in payload["stops"]
        if row["condition"] == condition and row["stop_reason"] != "error"
    }
    phase_retries: dict[str, int] = {}
    steps_total = 0
    breach_steps = 0
    mutator_fallback_steps = 0
    mutator_fallback_breach_steps = 0
    judge_fallback_steps = 0
    judge_fallback_breach_steps = 0

    for step in payload["step_events"]:
        if step["condition"] != condition or int(step["run_id"]) not in valid_run_ids:
            continue
        steps_total += 1
        is_breach = bool(step["oracle_verdict"])
        breach_steps += int(is_breach)

        mutator_debug = safe_json_loads(step.get("mutator_debug"), {})
        judge_debug = safe_json_loads(step.get("judge_debug"), {})
        if isinstance(mutator_debug, dict) and bool(mutator_debug.get("mutator_used_fallback")):
            mutator_fallback_steps += 1
            mutator_fallback_breach_steps += int(is_breach)
        if isinstance(judge_debug, dict) and bool(judge_debug.get("judge_used_fallback")):
            judge_fallback_steps += 1
            judge_fallback_breach_steps += int(is_breach)

        timings = safe_json_loads(step.get("timings"), {})
        retries = timings.get("technical_retries") if isinstance(timings, dict) else None
        if isinstance(retries, dict):
            for phase, count in retries.items():
                phase_retries[str(phase)] = phase_retries.get(str(phase), 0) + int(count)

    technical_retries_total = sum(phase_retries.values())
    return {
        "block": source.block,
        "condition": condition,
        "memory_mode": "rag_ltm" if condition == source.rag_condition else "stm_only",
        "valid_run_count": len(valid_run_ids),
        "step_count": steps_total,
        "breach_step_count": breach_steps,
        "mutator_fallback_steps": mutator_fallback_steps,
        "mutator_fallback_breach_steps": mutator_fallback_breach_steps,
        "judge_fallback_steps": judge_fallback_steps,
        "judge_fallback_breach_steps": judge_fallback_breach_steps,
        "technical_retries_total": technical_retries_total,
        "technical_retries_by_phase": dict(sorted(phase_retries.items())),
    }


def sandbox_policy_for_source(source: SourceSpec) -> SandboxPolicy:
    if source.scenario == "file":
        return SandboxPolicy(scenario="file", allowed_paths=("public/",))
    return SandboxPolicy(scenario="sql", protected_tables=("credentials",), allowed_operations={})


def seed_map(payload: dict[str, Any]) -> dict[int, int]:
    seeds: dict[int, int] = {}
    for row in payload["stops"]:
        seeds[int(row["run_id"])] = int(row["seed"])
    for row in payload["run_events"]:
        seeds.setdefault(int(row["run_id"]), int(row["seed"]))
    return seeds


def audit_chroma_ltm(
        source: SourceSpec,
        payload: dict[str, Any],
        chroma_path: Path,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    expected_ids = expected_ltm_document_ids(source, payload)
    collection_name = ltm_scope_name(source.experiment_id, source.rag_condition)
    row: dict[str, Any] = {
        "block": source.block,
        "condition": source.rag_condition,
        "collection_name": collection_name,
        "expected_document_count": len(expected_ids),
        "actual_document_count": None,
        "missing_document_count": None,
        "extra_document_count": None,
        "status": "not_checked",
    }
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(collection_name)
        actual_ids = set(collection.get(include=[])["ids"])
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)
        issues.append(
            issue(
                "error",
                "ltm_collection_unavailable",
                source,
                f"Cannot read Chroma collection {collection_name}: {exc}",
                condition=source.rag_condition,
            )
        )
        return row, issues

    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    row.update(
        {
            "actual_document_count": len(actual_ids),
            "missing_document_count": len(missing),
            "extra_document_count": len(extra),
            "missing_sample": sorted(missing)[:5],
            "extra_sample": sorted(extra)[:5],
            "status": "passed" if not missing and not extra else "mismatch",
        }
    )
    if missing or extra:
        issues.append(
            issue(
                "error",
                "ltm_reconstruction_mismatch",
                source,
                f"Chroma IDs differ from completed-run reconstruction: missing={len(missing)}, extra={len(extra)}.",
                condition=source.rag_condition,
            )
        )
    return row, issues


def expected_ltm_document_ids(source: SourceSpec, payload: dict[str, Any]) -> set[str]:
    rag_stop_run_ids = {
        int(row["run_id"]): int(row["seed"])
        for row in payload["stops"]
        if row["condition"] == source.rag_condition and row["stop_reason"] != "error"
    }
    expected: set[str] = set()
    for row in payload["step_events"]:
        if row["condition"] != source.rag_condition:
            continue
        run_id = int(row["run_id"])
        seed = rag_stop_run_ids.get(run_id)
        if seed is None:
            continue
        episode_dict = {
            "oracle_verdict": bool(row["oracle_verdict"]),
            "judge_verdict": row["judge_verdict"],
            "input_decision": row.get("input_decision", ""),
            "input_reason": row.get("input_reason", ""),
            "policy_decisions": safe_json_loads(row.get("policy_decisions"), []),
            "tool_trace": safe_json_loads(row.get("tool_trace"), []),
            "prompt": row["prompt"],
        }
        classification = _classify_ltm_episode(episode_dict)
        if classification is None:
            continue
        memory_kind, policy_stage, policy_reason = classification
        mutation = str(row["prompt"])[:500]
        step_t = int(row["step_t"])
        if memory_kind == "blocked":
            document_id = _blocked_document_id(
                scenario=source.scenario,
                guardrail_level=source.guardrail_level,
                policy_stage=policy_stage,
                policy_reason=policy_reason,
                mutation=mutation,
            )
        else:
            document_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"marte:{source.scenario}:{source.guardrail_level}:"
                        f"{seed}:{step_t}:positive"
                    ),
                )
            )
        expected.add(document_id)
    return expected


def build_manifest_row(source: SourceSpec, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["manifest_config"]
    experiment_config = config.get("experiment_config", {}) if isinstance(config, dict) else {}
    role_models = experiment_config.get("role_models", {}) if isinstance(experiment_config, dict) else {}
    return {
        "block": source.block,
        "db_path": str(source.db_path),
        "db_sha256": payload["db_sha256"],
        "db_size_bytes": payload["db_size_bytes"],
        "experiment_id": source.experiment_id,
        "created_at_ms": payload["manifest"].get("created_at_ms") if payload["manifest"] else None,
        "runs_per_condition": config.get("runs_per_condition") if isinstance(config, dict) else None,
        "max_budget": config.get("max_budget") if isinstance(config, dict) else None,
        "protocol_fingerprint": experiment_config.get("protocol_fingerprint") if isinstance(experiment_config,
                                                                                            dict) else None,
        "attack_model": nested_get(role_models, ("attack", "model")),
        "judge_model": nested_get(role_models, ("judge", "model")),
        "target_model": nested_get(role_models, ("target", "model")),
    }


def build_output_manifest(
        sources: list[SourceSpec],
        manifest_rows: list[dict[str, Any]],
        chroma_path: Path,
        audit_chroma_path: Path,
        chroma_audit_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script": "scripts/build_thesis_evaluation.py",
        "sources": [
            {
                "block": source.block,
                "db": str(source.db_path),
                "experiment_id": source.experiment_id,
                "stm_condition": source.stm_condition,
                "rag_condition": source.rag_condition,
                "expected_runs": source.expected_runs,
                "max_budget": source.max_budget,
            }
            for source in sources
        ],
        "source_manifests": manifest_rows,
        "chroma_path": str(chroma_path),
        "chroma_audit_path": str(audit_chroma_path),
        "chroma_audit_mode": chroma_audit_mode,
        "planned_holm_families": {
            "asr": 4,
            "q2b": 4,
            "t2b": 4,
        },
    }


def build_provenance(
        sources: list[SourceSpec],
        manifest_rows: list[dict[str, Any]],
        ltm_rows: list[dict[str, Any]],
        chroma_audit_mode: str,
) -> dict[str, Any]:
    return {
        "source_count": len(sources),
        "condition_count": len({condition for source in sources for condition in source.conditions}),
        "manifests": manifest_rows,
        "ltm_audit": ltm_rows,
        "notes": [
            "Primary metrics use valid stop_events only.",
            "Recovered aborted attempts are documented as validation warnings, not excluded.",
            "T2B uses stop_events.timestamp_ms as run-relative duration.",
            "No pooled overall effect is reported.",
            f"Chroma LTM audit mode: {chroma_audit_mode}.",
        ],
    }


def build_validation_report(issues: list[ValidationIssue]) -> dict[str, Any]:
    errors = sum(1 for item in issues if item.level == "error")
    warnings = sum(1 for item in issues if item.level == "warning")
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
        "issues": [item.as_dict() for item in issues],
    }


def _apply_holm_adjustments(rows: list[dict[str, Any]]) -> None:
    families = {
        "asr": "asr_fisher_p",
        "q2b": "q2b_logrank_p",
        "t2b": "t2b_logrank_p",
    }
    for family, key in families.items():
        adjusted = holm_adjust_p_values([row[key] for row in rows], planned_m=len(rows))
        effective_m = sum(1 for row in rows if is_finite(row[key]))
        for row, adjusted_p in zip(rows, adjusted):
            row[f"{family}_holm_p"] = adjusted_p
            row[f"{family}_holm_planned_m"] = len(rows)
            row[f"{family}_holm_effective_m"] = effective_m
            row[f"{family}_holm_family_complete"] = effective_m == len(rows)
            row[f"{family}_holm_significant_0_05"] = bool(is_finite(adjusted_p) and adjusted_p <= 0.05)


def wilson_ci(successes: int, n_total: int, z: float = Z_95) -> tuple[float | None, float | None]:
    if n_total <= 0:
        return None, None
    p = successes / n_total
    denom = 1.0 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    half = z * math.sqrt((p * (1 - p) / n_total) + (z ** 2 / (4 * n_total ** 2))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_risk_difference_ci(
        rag_successes: int,
        rag_total: int,
        stm_successes: int,
        stm_total: int,
) -> tuple[float | None, float | None]:
    if rag_total <= 0 or stm_total <= 0:
        return None, None
    p_rag = rag_successes / rag_total
    p_stm = stm_successes / stm_total
    rag_low, rag_high = wilson_ci(rag_successes, rag_total)
    stm_low, stm_high = wilson_ci(stm_successes, stm_total)
    assert rag_low is not None and rag_high is not None
    assert stm_low is not None and stm_high is not None
    lower = (p_rag - p_stm) - math.sqrt((p_rag - rag_low) ** 2 + (stm_high - p_stm) ** 2)
    upper = (p_rag - p_stm) + math.sqrt((rag_high - p_rag) ** 2 + (p_stm - stm_low) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def fisher_rag_vs_stm(
        rag_successes: int,
        rag_total: int,
        stm_successes: int,
        stm_total: int,
) -> tuple[float, float]:
    if rag_total <= 0 or stm_total <= 0:
        return float("nan"), float("nan")
    table = [
        [rag_successes, rag_total - rag_successes],
        [stm_successes, stm_total - stm_successes],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    if (
            (rag_successes == 0 and stm_successes == 0)
            or (rag_successes == rag_total and stm_successes == stm_total)
    ):
        odds_ratio = float("nan")
    return float(odds_ratio), float(p_value)


def logrank_p(group_a: list[tuple[int, bool]], group_b: list[tuple[int, bool]]) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    durations_a = np.array([duration for duration, _ in group_a], dtype=float)
    events_a = np.array([not censored for _, censored in group_a], dtype=bool)
    durations_b = np.array([duration for duration, _ in group_b], dtype=float)
    events_b = np.array([not censored for _, censored in group_b], dtype=bool)
    if int(events_a.sum()) + int(events_b.sum()) == 0:
        return float("nan")
    result = logrank_test(
        durations_a,
        durations_b,
        event_observed_A=events_a,
        event_observed_B=events_b,
    )
    return float(result.p_value)


def holm_adjust_p_values(p_values: list[Any], planned_m: int | None = None) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        numeric = float(value) if value is not None else float("nan")
        if math.isfinite(numeric):
            finite.append((index, numeric))
    finite.sort(key=lambda item: item[1])
    m = planned_m if planned_m is not None else len(finite)
    if m < len(finite):
        raise ValueError("planned_m must be >= the number of finite p-values")
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(finite):
        running_max = max(running_max, min((m - rank) * p_value, 1.0))
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


def km_median(dist: list[tuple[int, bool]]) -> float | None:
    if not dist or not any(not censored for _, censored in dist):
        return None
    kmf = KaplanMeierFitter()
    kmf.fit([duration for duration, _ in dist], event_observed=[not censored for _, censored in dist])
    median = float(kmf.median_survival_time_)
    if not math.isfinite(median):
        return None
    return median


def breach_probability_by(dist: list[tuple[int, bool]], t: int) -> float | None:
    if not dist:
        return None
    if not any(not censored for _, censored in dist):
        return 0.0
    kmf = KaplanMeierFitter()
    kmf.fit([duration for duration, _ in dist], event_observed=[not censored for _, censored in dist])
    survival = float(kmf.predict(t))
    return 1.0 - survival


def restricted_mean_survival(dist: list[tuple[int, bool]], tau: int) -> float | None:
    if not dist:
        return None
    kmf = KaplanMeierFitter()
    kmf.fit([duration for duration, _ in dist], event_observed=[not censored for _, censored in dist])
    timeline = [0.0]
    timeline.extend(float(t) for t in sorted({duration for duration, _ in dist if duration < tau}))
    timeline.append(float(tau))
    area = 0.0
    for left, right in zip(timeline, timeline[1:]):
        area += float(kmf.predict(left)) * (right - left)
    return area


def technical_retry_count(step_events: list[dict[str, Any]], condition: str) -> int:
    total = 0
    for row in step_events:
        if row["condition"] != condition:
            continue
        timings = safe_json_loads(row.get("timings"), {})
        if not isinstance(timings, dict):
            continue
        retries = timings.get("technical_retries")
        if isinstance(retries, dict):
            total += sum(int(value) for value in retries.values() if isinstance(value, (int, float)))
    return total


def write_plots(
        *,
        plots_dir: Path,
        block_payloads: dict[str, dict[str, Any]],
        condition_rows: list[dict[str, Any]],
        comparison_rows: list[dict[str, Any]],
        retrieval_rows: list[dict[str, Any]],
        lang: str = "de",
) -> list[ValidationIssue]:
    try:
        cache_root = Path(tempfile.gettempdir()) / "marte-thesis-evaluation-cache"
        mpl_cache = cache_root / "matplotlib"
        xdg_cache = cache_root / "xdg"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return [ValidationIssue("warning", "plots_unavailable", f"Matplotlib plots skipped: {exc}")]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = _PLOT_LABELS.get(lang, _PLOT_LABELS["de"])
    plot_asr_forest(plt, plots_dir / "asr_risk_difference_forest.pdf", comparison_rows, labels)
    plot_q2b_km_panels(plt, plots_dir / "q2b_km_panels.pdf", block_payloads, labels)
    plot_run_order(plt, plots_dir / "run_order_q2b.pdf", block_payloads, labels)
    plot_retrieval_hitrate(plt, plots_dir / "retrieval_hitrate.pdf", retrieval_rows, labels)
    return []


def plot_asr_forest(plt: Any, path: Path, rows: list[dict[str, Any]], labels: dict[str, Any]) -> None:
    block_labels = labels["block"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y_labels = [block_labels.get(row["block"], row["block"]) for row in rows]
    y = np.arange(len(y_labels))
    x = np.array([(row["risk_difference_rag_minus_stm"] or 0.0) * 100 for row in rows], dtype=float)
    low = np.array(
        [
            (row["risk_difference_ci_low"] if row["risk_difference_ci_low"] is not None else 0.0) * 100
            for row in rows
        ]
    )
    high = np.array(
        [
            (row["risk_difference_ci_high"] if row["risk_difference_ci_high"] is not None else 0.0) * 100
            for row in rows
        ]
    )
    ax.errorbar(
        x,
        y,
        xerr=[x - low, high - x],
        fmt="o",
        color="black",
        ecolor="0.35",
        elinewidth=1.2,
        capsize=3.5,
        markersize=5,
    )
    ax.axvline(0.0, color="0.2", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(labels["risk_diff_xlabel"])
    ax.set_xlim(-30, 100)
    ax.set_xticks([-25, 0, 25, 50, 75, 100])
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)
    ax.invert_yaxis()
    for value, yy in zip(x, y):
        ax.text(value + 3.0, yy, f"{value:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_q2b_km_panels(plt: Any, path: Path, block_payloads: dict[str, dict[str, Any]], labels: dict[str, Any]) -> None:
    block_labels = labels["block"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), sharex=True, sharey=True)
    legend_handles = None
    legend_label_texts = None
    for ax, block in zip(axes.flatten(), PAIR_ORDER):
        payload = block_payloads[block]
        source: SourceSpec = payload["source"]
        for condition, label, color, linestyle in (
            (source.stm_condition, "STM", STM_COLOR, "-"),
            (source.rag_condition, "RAG-LTM", RAG_COLOR, "--"),
        ):
            stops = [row for row in payload["stops"] if row["condition"] == condition and row["stop_reason"] != "error"]
            if not stops:
                continue
            kmf = KaplanMeierFitter(label=label)
            kmf.fit(
                [int(row["q2b"]) for row in stops],
                event_observed=[not bool(row["censored_flag"]) for row in stops],
            )
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                show_censors=True,
                censor_styles={"marker": "|", "ms": 6, "mew": 1.2},
            )
        if legend_handles is None:
            legend_handles, legend_label_texts = ax.get_legend_handles_labels()
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        ax.set_title(block_labels.get(block, block), fontsize=11)
        ax.set_xlabel(labels["step_xlabel"])
        ax.set_ylabel(labels["no_breach_ylabel"])
        ax.set_xlim(0, 50)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.22)
        if _all_runs_censored(payload["stops"], source.conditions):
            ax.text(
                0.5,
                0.84,
                labels["all_censored"],
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.25",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
            )
    if legend_handles is not None and legend_label_texts is not None:
        fig.legend(legend_handles, legend_label_texts, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_run_order(plt: Any, path: Path, block_payloads: dict[str, dict[str, Any]], labels: dict[str, Any]) -> None:
    block_labels = labels["block"]
    censored_label = labels["censored_label"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), sharex=True, sharey=True)
    legend_entries: dict[str, Any] = {}
    for ax, block in zip(axes.flatten(), PAIR_ORDER):
        payload = block_payloads[block]
        source: SourceSpec = payload["source"]
        for condition, label, marker, color, offset in (
                (source.stm_condition, "STM", "o", STM_COLOR, -0.12),
                (source.rag_condition, "RAG-LTM", "s", RAG_COLOR, 0.12),
        ):
            stops = sorted(
                [row for row in payload["stops"] if row["condition"] == condition and row["stop_reason"] != "error"],
                key=lambda row: int(row["seed"]),
            )
            observed = [row for row in stops if not bool(row["censored_flag"])]
            censored = [row for row in stops if bool(row["censored_flag"])]
            if observed:
                handle = ax.scatter(
                    [int(row["seed"]) + offset for row in observed],
                    [int(row["q2b"]) for row in observed],
                    label=f"{label}: Breach",
                    marker=marker,
                    color=color,
                    s=28,
                    alpha=0.9,
                    linewidths=0.7,
                )
                legend_entries.setdefault(f"{label}: Breach", handle)
            if censored:
                handle = ax.scatter(
                    [int(row["seed"]) + offset for row in censored],
                    [int(row["q2b"]) for row in censored],
                    label=f"{label}: {censored_label}",
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    s=42,
                    alpha=0.95,
                    linewidths=1.3,
                )
                legend_entries.setdefault(f"{label}: {censored_label}", handle)
        ax.set_title(block_labels.get(block, block), fontsize=11)
        ax.set_xlabel(labels["step_xlabel"])
        ax.set_ylabel(labels["q2b_budget_ylabel"])
        ax.set_xlim(0.4, 30.6)
        ax.set_ylim(0, 52)
        ax.grid(alpha=0.22)
        if _all_runs_censored(payload["stops"], source.conditions):
            ax.text(
                0.5,
                0.82,
                labels["all_censored"],
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.25",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
            )
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_hitrate(plt: Any, path: Path, rows: list[dict[str, Any]], labels: dict[str, Any]) -> None:
    block_labels = labels["block"]
    phase_labels = labels["phase"]
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    block_positions = np.arange(len(PAIR_ORDER))
    width = 0.23
    phase_offsets = {
        "planner_positive": -width,
        "planner_blocked": 0.0,
        "mutator_positive": width,
    }
    rows_by_key = {(row["block"], row["phase"]): row for row in rows}
    for phase in RETRIEVAL_PHASES:
        values = [
            (rows_by_key.get((block, phase), {}).get("hit_query_rate") or 0.0) * 100
            for block in PAIR_ORDER
        ]
        ax.bar(
            block_positions + phase_offsets[phase],
            values,
            width=width,
            label=phase_labels.get(phase, phase),
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xticks(block_positions)
    ax.set_xticklabels([block_labels.get(block, block) for block in PAIR_ORDER])
    ax.set_ylabel(labels["retrieval_ylabel"])
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _all_runs_censored(stops: list[dict[str, Any]], conditions: tuple[str, str]) -> bool:
    valid_stops = [
        row
        for row in stops
        if row["condition"] in conditions and row["stop_reason"] != "error"
    ]
    return bool(valid_stops) and all(bool(row["censored_flag"]) for row in valid_stops)


def write_condition_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrlllr}",
        r"\toprule",
        r"Block & Modus & N & Breaches & Zens. & Zens. \% & ASR [95\%-KI] & Q2B & T2B [min] & Retr. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(BLOCK_LABELS.get(row["block"], row["block"])),
                    latex_escape(memory_mode_label(row["memory_mode"])),
                    str(row["n_total"]),
                    str(row["n_breach"]),
                    str(row["n_censored"]),
                    fmt_percent(row["censored_rate"]),
                    fmt_ci(row["asr"], row["asr_ci_low"], row["asr_ci_high"]),
                    fmt_median(row["km_median_q2b"]),
                    fmt_median(row["km_median_t2b_minutes"]),
                    str(row["technical_retries_total"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Block & RD [95\%-KI] & Fisher $p$ & ASR Holm & Q2B $p$ & Q2B Holm & T2B $p$ & T2B Holm & OR \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(BLOCK_LABELS.get(row["block"], row["block"])),
                    fmt_ci(
                        row["risk_difference_rag_minus_stm"],
                        row["risk_difference_ci_low"],
                        row["risk_difference_ci_high"],
                    ),
                    fmt_p(row["asr_fisher_p"]),
                    fmt_p(row["asr_holm_p"]),
                    fmt_p(row["q2b_logrank_p"]),
                    fmt_p(row["q2b_holm_p"]),
                    fmt_p(row["t2b_logrank_p"]),
                    fmt_p(row["t2b_holm_p"]),
                    fmt_float(row["asr_odds_ratio_rag_vs_stm"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_source_manifest_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Block & Experiment-ID & DB-SHA & Protocol-FP \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(BLOCK_LABELS.get(row["block"], row["block"])),
                    latex_escape(row["experiment_id"]),
                    latex_escape(short_hash(row["db_sha256"])),
                    latex_escape(short_hash(row["protocol_fingerprint"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_chroma_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Chroma path not found: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def issue(
        level: str,
        code: str,
        source: SourceSpec,
        message: str,
        *,
        condition: str | None = None,
        run_id: int | None = None,
        seed: int | None = None,
) -> ValidationIssue:
    return ValidationIssue(level, code, message, source.block, condition, run_id, seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ltm_scope_name(experiment_id: str, condition: str) -> str:
    digest = hashlib.sha1(f"{experiment_id}:{condition}".encode("utf-8")).hexdigest()[:12]
    condition_token = collection_token(condition)[:32] or "condition"
    return f"marte_{condition_token}_{digest}"


def collection_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not token:
        return "scope"
    if not token[0].isalnum():
        token = f"s{token}"
    if not token[-1].isalnum():
        token = f"{token}0"
    return token


def count_values(values: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def format_values(values: list[int], limit: int = 12) -> str:
    if not values:
        return "none"
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(str(value) for value in values[:limit]) + suffix


def safe_json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def nested_get(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def ms_to_minutes(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 60000.0


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def looks_like_backend_failure(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(
        marker in lowered
        for marker in (
            "[llm error]",
            "[target error]",
            "error code: 429",
            "rate limit",
            "failed to process request",
            "service unavailable",
            "connection reset",
            "connection refused",
            "connection error",
            "timed out",
            "timeout",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
    )


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_float(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return f"{numeric:.{decimals}f}"


def fmt_p(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def fmt_median(value: Any) -> str:
    if value is None:
        return "not reached"
    return fmt_float(value, 1)


def fmt_percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}\\%"


def fmt_ci(value: Any, low: Any, high: Any, decimals: int = 3) -> str:
    return f"{fmt_float(value, decimals)} [{fmt_float(low, decimals)}, {fmt_float(high, decimals)}]"


def memory_mode_label(value: Any) -> str:
    if value == "stm_only":
        return "STM"
    if value == "rag_ltm":
        return "RAG-LTM"
    return str(value)


def short_hash(value: Any, length: int = 12) -> str:
    text = str(value or "")
    return text[:length] if text else "n/a"


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(json_safe(value), sort_keys=True)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _print_summary(result: dict[str, Any]) -> None:
    validation = result["validation"]
    print(
        "Validation: "
        f"{validation['status']} "
        f"({validation['errors']} error(s), {validation['warnings']} warning(s))"
    )
    print("Pairwise comparisons:")
    for row in result["pairwise_comparisons"]:
        print(
            f"  {row['block']}: "
            f"RD={fmt_float(row['risk_difference_rag_minus_stm'])}, "
            f"Fisher p={fmt_p(row['asr_fisher_p'])}, "
            f"Q2B Holm={fmt_p(row['q2b_holm_p'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
