"""SQLite row mapping and JSON encoding for the EventStore."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from marte.persistence.schemas import RunEvent, StepEvent, StopEvent

_STEP_EVENT_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("memory_context", "TEXT NOT NULL DEFAULT '{}'"),
    ("plan", "TEXT NOT NULL DEFAULT '{}'"),
    ("pre_snapshot", "TEXT NOT NULL DEFAULT '{}'"),
    ("post_snapshot", "TEXT NOT NULL DEFAULT '{}'"),
    ("input_decision", "TEXT NOT NULL DEFAULT ''"),
    ("output_decision", "TEXT NOT NULL DEFAULT ''"),
    ("input_reason", "TEXT NOT NULL DEFAULT ''"),
    ("output_reason", "TEXT NOT NULL DEFAULT ''"),
    ("retrieval_trace", "TEXT NOT NULL DEFAULT '{}'"),
    ("timings", "TEXT NOT NULL DEFAULT '{}'"),
    ("errors", "TEXT NOT NULL DEFAULT '[]'"),
    ("planner_debug", "TEXT NOT NULL DEFAULT '{}'"),
    ("mutator_debug", "TEXT NOT NULL DEFAULT '{}'"),
    ("judge_debug", "TEXT NOT NULL DEFAULT '{}'"),
)


def _row_to_stop_event(row: sqlite3.Row) -> StopEvent:
    return StopEvent(
        experiment_id=row["experiment_id"],
        run_id=row["run_id"],
        condition=row["condition"],
        stop_reason=row["stop_reason"],
        q2b=row["q2b"],
        censored_flag=bool(row["censored_flag"]),
        timestamp_ms=row["timestamp_ms"],
        seed=row["seed"],
    )


def _row_to_run_event(row: sqlite3.Row) -> RunEvent:
    return RunEvent(
        event_id=row["event_id"],
        experiment_id=row["experiment_id"],
        run_id=row["run_id"],
        condition=row["condition"],
        seed=row["seed"],
        status=row["status"],
        timestamp_ms=row["timestamp_ms"],
        last_step_t=row["last_step_t"],
        stop_reason=row["stop_reason"],
        error=row["error"],
    )


def _row_to_step_event(row: sqlite3.Row) -> StepEvent:
    return StepEvent(
        event_id=row["event_id"],
        experiment_id=row["experiment_id"],
        run_id=row["run_id"],
        condition=row["condition"],
        step_t=row["step_t"],
        timestamp_ms=row["timestamp_ms"],
        prompt=row["prompt"],
        response=row["response"],
        tool_trace=_load_json(row["tool_trace"]),
        policy_decisions=_load_json(row["policy_decisions"]),
        oracle_verdict=bool(row["oracle_verdict"]),
        judge_verdict=row["judge_verdict"],
        memory_context=_load_json(_row_value(row, "memory_context", "{}")),
        plan=_load_json(_row_value(row, "plan", "{}")),
        pre_snapshot=_load_json(_row_value(row, "pre_snapshot", "{}")),
        post_snapshot=_load_json(_row_value(row, "post_snapshot", "{}")),
        input_decision=str(_row_value(row, "input_decision", "")),
        output_decision=str(_row_value(row, "output_decision", "")),
        input_reason=str(_row_value(row, "input_reason", "")),
        output_reason=str(_row_value(row, "output_reason", "")),
        retrieval_trace=_load_json(_row_value(row, "retrieval_trace", "{}")),
        timings=_load_json(_row_value(row, "timings", "{}")),
        errors=_load_json(_row_value(row, "errors", "[]")),
        planner_debug=_load_json(_row_value(row, "planner_debug", "{}")),
        mutator_debug=_load_json(_row_value(row, "mutator_debug", "{}")),
        judge_debug=_load_json(_row_value(row, "judge_debug", "{}")),
    )


def _ensure_step_event_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(step_events)")}
    for column_name, column_spec in _STEP_EVENT_EXTRA_COLUMNS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE step_events ADD COLUMN {column_name} {column_spec}")


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_value(row: sqlite3.Row, key: str, default: Any) -> Any:
    if key in row.keys():
        return row[key]
    return default
