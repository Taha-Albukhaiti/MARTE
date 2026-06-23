from __future__ import annotations

import sqlite3
from pathlib import Path

from marte.persistence.sqlite_codec import (
    _dump_json,
    _ensure_step_event_columns,
    _load_json,
    _row_to_run_event,
    _row_to_step_event,
    _row_to_stop_event,
)
from marte.persistence.schemas import ExperimentManifest, RunEvent, StepEvent, StopEvent

_CREATE_STEP_EVENTS = """
CREATE TABLE IF NOT EXISTS step_events (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    TEXT    NOT NULL,
    run_id           INTEGER NOT NULL,
    condition        TEXT    NOT NULL,
    step_t           INTEGER NOT NULL,
    timestamp_ms     INTEGER NOT NULL,
    prompt           TEXT    NOT NULL,
    response         TEXT    NOT NULL,
    tool_trace       TEXT    NOT NULL,
    policy_decisions TEXT    NOT NULL,
    oracle_verdict   INTEGER NOT NULL,
    judge_verdict    TEXT    NOT NULL,
    memory_context   TEXT    NOT NULL DEFAULT '{}',
    plan             TEXT    NOT NULL DEFAULT '{}',
    pre_snapshot     TEXT    NOT NULL DEFAULT '{}',
    post_snapshot    TEXT    NOT NULL DEFAULT '{}',
    input_decision   TEXT    NOT NULL DEFAULT '',
    output_decision  TEXT    NOT NULL DEFAULT '',
    input_reason     TEXT    NOT NULL DEFAULT '',
    output_reason    TEXT    NOT NULL DEFAULT '',
    retrieval_trace  TEXT    NOT NULL DEFAULT '{}',
    timings          TEXT    NOT NULL DEFAULT '{}',
    errors           TEXT    NOT NULL DEFAULT '[]',
    planner_debug    TEXT    NOT NULL DEFAULT '{}',
    mutator_debug    TEXT    NOT NULL DEFAULT '{}',
    judge_debug      TEXT    NOT NULL DEFAULT '{}'
)
"""

_CREATE_STOP_EVENTS = """
CREATE TABLE IF NOT EXISTS stop_events (
    experiment_id TEXT    NOT NULL,
    run_id        INTEGER NOT NULL,
    condition     TEXT    NOT NULL,
    stop_reason   TEXT    NOT NULL,
    q2b           INTEGER NOT NULL,
    censored_flag INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    seed          INTEGER NOT NULL
)
"""

_CREATE_RUN_EVENTS = """
CREATE TABLE IF NOT EXISTS run_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT    NOT NULL,
    run_id        INTEGER NOT NULL,
    condition     TEXT    NOT NULL,
    seed          INTEGER NOT NULL,
    status        TEXT    NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    last_step_t   INTEGER NOT NULL DEFAULT 0,
    stop_reason   TEXT    NOT NULL DEFAULT '',
    error         TEXT    NOT NULL DEFAULT ''
)
"""

_CREATE_EXPERIMENT_MANIFESTS = """
CREATE TABLE IF NOT EXISTS experiment_manifests (
    experiment_id TEXT PRIMARY KEY,
    config_json   TEXT    NOT NULL,
    created_at_ms INTEGER NOT NULL
)
"""

_INSERT_STEP = """
INSERT INTO step_events
    (experiment_id, run_id, condition, step_t, timestamp_ms,
     prompt, response, tool_trace, policy_decisions, oracle_verdict, judge_verdict,
     memory_context, plan, pre_snapshot, post_snapshot, input_decision, output_decision,
     input_reason, output_reason, retrieval_trace, timings, errors,
     planner_debug, mutator_debug, judge_debug)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_STOP = """
INSERT INTO stop_events
    (experiment_id, run_id, condition, stop_reason, q2b, censored_flag, timestamp_ms, seed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_RUN = """
INSERT INTO run_events
    (experiment_id, run_id, condition, seed, status, timestamp_ms,
     last_step_t, stop_reason, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_EXPERIMENT_MANIFEST = """
INSERT INTO experiment_manifests
    (experiment_id, config_json, created_at_ms)
VALUES (?, ?, ?)
"""

_SELECT_STEPS_ALL = (
    "SELECT * FROM step_events WHERE experiment_id = ? ORDER BY event_id"
)
_SELECT_STEPS_BY_RUN = (
    "SELECT * FROM step_events WHERE experiment_id = ? AND run_id = ? ORDER BY event_id"
)
_SELECT_STOPS = (
    "SELECT * FROM stop_events WHERE experiment_id = ? ORDER BY rowid"
)
_SELECT_RUNS_ALL = "SELECT * FROM run_events ORDER BY event_id"
_SELECT_RUNS_BY_EXPERIMENT = (
    "SELECT * FROM run_events WHERE experiment_id = ? ORDER BY event_id"
)


class EventStore:
    """Append-only SQLite store for step_events, stop_events, and run_events.

    The database file is created on first use and persists across program
    restarts. Multiple experiments can coexist in the same file; every query
    is scoped by ``experiment_id``.

    Args:
        db_path: Filesystem path to the SQLite database file. Parent
            directories are created if they do not exist.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Write API (append-only — no UPDATE, no DELETE)
    # ------------------------------------------------------------------

    def write_step_event(self, event: StepEvent) -> None:
        """Append one step record to the step_events table.

        Args:
            event: Fully populated StepEvent (event_id may be None).
        """
        with self._connect() as conn:
            conn.execute(
                _INSERT_STEP,
                (
                    event.experiment_id,
                    event.run_id,
                    event.condition,
                    event.step_t,
                    event.timestamp_ms,
                    event.prompt,
                    event.response,
                    _dump_json(event.tool_trace),
                    _dump_json(event.policy_decisions),
                    int(event.oracle_verdict),
                    event.judge_verdict,
                    _dump_json(event.memory_context),
                    _dump_json(event.plan),
                    _dump_json(event.pre_snapshot),
                    _dump_json(event.post_snapshot),
                    event.input_decision,
                    event.output_decision,
                    event.input_reason,
                    event.output_reason,
                    _dump_json(event.retrieval_trace),
                    _dump_json(event.timings),
                    _dump_json(event.errors),
                    _dump_json(event.planner_debug),
                    _dump_json(event.mutator_debug),
                    _dump_json(event.judge_debug),
                ),
            )

    def write_stop_event(self, event: StopEvent) -> None:
        """Append one run-termination record to the stop_events table.

        Args:
            event: Fully populated StopEvent.
        """
        with self._connect() as conn:
            conn.execute(
                _INSERT_STOP,
                (
                    event.experiment_id,
                    event.run_id,
                    event.condition,
                    event.stop_reason,
                    event.q2b,
                    int(event.censored_flag),
                    event.timestamp_ms,
                    event.seed,
                ),
            )

    def write_run_event(self, event: RunEvent) -> None:
        """Append one run lifecycle record to the run_events table."""
        with self._connect() as conn:
            conn.execute(
                _INSERT_RUN,
                (
                    event.experiment_id,
                    event.run_id,
                    event.condition,
                    event.seed,
                    event.status,
                    event.timestamp_ms,
                    event.last_step_t,
                    event.stop_reason,
                    event.error,
                ),
            )

    def write_experiment_manifest(self, manifest: ExperimentManifest) -> None:
        """Record the immutable configuration for one experiment ID."""
        with self._connect() as conn:
            conn.execute(
                _INSERT_EXPERIMENT_MANIFEST,
                (
                    manifest.experiment_id,
                    _dump_json(manifest.config),
                    manifest.created_at_ms,
                ),
            )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_stop_events(self, experiment_id: str) -> list[StopEvent]:
        """Return all stop_events for the given experiment, in insertion order.

        Args:
            experiment_id: Experiment to filter by.

        Returns:
            List of StopEvent objects.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(_SELECT_STOPS, (experiment_id,))
            return [_row_to_stop_event(row) for row in cursor.fetchall()]

    def get_run_events(self, experiment_id: str | None = None) -> list[RunEvent]:
        """Return run lifecycle events, optionally scoped by experiment."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if experiment_id is None:
                cursor = conn.execute(_SELECT_RUNS_ALL)
            else:
                cursor = conn.execute(_SELECT_RUNS_BY_EXPERIMENT, (experiment_id,))
            return [_row_to_run_event(row) for row in cursor.fetchall()]

    def get_experiment_manifest(
        self,
        experiment_id: str,
    ) -> ExperimentManifest | None:
        """Return the fixed experiment configuration, if one was recorded."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiment_manifests WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return ExperimentManifest(
            experiment_id=row["experiment_id"],
            config=_load_json(row["config_json"]),
            created_at_ms=row["created_at_ms"],
        )

    def get_open_run_events(self, experiment_id: str | None = None) -> list[RunEvent]:
        """Return latest lifecycle events for runs that have no terminal marker."""
        latest: dict[tuple[str, int], RunEvent] = {}
        for event in self.get_run_events(experiment_id):
            latest[(event.experiment_id, event.run_id)] = event
        return [
            event
            for event in latest.values()
            if event.status not in {"completed", "aborted", "retry_aborted"}
        ]

    def get_step_events(self, experiment_id: str, run_id: int | None = None) -> list[StepEvent]:
        """Return step_events for the given experiment, optionally filtered by run.

        Args:
            experiment_id: Experiment to filter by.
            run_id: If provided, only return events for this run.

        Returns:
            List of StepEvent objects in insertion order.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if run_id is not None:
                cursor = conn.execute(_SELECT_STEPS_BY_RUN, (experiment_id, run_id))
            else:
                cursor = conn.execute(_SELECT_STEPS_ALL, (experiment_id,))
            return [_row_to_step_event(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_STEP_EVENTS)
            conn.execute(_CREATE_STOP_EVENTS)
            conn.execute(_CREATE_RUN_EVENTS)
            conn.execute(_CREATE_EXPERIMENT_MANIFESTS)
            _ensure_step_event_columns(conn)
