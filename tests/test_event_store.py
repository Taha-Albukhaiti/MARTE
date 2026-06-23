from __future__ import annotations

import sqlite3
from pathlib import Path

from marte.persistence.event_store import EventStore
from marte.persistence.schemas import ExperimentManifest, RunEvent, StepEvent, StopEvent


# ------------------------------------------------------------------
# Shared factories
# ------------------------------------------------------------------

def _step(
    experiment_id: str = "exp-1",
    run_id: int = 1,
    condition: str = "sql_moderate_stm_only",
    step_t: int = 1,
    oracle_verdict: bool = False,
    judge_verdict: str = "fail",
) -> StepEvent:
    return StepEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        condition=condition,
        step_t=step_t,
        timestamp_ms=1000,
        prompt="SELECT * FROM users",
        response="Error: access denied",
        tool_trace=[{"op": "SELECT", "table": "users"}],
        policy_decisions=["deny"],
        oracle_verdict=oracle_verdict,
        judge_verdict=judge_verdict,
    )


def _stop(
    experiment_id: str = "exp-1",
    run_id: int = 1,
    condition: str = "sql_moderate_stm_only",
    censored: bool = True,
    seed: int = 1,
    q2b: int = 50,
) -> StopEvent:
    return StopEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        condition=condition,
        stop_reason="budget_exhausted" if censored else "breach_stop",
        q2b=q2b,
        censored_flag=censored,
        timestamp_ms=5000,
        seed=seed,
    )


def _run_event(
    status: str = "started",
    experiment_id: str = "exp-1",
    run_id: int = 1,
    condition: str = "sql_moderate_stm_only",
    seed: int = 1,
    last_step_t: int = 0,
    stop_reason: str = "",
    error: str = "",
) -> RunEvent:
    return RunEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        condition=condition,
        seed=seed,
        status=status,
        timestamp_ms=123456,
        last_step_t=last_step_t,
        stop_reason=stop_reason,
        error=error,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_write_and_read_step_event(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_step_event(_step(step_t=3, oracle_verdict=True, judge_verdict="success"))

    results = store.get_step_events("exp-1")

    assert len(results) == 1
    ev = results[0]
    assert ev.experiment_id == "exp-1"
    assert ev.step_t == 3
    assert ev.oracle_verdict is True
    assert ev.judge_verdict == "success"
    assert ev.prompt == "SELECT * FROM users"
    assert ev.tool_trace == [{"op": "SELECT", "table": "users"}]


def test_write_and_read_stop_event(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_stop(censored=True, q2b=50, seed=7))

    results = store.get_stop_events("exp-1")

    assert len(results) == 1
    ev = results[0]
    assert ev.experiment_id == "exp-1"
    assert ev.stop_reason == "budget_exhausted"
    assert ev.censored_flag is True
    assert ev.q2b == 50
    assert ev.seed == 7


def test_write_and_read_run_lifecycle_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_run_event(_run_event(status="started"))
    store.write_run_event(
        _run_event(
            status="completed",
            last_step_t=3,
            stop_reason="budget_exhausted",
        )
    )

    events = store.get_run_events("exp-1")

    assert [ev.status for ev in events] == ["started", "completed"]
    assert events[1].last_step_t == 3
    assert events[1].stop_reason == "budget_exhausted"


def test_write_and_read_experiment_manifest(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_experiment_manifest(
        ExperimentManifest(
            experiment_id="exp-1",
            config={"max_budget": 50, "models": {"attack": "glm-4.7"}},
            created_at_ms=123456,
        )
    )

    manifest = store.get_experiment_manifest("exp-1")

    assert manifest is not None
    assert manifest.config == {
        "max_budget": 50,
        "models": {"attack": "glm-4.7"},
    }
    assert store.get_experiment_manifest("missing") is None


def test_open_run_events_return_latest_nonterminal_runs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_run_event(_run_event(experiment_id="exp-open", run_id=1, status="started"))
    store.write_run_event(_run_event(experiment_id="exp-done", run_id=1, status="started"))
    store.write_run_event(_run_event(experiment_id="exp-done", run_id=1, status="completed"))

    open_events = store.get_open_run_events()

    assert len(open_events) == 1
    assert open_events[0].experiment_id == "exp-open"
    assert open_events[0].status == "started"


def test_experiment_id_filter_isolates(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_stop(experiment_id="exp-A"))
    store.write_stop_event(_stop(experiment_id="exp-B"))
    store.write_stop_event(_stop(experiment_id="exp-B", run_id=2))

    a = store.get_stop_events("exp-A")
    b = store.get_stop_events("exp-B")

    assert len(a) == 1
    assert all(e.experiment_id == "exp-A" for e in a)
    assert len(b) == 2
    assert all(e.experiment_id == "exp-B" for e in b)


def test_censored_flag_correct(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_stop(run_id=1, censored=True))
    store.write_stop_event(_stop(run_id=2, censored=False))

    events = store.get_stop_events("exp-1")

    censored_flags = {e.run_id: e.censored_flag for e in events}
    assert censored_flags[1] is True
    assert censored_flags[2] is False
    assert events[1].stop_reason == "breach_stop"


def test_multiple_experiments_coexist(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    for i in range(3):
        store.write_stop_event(_stop(experiment_id=f"exp-{i}"))
        store.write_step_event(_step(experiment_id=f"exp-{i}"))

    for i in range(3):
        stops = store.get_stop_events(f"exp-{i}")
        steps = store.get_step_events(f"exp-{i}")
        assert len(stops) == 1
        assert len(steps) == 1


def test_db_file_persists(tmp_path: Path) -> None:
    """Data written by one EventStore instance survives after it goes out of scope."""
    db_path = tmp_path / "events.db"

    store1 = EventStore(db_path)
    store1.write_stop_event(_stop())
    store1.write_step_event(_step())
    del store1  # connection closed

    assert db_path.exists(), "SQLite file must be on disk, not in-memory"

    store2 = EventStore(db_path)
    assert len(store2.get_stop_events("exp-1")) == 1
    assert len(store2.get_step_events("exp-1")) == 1


def test_get_step_events_filtered_by_run_id(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_step_event(_step(run_id=1, step_t=1))
    store.write_step_event(_step(run_id=1, step_t=2))
    store.write_step_event(_step(run_id=2, step_t=1))

    run1 = store.get_step_events("exp-1", run_id=1)
    run2 = store.get_step_events("exp-1", run_id=2)
    all_steps = store.get_step_events("exp-1")

    assert len(run1) == 2
    assert len(run2) == 1
    assert len(all_steps) == 3


def test_step_event_audit_fields_roundtrip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = _step()
    event = event.model_copy(
        update={
            "memory_context": {"stm_recent": [{"step_t": 1}], "ltm_hits": []},
            "plan": {"strategy": "projection_probe"},
            "pre_snapshot": {"db_hash": "before", "fs_hash": "same"},
            "post_snapshot": {"db_hash": "after", "fs_hash": "same"},
            "input_decision": "allow",
            "output_decision": "allow",
            "input_reason": "",
            "output_reason": "response leaks password_hash",
            "retrieval_trace": {
                "planner_positive": {
                    "phase": "planner_positive",
                    "hits": [],
                }
            },
            "timings": {"planner_ms": 12.5, "target_ms": 40.0},
            "errors": ["planner fallback"],
            "planner_debug": {"raw_response": "{...}", "selected_strategy": "probe"},
            "mutator_debug": {"raw_response": "SELECT ...", "used_fallback": False},
            "judge_debug": {"rationale": "useful probe", "confidence": 0.8},
        }
    )

    store.write_step_event(event)

    [stored] = store.get_step_events("exp-1")
    assert stored.memory_context == {"stm_recent": [{"step_t": 1}], "ltm_hits": []}
    assert stored.plan == {"strategy": "projection_probe"}
    assert stored.pre_snapshot == {"db_hash": "before", "fs_hash": "same"}
    assert stored.post_snapshot == {"db_hash": "after", "fs_hash": "same"}
    assert stored.input_decision == "allow"
    assert stored.output_decision == "allow"
    assert stored.input_reason == ""
    assert stored.output_reason == "response leaks password_hash"
    assert stored.retrieval_trace["planner_positive"]["phase"] == (
        "planner_positive"
    )
    assert stored.timings == {"planner_ms": 12.5, "target_ms": 40.0}
    assert stored.errors == ["planner fallback"]
    assert stored.planner_debug["selected_strategy"] == "probe"
    assert stored.mutator_debug["used_fallback"] is False
    assert stored.judge_debug["rationale"] == "useful probe"


def test_legacy_step_events_schema_is_migrated(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE step_events (
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
                judge_verdict    TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE stop_events (
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
        )
        conn.execute(
            """
            INSERT INTO step_events
                (experiment_id, run_id, condition, step_t, timestamp_ms,
                 prompt, response, tool_trace, policy_decisions, oracle_verdict, judge_verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "exp-legacy",
                1,
                "sql_moderate_stm_only",
                1,
                100,
                "prompt",
                "response",
                "[]",
                "[]",
                0,
                "fail",
            ),
        )

    store = EventStore(db_path)
    [event] = store.get_step_events("exp-legacy")

    assert event.memory_context == {}
    assert event.plan == {}
    assert event.pre_snapshot == {}
    assert event.post_snapshot == {}
    assert event.input_decision == ""
    assert event.output_decision == ""
    assert event.input_reason == ""
    assert event.output_reason == ""
    assert event.retrieval_trace == {}
    assert event.timings == {}
    assert event.errors == []
    assert event.planner_debug == {}
    assert event.mutator_debug == {}
    assert event.judge_debug == {}
    assert store.get_run_events("exp-legacy") == []
