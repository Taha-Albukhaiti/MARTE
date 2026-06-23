from __future__ import annotations

from pathlib import Path

import pytest
from marte.attack.orchestrator import RunResult
from marte.config.snapshot import ConfigSnapshot
from marte.experiment.runner import ComparativeRunner, ExperimentInterruptedError
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import RunEvent

# ------------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------------

class _SpyGateway:
    """Records how many times reset_ltm() was called."""

    def __init__(self) -> None:
        self.reset_ltm_count = 0
        self.ltm_scopes: list[str] = []
        self.reset_scopes: list[str] = []
        self.current_scope = ""
        self.begin_count = 0
        self.commit_count = 0
        self.rollback_count = 0

    def reset_ltm(self) -> None:
        self.reset_ltm_count += 1
        self.reset_scopes.append(self.current_scope)

    def set_ltm_scope(self, scope_name: str) -> None:
        self.current_scope = scope_name
        self.ltm_scopes.append(scope_name)

    def begin_run(self) -> None:
        self.begin_count += 1

    def commit_run(self) -> None:
        self.commit_count += 1

    def rollback_run(self) -> None:
        self.rollback_count += 1


def _fixed_result() -> RunResult:
    """Minimal RunResult returned by the spy run_callback."""
    return RunResult(
        stop_reason="budget_exhausted",
        q2b=5,
        censored_flag=True,
        budget_used=5,
        step_events=[],
        stop_event={"timestamp_ms": 1000},
        run_state={},
    )


def _raw_step() -> dict:
    return {
        "step_index": 1,
        "timestamp_ms": 25,
        "memory_context": {"stm_recent": [{"step_t": 0}], "ltm_hits": []},
        "prompt": "probe",
        "response_text": "denied",
        "plan": {"strategy": "probe"},
        "planner_raw_response": '{"strategy":"probe"}',
        "planner_error": None,
        "planner_selected_strategy": "probe",
        "mutator_raw_response": "probe",
        "mutator_used_fallback": False,
        "mutator_fallback_reason": None,
        "oracle_verdict": False,
        "judge_verdict": "fail",
        "pre_snapshot": {"db_hash": "before", "fs_hash": "same"},
        "post_snapshot": {"db_hash": "after", "fs_hash": "same"},
        "tool_trace": [{"tool": "sql", "operation": "SELECT"}],
        "policy_decisions": ["allow"],
        "input_decision": "allow",
        "output_decision": "allow",
        "timings": {"planner_ms": 1.5},
        "errors": ["oracle fallback"],
    }


def _make_runner(
    tmp_path: Path,
    spy_gateway: _SpyGateway | None = None,
    captured_configs: list[ConfigSnapshot] | None = None,
    prepared_configs: list[ConfigSnapshot] | None = None,
) -> tuple[ComparativeRunner, EventStore]:
    store = EventStore(tmp_path / "events.db")
    gateway = spy_gateway or _SpyGateway()

    def prepare_run(config: ConfigSnapshot) -> None:
        if prepared_configs is not None:
            prepared_configs.append(config)

    def run_callback(config: ConfigSnapshot, step_event_sink=None) -> RunResult:
        if captured_configs is not None:
            captured_configs.append(config)
        return _fixed_result()

    runner = ComparativeRunner(
        prepare_run=prepare_run,
        run_callback=run_callback,
        memory_gateway=gateway,
        event_store=store,
    )
    return runner, store


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_8_conditions_produce_correct_stop_events(tmp_path: Path) -> None:
    """8 conditions × 2 runs → 16 stop_events in EventStore."""
    runner, store = _make_runner(tmp_path)
    exp_id = runner.run_experiment(runs_per_condition=2, max_budget=5)

    events = store.get_stop_events(exp_id)
    assert len(events) == 8 * 2


def test_reset_ltm_called_per_condition_not_per_run(tmp_path: Path) -> None:
    """reset_ltm() must fire exactly 8 times — once per condition, not per run."""
    spy = _SpyGateway()
    runner, _ = _make_runner(tmp_path, spy_gateway=spy)
    runner.run_experiment(runs_per_condition=3, max_budget=5)

    assert spy.reset_ltm_count == 8
    assert len(spy.reset_scopes) == 8
    assert len(set(spy.reset_scopes)) == 8


def test_memory_mode_order_is_counterbalanced_within_each_seed(tmp_path: Path) -> None:
    """Matched modes stay adjacent and alternate which mode executes first."""
    captured: list[ConfigSnapshot] = []
    runner, _ = _make_runner(tmp_path, captured_configs=captured)

    runner.run_experiment(
        runs_per_condition=4,
        max_budget=5,
        conditions=(
            ("sql", "moderate", "stm_only"),
            ("sql", "moderate", "rag_ltm"),
        ),
    )

    assert [(cfg.seed, cfg.memory_mode) for cfg in captured] == [
        (1, "stm_only"),
        (1, "rag_ltm"),
        (2, "rag_ltm"),
        (2, "stm_only"),
        (3, "stm_only"),
        (3, "rag_ltm"),
        (4, "rag_ltm"),
        (4, "stm_only"),
    ]


def test_counterbalancing_preserves_ltm_seed_order_and_balances_first_position(
    tmp_path: Path,
) -> None:
    captured: list[ConfigSnapshot] = []
    runner, _ = _make_runner(tmp_path, captured_configs=captured)

    runner.run_experiment(
        runs_per_condition=30,
        max_budget=5,
        conditions=(
            ("file", "enterprise", "stm_only"),
            ("file", "enterprise", "rag_ltm"),
        ),
    )

    pairs = [captured[index:index + 2] for index in range(0, len(captured), 2)]
    first_modes = [pair[0].memory_mode for pair in pairs]
    rag_seeds = [cfg.seed for cfg in captured if cfg.memory_mode == "rag_ltm"]

    assert first_modes.count("stm_only") == 15
    assert first_modes.count("rag_ltm") == 15
    assert rag_seeds == list(range(1, 31))


def test_single_memory_mode_subset_keeps_seed_order(tmp_path: Path) -> None:
    captured: list[ConfigSnapshot] = []
    runner, _ = _make_runner(tmp_path, captured_configs=captured)

    runner.run_experiment(
        runs_per_condition=3,
        max_budget=5,
        conditions=(("sql", "moderate", "rag_ltm"),),
    )

    assert [(cfg.seed, cfg.memory_mode) for cfg in captured] == [
        (1, "rag_ltm"),
        (2, "rag_ltm"),
        (3, "rag_ltm"),
    ]


def test_experiment_id_unique_and_consistent(tmp_path: Path) -> None:
    """All stop_events from one run_experiment() share the same experiment_id."""
    runner, store = _make_runner(tmp_path)
    exp_id = runner.run_experiment(runs_per_condition=2, max_budget=5)

    events = store.get_stop_events(exp_id)
    assert len(events) == 16
    assert all(e.experiment_id == exp_id for e in events)


def test_condition_strings_correct_format(tmp_path: Path) -> None:
    """Condition strings must follow the 'scenario_guardrail_memory' pattern."""
    runner, store = _make_runner(tmp_path)
    exp_id = runner.run_experiment(runs_per_condition=1, max_budget=5)

    conditions = {e.condition for e in store.get_stop_events(exp_id)}

    assert "sql_moderate_stm_only" in conditions
    assert "sql_moderate_rag_ltm" in conditions
    assert "sql_enterprise_stm_only" in conditions
    assert "sql_enterprise_rag_ltm" in conditions
    assert "file_moderate_stm_only" in conditions
    assert "file_moderate_rag_ltm" in conditions
    assert "file_enterprise_stm_only" in conditions
    assert "file_enterprise_rag_ltm" in conditions
    assert len(conditions) == 8


def test_seed_values_per_condition_are_1_to_n(tmp_path: Path) -> None:
    """Seeds for each condition must be 1..N (not a global counter)."""
    runner, store = _make_runner(tmp_path)
    exp_id = runner.run_experiment(runs_per_condition=3, max_budget=5)

    stops = store.get_stop_events(exp_id)
    cond = "sql_moderate_stm_only"
    seeds = sorted(e.seed for e in stops if e.condition == cond)
    assert seeds == [1, 2, 3]


def test_config_passed_to_callback_has_correct_fields(tmp_path: Path) -> None:
    """ConfigSnapshot passed to run_callback must reflect the current condition."""
    captured: list[ConfigSnapshot] = []
    runner, store = _make_runner(tmp_path, captured_configs=captured)
    exp_id = runner.run_experiment(runs_per_condition=1, max_budget=7)

    scenarios = {c.scenario for c in captured}
    guardrails = {c.guardrail_level for c in captured}
    modes = {c.memory_mode for c in captured}

    assert scenarios == {"sql", "file"}
    assert guardrails == {"moderate", "enterprise"}
    assert modes == {"stm_only", "rag_ltm"}
    assert all(c.max_budget == 7 for c in captured)
    assert all(c.experiment_id == exp_id for c in captured)


def test_prepare_run_called_once_per_run_with_same_config(tmp_path: Path) -> None:
    prepared: list[ConfigSnapshot] = []
    captured: list[ConfigSnapshot] = []
    runner, _ = _make_runner(
        tmp_path,
        prepared_configs=prepared,
        captured_configs=captured,
    )

    runner.run_experiment(runs_per_condition=1, max_budget=5)

    assert len(prepared) == len(captured) == 8
    assert [p.seed for p in prepared] == [c.seed for c in captured]
    assert [p.memory_mode for p in prepared] == [c.memory_mode for c in captured]


def test_runner_persists_raw_step_audit_fields(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()

    def prepare_run(config: ConfigSnapshot) -> None:
        return None

    def run_callback(config: ConfigSnapshot, step_event_sink=None) -> RunResult:
        return RunResult(
            stop_reason="budget_exhausted",
            q2b=1,
            censored_flag=True,
            budget_used=1,
            step_events=[_raw_step()],
            stop_event={"timestamp_ms": 30},
            run_state={},
        )

    runner = ComparativeRunner(
        prepare_run=prepare_run,
        run_callback=run_callback,
        memory_gateway=gateway,
        event_store=store,
    )
    exp_id = runner.run_experiment(
        runs_per_condition=1,
        max_budget=1,
        conditions=(("sql", "moderate", "stm_only"),),
    )

    [event] = store.get_step_events(exp_id)
    assert event.memory_context == {"stm_recent": [{"step_t": 0}], "ltm_hits": []}
    assert event.plan == {"strategy": "probe"}
    assert event.pre_snapshot == {"db_hash": "before", "fs_hash": "same"}
    assert event.post_snapshot == {"db_hash": "after", "fs_hash": "same"}
    assert event.input_decision == "allow"
    assert event.output_decision == "allow"
    assert event.timings == {"planner_ms": 1.5}
    assert event.errors == ["oracle fallback"]
    assert event.planner_debug["selected_strategy"] == "probe"
    assert event.mutator_debug["used_fallback"] is False


def test_runner_writes_step_immediately_when_callback_accepts_sink(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()

    def prepare_run(config: ConfigSnapshot) -> None:
        return None

    def run_callback(config: ConfigSnapshot, step_event_sink) -> RunResult:
        raw_step = _raw_step()
        step_event_sink(raw_step)
        return RunResult(
            stop_reason="budget_exhausted",
            q2b=1,
            censored_flag=True,
            budget_used=1,
            step_events=[raw_step],
            stop_event={"timestamp_ms": 30},
            run_state={},
            step_events_persisted=True,
        )

    runner = ComparativeRunner(
        prepare_run=prepare_run,
        run_callback=run_callback,
        memory_gateway=gateway,
        event_store=store,
    )

    exp_id = runner.run_experiment(
        runs_per_condition=1,
        max_budget=1,
        conditions=(("sql", "moderate", "stm_only"),),
    )

    assert len(store.get_step_events(exp_id)) == 1
    assert len(store.get_stop_events(exp_id)) == 1
    assert [ev.status for ev in store.get_run_events(exp_id)] == ["started", "completed"]


def test_runner_aborts_and_raises_on_run_failure(tmp_path: Path) -> None:
    """A failing run is marked aborted; ExperimentInterruptedError is raised immediately."""
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()

    def run_callback(config: ConfigSnapshot, step_event_sink) -> RunResult:
        step_event_sink(_raw_step())
        raise RuntimeError("synthetic crash after step")

    runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=run_callback,
        memory_gateway=gateway,
        event_store=store,
    )

    with pytest.raises(ExperimentInterruptedError, match="synthetic crash"):
        runner.run_experiment(
            runs_per_condition=1,
            max_budget=1,
            conditions=(("sql", "moderate", "rag_ltm"),),
        )

    run_events = store.get_run_events()
    assert [ev.status for ev in run_events] == ["started", "aborted"]
    assert run_events[-1].last_step_t == 1
    assert "synthetic crash" in run_events[-1].error
    # Step from the partial run is persisted; no stop_event for a failed run
    assert len(store.get_step_events(run_events[0].experiment_id)) == 1
    assert store.get_stop_events(run_events[0].experiment_id) == []


def test_runner_persists_stop_event_only_for_successful_run(tmp_path: Path) -> None:
    """A completed run produces a stop_event; a failed run does not."""
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()

    def run_callback(config: ConfigSnapshot, step_event_sink) -> RunResult:
        step_event_sink(_raw_step())
        return _fixed_result()

    runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=run_callback,
        memory_gateway=gateway,
        event_store=store,
    )

    experiment_id = runner.run_experiment(
        runs_per_condition=1,
        max_budget=5,
        conditions=(("sql", "moderate", "rag_ltm"),),
    )

    [stop] = store.get_stop_events(experiment_id)
    assert stop.seed == 1
    assert [ev.status for ev in store.get_run_events(experiment_id)] == ["started", "completed"]


def test_runner_resumes_only_missing_condition_seed_pairs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()
    first_calls: list[int] = []

    def first_callback(config: ConfigSnapshot, step_event_sink=None) -> RunResult:
        first_calls.append(config.seed)
        if config.seed == 2:
            raise RuntimeError("provider unavailable")
        return _fixed_result()

    first_runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=first_callback,
        memory_gateway=gateway,
        event_store=store,
    )

    with pytest.raises(ExperimentInterruptedError) as caught:
        first_runner.run_experiment(
            runs_per_condition=2,
            max_budget=5,
            conditions=(("sql", "moderate", "rag_ltm"),),
        )

    experiment_id = caught.value.experiment_id
    resumed_calls: list[int] = []
    resumed_runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: (
            resumed_calls.append(config.seed) or _fixed_result()
        ),
        memory_gateway=gateway,
        event_store=store,
    )

    resumed_id = resumed_runner.run_experiment(
        runs_per_condition=2,
        max_budget=5,
        conditions=(("sql", "moderate", "rag_ltm"),),
        resume_experiment_id=experiment_id,
    )

    assert resumed_id == experiment_id
    assert first_calls == [1, 2]
    assert resumed_calls == [2]
    stops = store.get_stop_events(experiment_id)
    assert [(event.seed, event.run_id) for event in stops] == [(1, 1), (2, 3)]
    assert gateway.reset_ltm_count == 1


def test_runner_resumes_a_new_condition_block_under_same_experiment_id(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()
    first_runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: _fixed_result(),
        memory_gateway=gateway,
        event_store=store,
    )
    experiment_id = first_runner.run_experiment(
        runs_per_condition=2,
        max_budget=5,
        conditions=(
            ("sql", "moderate", "stm_only"),
            ("sql", "moderate", "rag_ltm"),
        ),
    )

    resumed_runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: _fixed_result(),
        memory_gateway=gateway,
        event_store=store,
    )
    resumed_runner.run_experiment(
        runs_per_condition=2,
        max_budget=5,
        conditions=(
            ("file", "moderate", "stm_only"),
            ("file", "moderate", "rag_ltm"),
        ),
        resume_experiment_id=experiment_id,
    )

    stops = store.get_stop_events(experiment_id)
    assert len(stops) == 8
    assert {event.condition for event in stops} == {
        "sql_moderate_stm_only",
        "sql_moderate_rag_ltm",
        "file_moderate_stm_only",
        "file_moderate_rag_ltm",
    }
    assert gateway.reset_ltm_count == 4


def test_runner_rejects_resume_with_changed_budget(tmp_path: Path) -> None:
    runner, _ = _make_runner(tmp_path)
    experiment_id = runner.run_experiment(
        runs_per_condition=1,
        max_budget=5,
        conditions=(("sql", "moderate", "stm_only"),),
    )

    with pytest.raises(ValueError, match="max_budget"):
        runner.run_experiment(
            runs_per_condition=1,
            max_budget=6,
            conditions=(("file", "moderate", "stm_only"),),
            resume_experiment_id=experiment_id,
        )


def test_runner_rejects_resume_with_changed_experiment_config(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    first = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: _fixed_result(),
        memory_gateway=_SpyGateway(),
        event_store=store,
        experiment_config={"attack_model": "model-a"},
    )
    experiment_id = first.run_experiment(
        runs_per_condition=1,
        max_budget=5,
        conditions=(("sql", "moderate", "stm_only"),),
    )
    changed = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: _fixed_result(),
        memory_gateway=_SpyGateway(),
        event_store=store,
        experiment_config={"attack_model": "model-b"},
    )

    with pytest.raises(ValueError, match="attack_model"):
        changed.run_experiment(
            runs_per_condition=1,
            max_budget=5,
            conditions=(("file", "moderate", "stm_only"),),
            resume_experiment_id=experiment_id,
        )


def test_runner_rejects_resume_with_different_seed_space(tmp_path: Path) -> None:
    runner, _ = _make_runner(tmp_path)
    experiment_id = runner.run_experiment(
        runs_per_condition=2,
        max_budget=5,
        conditions=(("sql", "moderate", "stm_only"),),
    )

    with pytest.raises(ValueError, match="does not match"):
        runner.run_experiment(
            runs_per_condition=1,
            max_budget=5,
            conditions=(("sql", "moderate", "stm_only"),),
            resume_experiment_id=experiment_id,
        )


def test_runner_refuses_open_runs_by_default(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_run_event(
        RunEvent(
            experiment_id="exp-open",
            run_id=1,
            condition="sql_moderate_rag_ltm",
            seed=1,
            status="started",
            timestamp_ms=123,
        )
    )

    runner = ComparativeRunner(
        prepare_run=lambda config: None,
        run_callback=lambda config, step_event_sink=None: _fixed_result(),
        memory_gateway=_SpyGateway(),
        event_store=store,
    )

    with pytest.raises(RuntimeError, match="started runs without terminal"):
        runner.run_experiment(
            runs_per_condition=1,
            max_budget=1,
            conditions=(("sql", "moderate", "stm_only"),),
        )


def test_runner_requires_prepare_run_callback(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    gateway = _SpyGateway()

    with pytest.raises(ValueError, match="prepare_run callback is required"):
        ComparativeRunner(
            prepare_run=None,  # type: ignore[arg-type]
            run_callback=lambda cfg, step_event_sink=None: _fixed_result(),
            memory_gateway=gateway,
            event_store=store,
        )
