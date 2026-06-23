from __future__ import annotations

from pathlib import Path

from marte.evaluation.evaluation_validator import CONDITION_PAIRS, EXPECTED_CONDITIONS, EvaluationValidator
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import RunEvent, StepEvent, StopEvent

_EXP = "exp-validator"


def _stop(
    run_id: int,
    condition: str,
    seed: int,
    *,
    stop_reason: str = "budget_exhausted",
    q2b: int = 50,
    censored_flag: bool = True,
    timestamp_ms: int = 1000,
) -> StopEvent:
    return StopEvent(
        experiment_id=_EXP,
        run_id=run_id,
        condition=condition,
        stop_reason=stop_reason,
        q2b=q2b,
        censored_flag=censored_flag,
        timestamp_ms=timestamp_ms,
        seed=seed,
    )


def _run_event(
    run_id: int,
    condition: str,
    seed: int,
    *,
    status: str = "completed",
    stop_reason: str = "budget_exhausted",
    error: str = "",
    last_step_t: int = 50,
) -> RunEvent:
    return RunEvent(
        experiment_id=_EXP,
        run_id=run_id,
        condition=condition,
        seed=seed,
        status=status,
        timestamp_ms=1000,
        last_step_t=last_step_t,
        stop_reason=stop_reason,
        error=error,
    )


def _write_completed_stop(store: EventStore, event: StopEvent) -> None:
    for step_t in range(1, event.q2b + 1):
        store.write_step_event(
            StepEvent(
                experiment_id=event.experiment_id,
                run_id=event.run_id,
                condition=event.condition,
                step_t=step_t,
                timestamp_ms=min(step_t * 10, event.timestamp_ms),
                prompt="prompt",
                response="response",
                oracle_verdict=(
                    event.stop_reason == "breach_stop"
                    and step_t == event.q2b
                ),
                judge_verdict=(
                    "success"
                    if event.stop_reason == "breach_stop"
                    and step_t == event.q2b
                    else "fail"
                ),
            )
        )
    store.write_stop_event(event)
    store.write_run_event(
        _run_event(
            event.run_id,
            event.condition,
            event.seed,
            status="started",
            stop_reason="",
            last_step_t=0,
        )
    )
    store.write_run_event(
        _run_event(
            event.run_id,
            event.condition,
            event.seed,
            status="completed",
            stop_reason=event.stop_reason,
            last_step_t=event.q2b,
        )
    )


def test_validator_accepts_complete_small_experiment(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = 1
    for stm_condition, rag_condition in CONDITION_PAIRS:
        for seed, ordered_conditions in (
            (1, (stm_condition, rag_condition)),
            (2, (rag_condition, stm_condition)),
        ):
            for condition in ordered_conditions:
                _write_completed_stop(
                    store,
                    _stop(
                        run_id,
                        condition,
                        seed=seed,
                        stop_reason="breach_stop" if seed == 1 else "budget_exhausted",
                        q2b=1 if seed == 1 else 50,
                        censored_flag=seed != 1,
                    ),
                )
                run_id += 1

    validator = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=2,
    )

    assert validator.check_all() == []


def test_validator_rejects_condition_major_execution_order(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = 1
    for condition in EXPECTED_CONDITIONS:
        for seed in (1, 2):
            _write_completed_stop(store, _stop(run_id, condition, seed=seed))
            run_id += 1

    violations = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=2,
    ).check_all()
    codes = {violation.code for violation in violations}

    assert "execution_order" in codes
    assert "execution_adjacency" in codes


def test_validator_detects_seed_gaps_duplicates_and_open_runs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    condition = "sql_moderate_stm_only"
    _write_completed_stop(store, _stop(1, condition, seed=1))
    _write_completed_stop(store, _stop(2, condition, seed=1))
    store.write_run_event(
        _run_event(
            99,
            "sql_moderate_rag_ltm",
            seed=2,
            status="started",
            stop_reason="",
        )
    )

    validator = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=2,
    )
    codes = {violation.code for violation in validator.check_all()}

    assert "duplicate_seed" in codes
    assert "missing_seed" in codes
    assert "open_run" in codes


def test_validator_flags_stop_event_invariants(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    _write_completed_stop(
        store,
        _stop(
            1,
            "sql_moderate_stm_only",
            seed=1,
            stop_reason="breach_stop",
            q2b=51,
            censored_flag=True,
        ),
    )
    _write_completed_stop(
        store,
        _stop(
            2,
            "sql_moderate_rag_ltm",
            seed=1,
            stop_reason="budget_exhausted",
            q2b=49,
            censored_flag=False,
        ),
    )
    _write_completed_stop(
        store,
        _stop(
            3,
            "sql_enterprise_stm_only",
            seed=1,
            stop_reason="error",
            q2b=1,
            censored_flag=True,
        ),
    )
    store.write_stop_event(_stop(1, "sql_moderate_stm_only", seed=1))

    validator = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=1,
    )
    codes = {violation.code for violation in validator.check_all()}

    assert "breach_censored" in codes
    assert "breach_q2b_over_budget" in codes
    assert "budget_not_censored" in codes
    assert "budget_q2b_mismatch" in codes
    assert "error_stop" in codes
    assert "duplicate_stop_run" in codes


def test_validator_flags_zero_timestamp_as_invalid_t2b(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    _write_completed_stop(
        store,
        _stop(
            1,
            "sql_moderate_stm_only",
            seed=1,
            stop_reason="breach_stop",
            q2b=1,
            censored_flag=False,
            timestamp_ms=0,
        ),
    )

    validator = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=1,
    )
    codes = {violation.code for violation in validator.check_all()}

    assert "invalid_t2b_timestamp" in codes


def test_validator_cross_checks_stop_metrics_with_step_and_oracle_data(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    condition = "sql_moderate_stm_only"
    store.write_step_event(
        StepEvent(
            experiment_id=_EXP,
            run_id=1,
            condition=condition,
            step_t=1,
            timestamp_ms=100,
            prompt="prompt",
            response="response",
            oracle_verdict=True,
            judge_verdict="success",
        )
    )
    store.write_step_event(
        StepEvent(
            experiment_id=_EXP,
            run_id=1,
            condition=condition,
            step_t=3,
            timestamp_ms=300,
            prompt="prompt",
            response="response",
            oracle_verdict=False,
            judge_verdict="fail",
        )
    )
    store.write_stop_event(
        _stop(
            1,
            condition,
            seed=1,
            stop_reason="budget_exhausted",
            q2b=3,
            censored_flag=True,
            timestamp_ms=250,
        )
    )

    codes = {
        violation.code
        for violation in EvaluationValidator(
            store,
            experiment_id=_EXP,
            expected_runs_per_condition=1,
            max_budget=3,
        ).check_all()
    }

    assert "step_sequence_mismatch" in codes
    assert "stop_before_last_step" in codes
    assert "censored_run_has_breach" in codes


def test_validator_rejects_breach_stop_without_matching_oracle_step(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    condition = "sql_moderate_stm_only"
    store.write_step_event(
        StepEvent(
            experiment_id=_EXP,
            run_id=1,
            condition=condition,
            step_t=1,
            timestamp_ms=100,
            prompt="prompt",
            response="response",
            oracle_verdict=False,
            judge_verdict="fail",
        )
    )
    store.write_stop_event(
        _stop(
            1,
            condition,
            seed=1,
            stop_reason="breach_stop",
            q2b=1,
            censored_flag=False,
            timestamp_ms=120,
        )
    )

    codes = {
        violation.code
        for violation in EvaluationValidator(
            store,
            experiment_id=_EXP,
            expected_runs_per_condition=1,
        ).check_all()
    }

    assert "breach_oracle_mismatch" in codes


def test_validator_rejects_completed_run_with_legacy_backend_failure(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    _write_completed_stop(
        store,
        _stop(
            1,
            "sql_moderate_stm_only",
            seed=1,
            stop_reason="breach_stop",
            q2b=2,
            censored_flag=False,
        ),
    )
    store.write_step_event(
        StepEvent(
            experiment_id=_EXP,
            run_id=1,
            condition="sql_moderate_stm_only",
            step_t=1,
            timestamp_ms=100,
            prompt="prompt",
            response="[LLM ERROR] Error code: 429 - failed to process request",
            oracle_verdict=False,
            judge_verdict="fail",
        )
    )

    validator = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=1,
    )
    codes = {violation.code for violation in validator.check_all()}

    assert "completed_backend_failure" in codes


def test_validator_rejects_completed_runs_with_critical_phase_failures(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    condition = "sql_moderate_stm_only"
    errors = (
        "executor failed: target unavailable",
        "pre snapshot failed: database locked",
        "post snapshot failed: database locked",
        "oracle failed: invalid trace",
    )
    for run_id, error in enumerate(errors, start=1):
        store.write_step_event(
            StepEvent(
                experiment_id=_EXP,
                run_id=run_id,
                condition=condition,
                step_t=1,
                timestamp_ms=100,
                prompt="prompt",
                response="response",
                oracle_verdict=True,
                judge_verdict="success",
                errors=[error],
            )
        )
        store.write_stop_event(
            _stop(
                run_id,
                condition,
                seed=run_id,
                stop_reason="breach_stop",
                q2b=1,
                censored_flag=False,
                timestamp_ms=120,
            )
        )

    violations = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=4,
    ).check_all()
    rejected_run_ids = {
        violation.run_id
        for violation in violations
        if violation.code == "completed_critical_phase_failure"
    }

    assert rejected_run_ids == {1, 2, 3, 4}


def test_validator_excludes_recovered_retry_from_completed_observations(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    condition = "sql_moderate_stm_only"
    store.write_run_event(
        _run_event(
            1,
            condition,
            seed=1,
            status="started",
            stop_reason="",
        )
    )
    store.write_run_event(
        _run_event(
            1,
            condition,
            seed=1,
            status="retry_aborted",
            stop_reason="technical_retry",
            error="429",
        )
    )
    store.write_step_event(
        StepEvent(
            experiment_id=_EXP,
            run_id=1,
            condition=condition,
            step_t=1,
            timestamp_ms=100,
            prompt="prompt",
            response="[LLM ERROR] Error code: 429",
            oracle_verdict=False,
            judge_verdict="fail",
        )
    )
    _write_completed_stop(store, _stop(2, condition, seed=1))

    violations = EvaluationValidator(
        store,
        experiment_id=_EXP,
        expected_runs_per_condition=1,
    ).check_all()
    recovered = [
        violation
        for violation in violations
        if violation.code == "recovered_run_attempt"
    ]

    assert len(recovered) == 1
    assert recovered[0].level == "warning"
    assert not any(
        violation.code == "completed_backend_failure"
        and violation.run_id == 1
        for violation in violations
    )
