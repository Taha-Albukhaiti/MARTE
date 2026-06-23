from __future__ import annotations

from pathlib import Path


from marte.evaluation.metrics import MetricsService
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import StopEvent

_CONDITION = "sql_moderate_stm_only"
_EXP = "exp-test"


def _stop(
    run_id: int,
    *,
    stop_reason: str = "budget_exhausted",
    censored_flag: bool = True,
    q2b: int = 50,
    timestamp_ms: int = 5000,
    condition: str = _CONDITION,
    experiment_id: str = _EXP,
    seed: int = 1,
) -> StopEvent:
    return StopEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        condition=condition,
        stop_reason=stop_reason,
        q2b=q2b,
        censored_flag=censored_flag,
        timestamp_ms=timestamp_ms,
        seed=seed,
    )


def _breach(run_id: int, q2b: int = 10, timestamp_ms: int = 2000, **kwargs: object) -> StopEvent:
    return _stop(run_id, stop_reason="breach_stop", censored_flag=False, q2b=q2b, timestamp_ms=timestamp_ms, **kwargs)  # type: ignore[arg-type]


def _censored(run_id: int, q2b: int = 50, timestamp_ms: int = 9000, **kwargs: object) -> StopEvent:
    return _stop(run_id, stop_reason="budget_exhausted", censored_flag=True, q2b=q2b, timestamp_ms=timestamp_ms, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_asr_calculation(tmp_path: Path) -> None:
    """3 breaches out of 5 runs → ASR = 0.6."""
    store = EventStore(tmp_path / "events.db")
    for i in range(3):
        store.write_stop_event(_breach(run_id=i + 1))
    for i in range(3, 5):
        store.write_stop_event(_censored(run_id=i + 1))

    metrics = MetricsService(store)
    asr = metrics.compute_asr(_EXP, _CONDITION)

    assert abs(asr - 0.6) < 1e-9


def test_asr_zero_when_no_breaches(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    for i in range(5):
        store.write_stop_event(_censored(run_id=i + 1))

    metrics = MetricsService(store)
    assert metrics.compute_asr(_EXP, _CONDITION) == 0.0


def test_asr_one_when_all_breach(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    for i in range(4):
        store.write_stop_event(_breach(run_id=i + 1))

    metrics = MetricsService(store)
    assert metrics.compute_asr(_EXP, _CONDITION) == 1.0


def test_asr_none_when_condition_has_no_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    metrics = MetricsService(store)
    assert metrics.compute_asr(_EXP, _CONDITION) is None


def test_q2b_includes_censored(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_breach(run_id=1, q2b=10))
    store.write_stop_event(_censored(run_id=2, q2b=50))

    metrics = MetricsService(store)
    dist = metrics.get_q2b_distribution(_EXP, _CONDITION)

    assert len(dist) == 2
    q2b_vals = [d[0] for d in dist]
    censored_flags = [d[1] for d in dist]
    assert 10 in q2b_vals
    assert 50 in q2b_vals
    assert True in censored_flags
    assert False in censored_flags


def test_q2b_all_censored_returns_full_list(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    for i in range(3):
        store.write_stop_event(_censored(run_id=i + 1, q2b=50))

    metrics = MetricsService(store)
    dist = metrics.get_q2b_distribution(_EXP, _CONDITION)

    assert len(dist) == 3
    assert all(censored for _, censored in dist)


def test_t2b_includes_all_observations(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_breach(run_id=1, timestamp_ms=2000))
    store.write_stop_event(_censored(run_id=2, timestamp_ms=9000))

    metrics = MetricsService(store)
    dist = metrics.get_t2b_distribution(_EXP, _CONDITION)

    assert len(dist) == 2
    timestamps = [t for t, _ in dist]
    assert 2000 in timestamps
    assert 9000 in timestamps


def test_t2b_empty_when_all_censored(tmp_path: Path) -> None:
    """When all runs are censored, no actual breach times exist."""
    store = EventStore(tmp_path / "events.db")
    for i in range(5):
        store.write_stop_event(_censored(run_id=i + 1, timestamp_ms=9000))

    metrics = MetricsService(store)
    dist = metrics.get_t2b_distribution(_EXP, _CONDITION)

    # All observations are censored — there are no breach times
    breach_times = [t for t, censored in dist if not censored]
    assert breach_times == []


def test_metrics_filter_by_condition(tmp_path: Path) -> None:
    """compute_asr must not bleed across conditions."""
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_breach(run_id=1, condition="sql_moderate_stm_only"))
    store.write_stop_event(_breach(run_id=2, condition="sql_moderate_stm_only"))
    store.write_stop_event(_censored(run_id=3, condition="sql_moderate_rag_ltm"))

    metrics = MetricsService(store)
    assert metrics.compute_asr(_EXP, "sql_moderate_stm_only") == 1.0
    assert metrics.compute_asr(_EXP, "sql_moderate_rag_ltm") == 0.0


def test_metrics_filter_by_experiment_id(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(_breach(run_id=1, experiment_id="exp-A"))
    store.write_stop_event(_censored(run_id=2, experiment_id="exp-B"))

    metrics = MetricsService(store)
    assert metrics.compute_asr("exp-A", _CONDITION) == 1.0
    assert metrics.compute_asr("exp-B", _CONDITION) == 0.0
