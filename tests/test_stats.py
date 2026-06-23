from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from marte.evaluation.metrics import MetricsService
from marte.evaluation.stats import StatsService, holm_adjust_p_values
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import StopEvent


_EXP = "exp-stats"
_COND_A = "sql_moderate_stm_only"
_COND_B = "sql_moderate_rag_ltm"


def _stop(
    run_id: int,
    condition: str,
    *,
    stop_reason: str,
    q2b: int,
    censored_flag: bool,
    timestamp_ms: int,
    seed: int = 1,
    experiment_id: str = _EXP,
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


def test_compare_conditions_returns_nan_for_survival_with_insufficient_observations(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")
    store.write_stop_event(
        _stop(
            run_id=1,
            condition=_COND_A,
            stop_reason="breach_stop",
            q2b=2,
            censored_flag=False,
            timestamp_ms=20,
        )
    )
    store.write_stop_event(
        _stop(
            run_id=2,
            condition=_COND_B,
            stop_reason="budget_exhausted",
            q2b=5,
            censored_flag=True,
            timestamp_ms=50,
        )
    )

    stats = StatsService(MetricsService(store))
    result = stats.compare_conditions(_EXP, _COND_A, _COND_B)

    assert result.asr_p_value == 1.0
    assert math.isnan(result.q2b_p_value)
    assert math.isnan(result.t2b_p_value)


def test_compare_conditions_fisher_detects_large_asr_difference(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    for i in range(1, 6):
        store.write_stop_event(
            _stop(
                run_id=i,
                condition=_COND_A,
                stop_reason="breach_stop",
                q2b=1,
                censored_flag=False,
                timestamp_ms=10,
            )
        )
    for i in range(6, 11):
        store.write_stop_event(
            _stop(
                run_id=i,
                condition=_COND_B,
                stop_reason="budget_exhausted",
                q2b=5,
                censored_flag=True,
                timestamp_ms=50,
            )
        )

    stats = StatsService(MetricsService(store))
    result = stats.compare_conditions(_EXP, _COND_A, _COND_B)

    assert result.asr_p_value < 0.05
    assert math.isinf(result.asr_odds_ratio)


def test_compare_conditions_survival_returns_finite_pvalues_with_enough_data(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")

    # Condition A: generally earlier breaches
    a_rows = [
        ("breach_stop", 1, False, 10),
        ("breach_stop", 2, False, 20),
        ("breach_stop", 3, False, 30),
        ("budget_exhausted", 5, True, 50),
    ]
    # Condition B: generally later breaches / more censoring
    b_rows = [
        ("breach_stop", 4, False, 40),
        ("breach_stop", 5, False, 50),
        ("budget_exhausted", 5, True, 50),
        ("budget_exhausted", 5, True, 50),
    ]

    run_id = 1
    for stop_reason, q2b, censored, t_ms in a_rows:
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_A,
                stop_reason=stop_reason,
                q2b=q2b,
                censored_flag=censored,
                timestamp_ms=t_ms,
            )
        )
        run_id += 1
    for stop_reason, q2b, censored, t_ms in b_rows:
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_B,
                stop_reason=stop_reason,
                q2b=q2b,
                censored_flag=censored,
                timestamp_ms=t_ms,
            )
        )
        run_id += 1

    stats = StatsService(MetricsService(store))
    result = stats.compare_conditions(_EXP, _COND_A, _COND_B)

    assert 0.0 <= result.q2b_p_value <= 1.0
    assert 0.0 <= result.t2b_p_value <= 1.0


def test_compare_conditions_survival_returns_nan_when_both_groups_all_censored(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = EventStore(tmp_path / "events.db")

    for run_id in range(1, 3):
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_A,
                stop_reason="budget_exhausted",
                q2b=5,
                censored_flag=True,
                timestamp_ms=50,
                seed=run_id,
            )
        )
    for run_id in range(3, 5):
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_B,
                stop_reason="budget_exhausted",
                q2b=5,
                censored_flag=True,
                timestamp_ms=50,
                seed=run_id - 2,
            )
        )

    caplog.set_level(logging.WARNING, logger="marte.evaluation.stats")
    stats = StatsService(MetricsService(store))
    result = stats.compare_conditions(_EXP, _COND_A, _COND_B)

    assert result.asr_p_value == 1.0
    assert math.isnan(result.asr_odds_ratio)
    assert math.isnan(result.q2b_p_value)
    assert math.isnan(result.t2b_p_value)
    assert "Degenerate Fisher table" in caplog.text
    assert "No events in either group" in caplog.text


def test_compare_conditions_fisher_reports_ceiling_as_degenerate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = EventStore(tmp_path / "events.db")

    for run_id in range(1, 3):
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_A,
                stop_reason="breach_stop",
                q2b=run_id,
                censored_flag=False,
                timestamp_ms=run_id * 10,
                seed=run_id,
            )
        )
    for run_id in range(3, 5):
        store.write_stop_event(
            _stop(
                run_id=run_id,
                condition=_COND_B,
                stop_reason="breach_stop",
                q2b=run_id - 2,
                censored_flag=False,
                timestamp_ms=(run_id - 2) * 10,
                seed=run_id - 2,
            )
        )

    caplog.set_level(logging.WARNING, logger="marte.evaluation.stats")
    result = StatsService(MetricsService(store)).compare_conditions(
        _EXP,
        _COND_A,
        _COND_B,
    )

    assert result.asr_p_value == 1.0
    assert math.isnan(result.asr_odds_ratio)
    assert "Degenerate Fisher table" in caplog.text


def test_holm_adjust_p_values_preserves_original_order_and_monotonicity() -> None:
    adjusted = holm_adjust_p_values([0.04, 0.001, 0.02, 0.20])

    assert adjusted == [0.08, 0.004, 0.06, 0.20]


def test_holm_adjust_p_values_preserves_nan_as_unavailable() -> None:
    adjusted = holm_adjust_p_values([0.01, float("nan"), 0.03])

    assert adjusted[0] == 0.02
    assert math.isnan(adjusted[1])
    assert adjusted[2] == 0.03
