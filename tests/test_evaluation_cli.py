from __future__ import annotations

from pathlib import Path

import pytest

from marte.evaluation.__main__ import (
    _build_comparison_table,
    _build_json_report,
    _fmt_asr,
    _fmt_float,
    _holm_family_sizes,
    _km_median_q2b,
    _list_experiment_ids,
    main,
)
from marte.evaluation.evaluation_validator import EXPECTED_CONDITIONS, Violation
from marte.evaluation.stats import ComparisonResult
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import RunEvent, StepEvent, StopEvent


def test_km_median_q2b_reports_none_when_median_not_reached() -> None:
    assert _km_median_q2b([(5, True), (5, True), (5, True)]) is None


def test_km_median_q2b_uses_survival_median_not_observed_only_median() -> None:
    assert _km_median_q2b([(1, False), (2, False), (5, True)]) == 2


def test_build_comparison_table_includes_holm_adjusted_p_values() -> None:
    class FakeStats:
        def __init__(self) -> None:
            self._results = iter(
                [
                    ComparisonResult(0.01, 1.0, 0.02, 0.03),
                    ComparisonResult(0.04, 1.0, 0.03, 0.02),
                    ComparisonResult(0.03, 1.0, 0.04, 0.01),
                    ComparisonResult(0.20, 1.0, 0.20, 0.20),
                ]
            )

        def compare_conditions(
            self,
            experiment_id: str,
            condition_a: str,
            condition_b: str,
        ) -> ComparisonResult:
            return next(self._results)

    rows = _build_comparison_table(FakeStats(), "exp-1")

    assert [r["asr_p_holm"] for r in rows] == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert [r["q2b_p_holm"] for r in rows] == pytest.approx([0.08, 0.09, 0.09, 0.20])
    assert [r["t2b_p_holm"] for r in rows] == pytest.approx([0.06, 0.06, 0.04, 0.20])


def test_holm_report_includes_effective_family_sizes_for_unavailable_p_values() -> None:
    class FakeStats:
        def __init__(self) -> None:
            self._results = iter(
                [
                    ComparisonResult(0.01, 1.0, 0.02, float("nan")),
                    ComparisonResult(0.04, 1.0, float("nan"), float("nan")),
                    ComparisonResult(0.03, 1.0, 0.04, 0.01),
                    ComparisonResult(0.20, 1.0, float("nan"), float("nan")),
                ]
            )

        def compare_conditions(
            self,
            experiment_id: str,
            condition_a: str,
            condition_b: str,
        ) -> ComparisonResult:
            return next(self._results)

    rows = _build_comparison_table(FakeStats(), "exp-1")
    sizes = _holm_family_sizes(rows)
    report = _build_json_report("exp-1", [], rows)

    assert sizes["asr"] == {"planned_m": 4, "effective_m": 4}
    assert sizes["q2b"] == {"planned_m": 4, "effective_m": 2}
    assert sizes["t2b"] == {"planned_m": 4, "effective_m": 1}
    assert report["holm"] == sizes


def test_empty_condition_asr_is_visible_missing_data_in_text_and_json() -> None:
    condition_summary = [
        {
            "condition": "sql_moderate_stm_only",
            "n_total": 0,
            "n_breach": 0,
            "asr": None,
            "median_q2b": None,
        }
    ]

    report = _build_json_report("exp-1", condition_summary, [])

    assert _fmt_asr(None) == "No data"
    assert report["conditions"][0]["asr"] == "n/a"


def test_infinite_odds_ratio_is_not_displayed_as_missing() -> None:
    assert _fmt_float(float("inf")) == "    inf"
    assert _fmt_float(float("-inf")) == "   -inf"


def test_list_experiment_ids_includes_run_event_only_experiments(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    store.write_run_event(
        RunEvent(
            experiment_id="exp-open",
            run_id=1,
            condition="sql_moderate_stm_only",
            seed=1,
            status="started",
            timestamp_ms=1000,
        )
    )
    store.write_stop_event(
        StopEvent(
            experiment_id="exp-stopped",
            run_id=1,
            condition="sql_moderate_stm_only",
            stop_reason="budget_exhausted",
            q2b=50,
            censored_flag=True,
            timestamp_ms=1000,
            seed=1,
        )
    )

    assert _list_experiment_ids(store) == ["exp-open", "exp-stopped"]


def test_build_json_report_includes_validation_status() -> None:
    violation = Violation(
        level="error",
        code="missing_condition",
        message="Missing expected condition: sql_moderate_stm_only.",
        condition="sql_moderate_stm_only",
    )

    report = _build_json_report("exp-1", [], [], violations=[violation])

    assert report["validation"]["status"] == "errors"
    assert report["validation"]["errors"] == 1
    assert report["validation"]["violations"][0]["code"] == "missing_condition"


def test_build_json_report_preserves_infinite_odds_ratio() -> None:
    report = _build_json_report(
        "exp-1",
        [],
        [
            {
                "pair": "sql_moderate_stm_only  vs  sql_moderate_rag_ltm",
                "asr_p": 1.0,
                "asr_p_holm": 1.0,
                "asr_or": float("inf"),
                "q2b_p": float("nan"),
                "q2b_p_holm": float("nan"),
                "t2b_p": float("nan"),
                "t2b_p_holm": float("nan"),
            }
        ],
    )

    assert report["comparisons"][0]["asr_odds_ratio"] == "inf"


def test_main_strict_exits_on_validation_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    store.write_stop_event(
        StopEvent(
            experiment_id="exp-strict",
            run_id=1,
            condition="sql_moderate_stm_only",
            stop_reason="budget_exhausted",
            q2b=50,
            censored_flag=True,
            timestamp_ms=1000,
            seed=1,
        )
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--db-path",
                str(db_path),
                "--experiment-id",
                "exp-strict",
                "--expected-runs",
                "1",
                "--strict",
            ]
        )

    assert exc_info.value.code == 1
    assert "Validation failed:" in capsys.readouterr().err


def test_main_non_strict_reports_breach_censored_inconsistency(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    for run_id, condition in enumerate(EXPECTED_CONDITIONS, start=1):
        is_inconsistent = condition == "sql_moderate_stm_only"
        stop = StopEvent(
            experiment_id="exp-warn",
            run_id=run_id,
            condition=condition,
            stop_reason="breach_stop" if is_inconsistent else "budget_exhausted",
            q2b=1 if is_inconsistent else 50,
            censored_flag=True,
            timestamp_ms=1000,
            seed=1,
        )
        for step_t in range(1, stop.q2b + 1):
            store.write_step_event(
                StepEvent(
                    experiment_id=stop.experiment_id,
                    run_id=run_id,
                    condition=condition,
                    step_t=step_t,
                    timestamp_ms=step_t * 10,
                    prompt="prompt",
                    response="response",
                    oracle_verdict=is_inconsistent and step_t == stop.q2b,
                    judge_verdict=(
                        "success"
                        if is_inconsistent and step_t == stop.q2b
                        else "fail"
                    ),
                )
            )
        store.write_stop_event(stop)
        store.write_run_event(
            RunEvent(
                experiment_id="exp-warn",
                run_id=run_id,
                condition=condition,
                seed=1,
                status="completed",
                timestamp_ms=1000,
                last_step_t=stop.q2b,
                stop_reason=stop.stop_reason,
            )
        )

    main(
        [
            "--db-path",
            str(db_path),
            "--experiment-id",
            "exp-warn",
            "--expected-runs",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "Validation issues: 1 error(s), 0 warning(s)." in captured.err
    assert "breach_stop but censored_flag=True" in captured.err
    assert "sql_moderate_stm_only" in captured.out
    assert "1.000" in captured.out
