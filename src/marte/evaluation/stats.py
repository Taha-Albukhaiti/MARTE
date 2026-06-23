from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np
from lifelines.statistics import logrank_test
from scipy import stats

from marte.evaluation.metrics import MetricsService

_MIN_OBSERVATIONS = 2  # minimum events per arm for a meaningful test
_FALLBACK_P = float("nan")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    """Statistical comparison between two experimental conditions.

    Args:
        asr_p_value: Two-sided p-value from Fisher's Exact Test on ASR.
        asr_odds_ratio: Odds ratio from Fisher's Exact Test.
        q2b_p_value: Log-rank p-value comparing Q2B survival curves.
        t2b_p_value: Log-rank p-value comparing Time-to-Breach survival curves.
    """

    asr_p_value: float
    asr_odds_ratio: float
    q2b_p_value: float
    t2b_p_value: float


def holm_adjust_p_values(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original order.

    NaN/inf values are treated as unavailable and preserved as NaN.
    """
    adjusted = [float("nan")] * len(p_values)
    finite: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        p_value = float(value)
        if math.isfinite(p_value):
            finite.append((index, p_value))

    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(finite):
        raw_adjusted = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, raw_adjusted)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


class StatsService:
    """Compares condition pairs (stm_only vs. rag_ltm).

    Uses Fisher's Exact Test for ASR (proportional data, small samples)
    and the Log-Rank Test with Kaplan-Meier for Q2B and Time-to-Breach
    (survival data with right-censoring). These are condition-level tests,
    not paired-sample tests over reused seeds.

    Args:
        metrics: MetricsService to pull per-condition data from.
    """

    def __init__(self, metrics: MetricsService) -> None:
        self._metrics = metrics

    def compare_conditions(self, experiment_id: str, condition_a: str, condition_b: str) -> ComparisonResult:
        """Run all three statistical tests for one condition pair.

        Args:
            experiment_id: Experiment batch to query.
            condition_a: First condition string (e.g. ``"sql_moderate_stm_only"``).
            condition_b: Second condition string (e.g. ``"sql_moderate_rag_ltm"``).

        Returns:
            ComparisonResult with p-values and odds ratio. Fields are NaN
            when there are insufficient observations for a valid test.
        """
        asr_or, asr_p = self._compare_asr(experiment_id, condition_a, condition_b)
        q2b_p = self._compare_survival(experiment_id, condition_a, condition_b, metric="q2b")
        t2b_p = self._compare_survival(experiment_id, condition_a, condition_b, metric="t2b")
        return ComparisonResult(asr_p_value=asr_p, asr_odds_ratio=asr_or, q2b_p_value=q2b_p, t2b_p_value=t2b_p)

    # ------------------------------------------------------------------
    # Internal test runners
    # ------------------------------------------------------------------

    def _compare_asr(self, experiment_id: str, condition_a: str, condition_b: str) -> tuple[float, float]:
        """Fisher's Exact Test on breach counts.

        Returns:
            ``(odds_ratio, p_value)`` tuple.
        """
        all_stops = self._metrics._store.get_stop_events(experiment_id)
        excluded = self._metrics._excluded_run_ids

        a_events = [e for e in all_stops if e.condition == condition_a and e.run_id not in excluded]
        b_events = [e for e in all_stops if e.condition == condition_b and e.run_id not in excluded]

        if not a_events or not b_events:
            return _FALLBACK_P, _FALLBACK_P

        a_breach = sum(1 for e in a_events if e.stop_reason == "breach_stop")
        a_no = len(a_events) - a_breach
        b_breach = sum(1 for e in b_events if e.stop_reason == "breach_stop")
        b_no = len(b_events) - b_breach

        contingency = [[a_breach, a_no], [b_breach, b_no]]
        odds_ratio, p_value = stats.fisher_exact(contingency)
        if (a_breach == 0 and b_breach == 0) or (a_no == 0 and b_no == 0):
            logger.warning(
                "Degenerate Fisher table %s: p-value is 1.0, but the odds ratio "
                "is undefined and the comparison has no discrimination.",
                contingency,
            )
        return float(odds_ratio), float(p_value)

    def _compare_survival(self, experiment_id: str, condition_a: str, condition_b: str, metric: str) -> float:
        """Log-Rank Test on Q2B or T2B survival data.

        Args:
            metric: ``"q2b"`` or ``"t2b"``.

        Returns:
            Log-rank p-value, or NaN when data are insufficient or when
            both groups are fully censored. With no observed breach in either
            arm, the survival curves cannot support an inferential statement.
        """
        if metric == "q2b":
            dist_a = self._metrics.get_q2b_distribution(experiment_id, condition_a)
            dist_b = self._metrics.get_q2b_distribution(experiment_id, condition_b)
        else:
            dist_a = self._metrics.get_t2b_distribution(experiment_id, condition_a)
            dist_b = self._metrics.get_t2b_distribution(experiment_id, condition_b)

        if len(dist_a) < _MIN_OBSERVATIONS or len(dist_b) < _MIN_OBSERVATIONS:
            return _FALLBACK_P

        durations_a = np.array([d[0] for d in dist_a], dtype=float)
        events_a = np.array([not d[1] for d in dist_a], dtype=bool)
        durations_b = np.array([d[0] for d in dist_b], dtype=float)
        events_b = np.array([not d[1] for d in dist_b], dtype=bool)

        if int(events_a.sum()) + int(events_b.sum()) == 0:
            logger.warning("No events in either group – p-value set to NaN.")
            return _FALLBACK_P

        result = logrank_test(durations_a, durations_b, event_observed_A=events_a, event_observed_B=events_b)
        return float(result.p_value)
