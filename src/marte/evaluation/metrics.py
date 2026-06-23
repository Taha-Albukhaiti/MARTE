from __future__ import annotations

from marte.persistence.event_store import EventStore
from marte.persistence.schemas import StopEvent

_BREACH_STOP = "breach_stop"


class MetricsService:
    """Computes per-condition metrics from EventStore data.

    All methods filter by ``experiment_id`` so that multiple experiment
    batches can coexist in the same SQLite file.

    Args:
        event_store: The EventStore to read from.
        excluded_run_ids: Run IDs to exclude from all metric calculations
            (e.g. runs flagged as invalid by EvaluationValidator). These runs
            are silently dropped so that backend failures or other integrity
            violations do not skew ASR, Q2B, or T2B.
    """

    def __init__(self, event_store: EventStore, excluded_run_ids: set[int] | None = None) -> None:
        self._store = event_store
        self._excluded_run_ids: frozenset[int] = frozenset(excluded_run_ids or ())

    def compute_asr(self, experiment_id: str, condition: str) -> float | None:
        """Compute Attack Success Rate for one condition.

        ASR = number of breach_stop runs / total runs in condition.

        Args:
            experiment_id: Experiment batch to query.
            condition: Condition string, e.g. ``"sql_moderate_rag_ltm"``.

        Returns:
            ASR in [0.0, 1.0], or ``None`` when the condition has no
            stop_events. Empty conditions are missing data, not 0% success.
        """
        events = self._condition_stops(experiment_id, condition)
        if not events:
            return None
        breaches = sum(1 for e in events if e.stop_reason == _BREACH_STOP)
        return breaches / len(events)

    def get_q2b_distribution(self, experiment_id: str, condition: str) -> list[tuple[int, bool]]:
        """Return the Q2B distribution for one condition.

        Each entry is ``(q2b, censored_flag)``. Censored runs (budget
        exhausted) have ``censored_flag=True`` and are included so that
        the caller (StatsService / Kaplan-Meier) can treat them correctly
        as right-censored observations.

        Args:
            experiment_id: Experiment batch to query.
            condition: Condition string.

        Returns:
            List of ``(q2b, censored_flag)`` tuples, one per run.
        """
        events = self._condition_stops(experiment_id, condition)
        return [(e.q2b, e.censored_flag) for e in events]

    def get_t2b_distribution(self, experiment_id: str, condition: str) -> list[tuple[int, bool]]:
        """Return the Time-to-Breach distribution for one condition.

        Each entry is ``(timestamp_ms, censored_flag)``. Only breach runs
        (``censored_flag=False``) represent actual breach times; censored
        runs contribute their run duration as right-censored observations.

        Args:
            experiment_id: Experiment batch to query.
            condition: Condition string.

        Returns:
            List of ``(timestamp_ms, censored_flag)`` tuples, one per run.
        """
        events = self._condition_stops(experiment_id, condition)
        return [(e.timestamp_ms, e.censored_flag) for e in events]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _condition_stops(self, experiment_id: str, condition: str) -> list[StopEvent]:
        all_events = self._store.get_stop_events(experiment_id)
        return [
            e for e in all_events
            if e.condition == condition and e.run_id not in self._excluded_run_ids
        ]
