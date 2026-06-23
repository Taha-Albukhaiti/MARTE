from __future__ import annotations

from collections import Counter, defaultdict

from marte.evaluation.validation_models import (
    CONDITION_PAIRS,
    EXPECTED_CONDITIONS,
    Violation,
    _VALID_STOP_REASONS,
    _looks_like_backend_failure,
)
from marte.evaluation.validation_utils import _format_values, _group_stops_by_condition
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import RunEvent, StopEvent


class EvaluationValidator:
    """Validate persisted MARTE-thesis-worktree experiment data before statistical reporting.

    The validator protects the thesis-level evaluation from silently accepting
    partial or corrupted EventStore data. It checks the canonical 2 x 2 x 2
    condition matrix, the expected number of valid runs per condition, seed
    completeness, STM/RAG seed pairing and counterbalanced execution order,
    run lifecycle consistency, and the StopEvent invariants required for ASR
    and right-censored survival analysis.

    StopEvents with ``stop_reason == "error"`` are reported as invalid runs and
    are excluded from expected-run and seed-completeness counts, because they do
    not represent successful observations of the experimental protocol.
    """

    def __init__(self, event_store: EventStore, experiment_id: str, expected_runs_per_condition: int = 30, max_budget: int = 50) -> None:
        if expected_runs_per_condition <= 0:
            raise ValueError("expected_runs_per_condition must be positive")
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        self._store = event_store
        self._experiment_id = experiment_id
        self._expected_runs = expected_runs_per_condition
        self._max_budget = max_budget

    def check_all(self) -> list[Violation]:
        """Run all integrity checks and return errors/warnings in stable order."""
        stops = self._store.get_stop_events(self._experiment_id)
        valid_stops = [event for event in stops if event.stop_reason != "error"]
        stops_by_condition = _group_stops_by_condition(valid_stops)
        violations: list[Violation] = []

        self._check_condition_matrix(stops_by_condition, violations)
        self._check_run_counts(stops_by_condition, violations)
        self._check_seed_sets(stops_by_condition, violations)
        self._check_seed_pairing(stops_by_condition, violations)
        self._check_execution_order(violations)
        self._check_run_lifecycle(stops, violations)
        self._check_completed_backend_failures(violations)
        self._check_completed_critical_phase_failures(violations)
        self._check_stop_invariants(stops, violations)
        self._check_stop_step_consistency(stops, violations)
        self._check_duplicate_stop_events(stops, violations)
        return violations

    def _check_completed_backend_failures(self, violations: list[Violation]) -> None:
        """Reject legacy completed runs containing API infrastructure failures."""
        successful_run_ids = {
            event.run_id
            for event in self._store.get_stop_events(self._experiment_id)
            if event.stop_reason != "error"
        }
        reported_runs: set[int] = set()
        for event in self._store.get_step_events(self._experiment_id):
            if event.run_id not in successful_run_ids:
                continue
            if event.run_id in reported_runs:
                continue
            evidence = [
                event.response,
                str(event.planner_debug.get("error", "")),
                str(event.judge_debug.get("error", "")),
                *(str(error) for error in event.errors),
            ]
            matching = next(
                (text for text in evidence if _looks_like_backend_failure(text)),
                None,
            )
            if matching is None:
                continue
            reported_runs.add(event.run_id)
            violations.append(
                Violation(
                    level="error",
                    code="completed_backend_failure",
                    condition=event.condition,
                    run_id=event.run_id,
                    message=(
                        f"Run {event.run_id} in {event.condition} was completed "
                        "despite an LLM backend failure and is not a valid "
                        f"evaluation run: {matching[:180]}."
                    ),
                )
            )

    def _check_completed_critical_phase_failures(self, violations: list[Violation]) -> None:
        """Reject completed runs whose execution or ground truth is unusable."""
        successful_run_ids = {
            event.run_id
            for event in self._store.get_stop_events(self._experiment_id)
            if event.stop_reason != "error"
        }
        reported_runs: set[int] = set()
        critical_prefixes = (
            "executor failed:",
            "target execution failed:",
            "pre snapshot failed:",
            "post snapshot failed:",
            "oracle failed:",
        )
        for event in self._store.get_step_events(self._experiment_id):
            if event.run_id not in successful_run_ids:
                continue
            if event.run_id in reported_runs:
                continue
            matching = next(
                (
                    str(error)
                    for error in event.errors
                    if str(error).strip().lower().startswith(critical_prefixes)
                ),
                None,
            )
            if matching is None:
                continue
            reported_runs.add(event.run_id)
            violations.append(
                Violation(
                    level="error",
                    code="completed_critical_phase_failure",
                    condition=event.condition,
                    run_id=event.run_id,
                    message=(
                        f"Run {event.run_id} in {event.condition} was completed "
                        "despite a critical execution or ground-truth failure "
                        f"and is not a valid evaluation run: {matching[:180]}."
                    ),
                )
            )

    def _check_condition_matrix(self, stops_by_condition: dict[str, list[StopEvent]], violations: list[Violation]) -> None:
        present = set(stops_by_condition)
        expected = set(EXPECTED_CONDITIONS)
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)

        for condition in missing:
            violations.append(
                Violation(level="error", code="missing_condition", condition=condition, message=f"Missing expected condition: {condition}.")
            )
        for condition in unexpected:
            violations.append(
                Violation(level="error", code="unexpected_condition", condition=condition, message=f"Unexpected condition in stop_events: {condition}.")
            )

        if not missing and not unexpected and len(present) != len(EXPECTED_CONDITIONS):
            violations.append(
                Violation(
                    level="error",
                    code="condition_count",
                    message=(
                        f"Expected {len(EXPECTED_CONDITIONS)} conditions, "
                        f"found {len(present)}."
                    ),
                )
            )

    def _check_run_counts(self, stops_by_condition: dict[str, list[StopEvent]], violations: list[Violation]) -> None:
        for condition in EXPECTED_CONDITIONS:
            actual = len(stops_by_condition.get(condition, []))
            if actual != self._expected_runs:
                violations.append(
                    Violation(
                        level="error",
                        code="run_count",
                        condition=condition,
                        message=(
                            f"Condition {condition} has {actual} valid stop_events; "
                            f"expected {self._expected_runs}."
                        ),
                    )
                )

    def _check_seed_sets(self, stops_by_condition: dict[str, list[StopEvent]], violations: list[Violation]) -> None:
        expected_seeds = set(range(1, self._expected_runs + 1))
        for condition in EXPECTED_CONDITIONS:
            seeds = [event.seed for event in stops_by_condition.get(condition, [])]
            seed_counts = Counter(seeds)
            duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
            missing = sorted(expected_seeds - set(seeds))
            unexpected = sorted(set(seeds) - expected_seeds)

            if duplicate_seeds:
                violations.append(
                    Violation(
                        level="error",
                        code="duplicate_seed",
                        condition=condition,
                        message=(
                            f"Condition {condition} has duplicate seeds: "
                            f"{_format_values(duplicate_seeds)}."
                        ),
                    )
                )
            if missing:
                violations.append(
                    Violation(
                        level="error",
                        code="missing_seed",
                        condition=condition,
                        message=(
                            f"Condition {condition} is missing seeds: "
                            f"{_format_values(missing)}."
                        ),
                    )
                )
            if unexpected:
                violations.append(
                    Violation(
                        level="error",
                        code="unexpected_seed",
                        condition=condition,
                        message=(
                            f"Condition {condition} has seeds outside 1.."
                            f"{self._expected_runs}: {_format_values(unexpected)}."
                        ),
                    )
                )

    def _check_seed_pairing(self, stops_by_condition: dict[str, list[StopEvent]], violations: list[Violation]) -> None:
        for condition_a, condition_b in CONDITION_PAIRS:
            seeds_a = {event.seed for event in stops_by_condition.get(condition_a, [])}
            seeds_b = {event.seed for event in stops_by_condition.get(condition_b, [])}
            if seeds_a != seeds_b:
                only_a = sorted(seeds_a - seeds_b)
                only_b = sorted(seeds_b - seeds_a)
                violations.append(
                    Violation(
                        level="error",
                        code="seed_pairing",
                        message=(
                            f"Seed sets differ between {condition_a} and {condition_b}; "
                            f"only in first: {_format_values(only_a)}, "
                            f"only in second: {_format_values(only_b)}."
                        ),
                    )
                )

    def _check_execution_order(self, violations: list[Violation]) -> None:
        """Verify order using only successful attempts, excluding technical retries."""
        successful_stops = [
            event
            for event in self._store.get_stop_events(self._experiment_id)
            if event.stop_reason != "error"
        ]
        successful_by_key = {
            (event.condition, event.seed): event
            for event in successful_stops
        }
        successful_run_order = {
            event.run_id: position
            for position, event in enumerate(
                sorted(successful_stops, key=lambda item: item.run_id)
            )
        }

        for stm_condition, rag_condition in CONDITION_PAIRS:
            for seed in range(1, self._expected_runs + 1):
                stm_stop = successful_by_key.get((stm_condition, seed))
                rag_stop = successful_by_key.get((rag_condition, seed))
                if stm_stop is None or rag_stop is None:
                    continue

                stm_run_id = stm_stop.run_id
                rag_run_id = rag_stop.run_id
                expected_first = stm_condition if seed % 2 == 1 else rag_condition
                actual_first = stm_condition if stm_run_id < rag_run_id else rag_condition
                if actual_first != expected_first:
                    violations.append(
                        Violation(
                            level="error",
                            code="execution_order",
                            condition=actual_first,
                            run_id=min(stm_run_id, rag_run_id),
                            message=(
                                f"Seed {seed} for {stm_condition} vs. {rag_condition} "
                                f"started with {actual_first}; expected {expected_first} "
                                "from the counterbalanced schedule."
                            ),
                        )
                    )
                if abs(
                    successful_run_order[stm_run_id]
                    - successful_run_order[rag_run_id]
                ) != 1:
                    violations.append(
                        Violation(
                            level="error",
                            code="execution_adjacency",
                            run_id=min(stm_run_id, rag_run_id),
                            message=(
                                f"Seed {seed} for {stm_condition} vs. {rag_condition} "
                                "was not executed in adjacent runs."
                            ),
                        )
                    )

    def _check_run_lifecycle(self, stops: list[StopEvent], violations: list[Violation]) -> None:
        run_events = self._store.get_run_events(self._experiment_id)
        if not run_events:
            violations.append(
                Violation(
                    level="warning",
                    code="missing_run_events",
                    message=(
                        "No run_events found; open/aborted run lifecycle checks "
                        "cannot be verified for this experiment."
                    ),
                )
            )
            return

        stops_by_run = defaultdict(list)
        for stop in stops:
            stops_by_run[stop.run_id].append(stop)
        successful_keys = {
            (stop.condition, stop.seed)
            for stop in stops
            if stop.stop_reason != "error"
        }

        latest_by_run: dict[int, RunEvent] = {}
        for event in run_events:
            latest_by_run[event.run_id] = event

        for event in self._get_open_run_events():
            if event.run_id not in stops_by_run:
                violations.append(
                    Violation(
                        level="error",
                        code="open_run",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} is still open "
                            "and has no stop_event."
                        ),
                    )
                )
            else:
                violations.append(
                    Violation(
                        level="warning",
                        code="open_lifecycle_with_stop",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} has a stop_event "
                            "but no terminal run_event."
                        ),
                    )
                )

        for run_id, event in latest_by_run.items():
            if event.status in {"aborted", "retry_aborted"}:
                recovered = (event.condition, event.seed) in successful_keys
                violations.append(
                    Violation(
                        level="warning" if recovered else "error",
                        code=(
                            "recovered_run_attempt"
                            if recovered
                            else "unrecovered_run_attempt"
                        ),
                        condition=event.condition,
                        run_id=run_id,
                        message=(
                            f"Run attempt {run_id} in {event.condition}, "
                            f"seed={event.seed}, ended as {event.status}"
                            f"{' and was superseded by a successful retry' if recovered else ''}: "
                            f"{event.error or 'no error message'}."
                        ),
                    )
                )
            elif event.status == "completed" and run_id not in stops_by_run:
                violations.append(
                    Violation(
                        level="error",
                        code="completed_without_stop",
                        condition=event.condition,
                        run_id=run_id,
                        message=(
                            f"Run {run_id} in {event.condition} has completed "
                            "run_event but no stop_event."
                        ),
                    )
                )
            elif event.status not in {
                "started",
                "completed",
                "aborted",
                "retry_aborted",
            }:
                violations.append(
                    Violation(
                        level="error",
                        code="unknown_run_status",
                        condition=event.condition,
                        run_id=run_id,
                        message=(
                            f"Run {run_id} in {event.condition} has unknown "
                            f"run_event status: {event.status!r}."
                        ),
                    )
                )

        for stop in stops:
            if stop.run_id not in latest_by_run:
                violations.append(
                    Violation(
                        level="warning",
                        code="stop_without_run_event",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} in {stop.condition} has a stop_event "
                            "but no run_events lifecycle record."
                        ),
                    )
                )

    def _check_stop_invariants(self, stops: list[StopEvent], violations: list[Violation]) -> None:
        for event in stops:
            if event.stop_reason not in _VALID_STOP_REASONS:
                violations.append(
                    Violation(
                        level="error",
                        code="unknown_stop_reason",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} has unknown "
                            f"stop_reason: {event.stop_reason!r}."
                        ),
                    )
                )
                continue

            if event.stop_reason == "error":
                violations.append(
                    Violation(
                        level="error",
                        code="error_stop",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} has "
                            "stop_reason='error' and is not a valid evaluation run."
                        ),
                    )
                )
                continue

            if event.q2b < 1:
                violations.append(
                    Violation(
                        level="error",
                        code="invalid_q2b",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} has q2b={event.q2b}; "
                            "q2b must be >= 1."
                        ),
                    )
                )

            if event.timestamp_ms <= 0:
                violations.append(
                    Violation(
                        level="error",
                        code="invalid_t2b_timestamp",
                        condition=event.condition,
                        run_id=event.run_id,
                        message=(
                            f"Run {event.run_id} in {event.condition} has "
                            f"timestamp_ms={event.timestamp_ms}; T2B requires "
                            "a positive run-relative duration."
                        ),
                    )
                )

            if event.stop_reason == "breach_stop":
                if event.censored_flag:
                    violations.append(
                        Violation(
                            level="error",
                            code="breach_censored",
                            condition=event.condition,
                            run_id=event.run_id,
                            message=(
                                f"Run {event.run_id} in {event.condition} is breach_stop "
                                "but censored_flag=True."
                            ),
                        )
                    )
                if event.q2b > self._max_budget:
                    violations.append(
                        Violation(
                            level="error",
                            code="breach_q2b_over_budget",
                            condition=event.condition,
                            run_id=event.run_id,
                            message=(
                                f"Run {event.run_id} in {event.condition} breached at "
                                f"q2b={event.q2b}, above max budget {self._max_budget}."
                            ),
                        )
                    )

            if event.stop_reason == "budget_exhausted":
                if not event.censored_flag:
                    violations.append(
                        Violation(
                            level="error",
                            code="budget_not_censored",
                            condition=event.condition,
                            run_id=event.run_id,
                            message=(
                                f"Run {event.run_id} in {event.condition} is "
                                "budget_exhausted but censored_flag=False."
                            ),
                        )
                    )
                if event.q2b != self._max_budget:
                    violations.append(
                        Violation(
                            level="error",
                            code="budget_q2b_mismatch",
                            condition=event.condition,
                            run_id=event.run_id,
                            message=(
                                f"Run {event.run_id} in {event.condition} is "
                                f"budget_exhausted with q2b={event.q2b}; "
                                f"expected q2b={self._max_budget}."
                            ),
                        )
                    )

    def _check_stop_step_consistency(self, stops: list[StopEvent], violations: list[Violation]) -> None:
        """Cross-check terminal metrics against persisted step and Oracle data."""
        steps_by_run = defaultdict(list)
        for step in self._store.get_step_events(self._experiment_id):
            steps_by_run[step.run_id].append(step)

        for stop in stops:
            if stop.stop_reason == "error":
                continue

            steps = sorted(
                steps_by_run.get(stop.run_id, []),
                key=lambda event: (event.step_t, event.event_id or 0),
            )
            if not steps:
                violations.append(
                    Violation(
                        level="error",
                        code="missing_step_events",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} in {stop.condition} has a valid "
                            "stop_event but no persisted step_events."
                        ),
                    )
                )
                continue

            mismatched_conditions = sorted(
                {step.condition for step in steps if step.condition != stop.condition}
            )
            if mismatched_conditions:
                violations.append(
                    Violation(
                        level="error",
                        code="step_condition_mismatch",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} stop condition is {stop.condition}, "
                            "but step_events also contain: "
                            f"{', '.join(mismatched_conditions)}."
                        ),
                    )
                )

            step_indices = [step.step_t for step in steps]
            expected_indices = (
                list(range(1, stop.q2b + 1))
                if stop.q2b >= 1
                else []
            )
            if step_indices != expected_indices:
                violations.append(
                    Violation(
                        level="error",
                        code="step_sequence_mismatch",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} in {stop.condition} has step indices "
                            f"{_format_values(step_indices)}, but q2b={stop.q2b} "
                            f"requires {_format_values(expected_indices)}."
                        ),
                    )
                )

            last_step_timestamp = max(step.timestamp_ms for step in steps)
            if stop.timestamp_ms < last_step_timestamp:
                violations.append(
                    Violation(
                        level="error",
                        code="stop_before_last_step",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} in {stop.condition} has stop duration "
                            f"{stop.timestamp_ms} ms before its last step timestamp "
                            f"{last_step_timestamp} ms."
                        ),
                    )
                )

            oracle_steps = [step.step_t for step in steps if step.oracle_verdict]
            if stop.stop_reason == "breach_stop":
                if oracle_steps != [stop.q2b]:
                    violations.append(
                        Violation(
                            level="error",
                            code="breach_oracle_mismatch",
                            condition=stop.condition,
                            run_id=stop.run_id,
                            message=(
                                f"Run {stop.run_id} in {stop.condition} ends with "
                                f"breach_stop at q2b={stop.q2b}, but positive Oracle "
                                f"steps are {_format_values(oracle_steps)}."
                            ),
                        )
                    )
            elif stop.stop_reason == "budget_exhausted" and oracle_steps:
                violations.append(
                    Violation(
                        level="error",
                        code="censored_run_has_breach",
                        condition=stop.condition,
                        run_id=stop.run_id,
                        message=(
                            f"Run {stop.run_id} in {stop.condition} is right-censored, "
                            "but contains positive Oracle verdicts at steps "
                            f"{_format_values(oracle_steps)}."
                        ),
                    )
                )

    def _check_duplicate_stop_events(self, stops: list[StopEvent], violations: list[Violation]) -> None:
        run_counts = Counter(event.run_id for event in stops)
        for run_id, count in sorted(run_counts.items()):
            if count > 1:
                violations.append(
                    Violation(
                        level="error",
                        code="duplicate_stop_run",
                        run_id=run_id,
                        message=(
                            f"Run {run_id} has {count} stop_events; expected exactly one "
                            "StopEvent per run_id."
                        ),
                    )
                )

        condition_seed_counts = Counter((event.condition, event.seed) for event in stops)
        for (condition, seed), count in sorted(condition_seed_counts.items()):
            if count > 1:
                violations.append(
                    Violation(
                        level="error",
                        code="duplicate_condition_seed",
                        condition=condition,
                        message=(
                            f"Condition {condition} has {count} stop_events for seed {seed}; "
                            "expected at most one."
                        ),
                    )
                )

    def _get_open_run_events(self) -> list[RunEvent]:
        getter = getattr(self._store, "get_open_runs", None)
        if getter is not None:
            return list(getter(self._experiment_id))
        return self._store.get_open_run_events(self._experiment_id)
