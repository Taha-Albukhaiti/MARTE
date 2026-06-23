from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from marte.attack.orchestrator import RunResult
from marte.config.snapshot import ConfigSnapshot
from marte.experiment.helper import (
    CONDITIONS,
    RunCallback,
    StepEventSink,
    all_conditions,
    build_execution_schedule,
    build_step_event,
    format_condition,
    generate_experiment_id,
    ltm_hits_count,
    ltm_scope_name,
    memory_context_length,
    safe_append_jsonl,
    safe_touch_debug_file,
    set_ltm_scope,
    wall_time_ms,
)
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import ExperimentManifest, RunEvent, StopEvent

__all__ = [
    "CONDITIONS",
    "RunCallback",
    "StepEventSink",
    "ComparativeRunner",
    "ExperimentInterruptedError",
    "all_conditions",
    "build_step_event",
    "format_condition",
]


class _LTMResettable(Protocol):
    """Minimal interface the ComparativeRunner needs from the memory gateway."""

    def reset_ltm(self) -> None: ...


PrepareRunCallback = Callable[[ConfigSnapshot], None]


class ExperimentInterruptedError(RuntimeError):
    """Raised after a run exhausts retries while preserving resume state."""

    def __init__(self, experiment_id: str, condition: str, seed: int, cause: BaseException) -> None:
        self.experiment_id = experiment_id
        self.condition = condition
        self.seed = seed
        self.cause = cause
        super().__init__(
            f"Experiment {experiment_id} paused at {condition}, seed={seed} after "
            f"all run attempts failed: {cause}. Resume with "
            f"--resume-experiment-id {experiment_id} and the same configuration."
        )


class ComparativeRunner:
    """Orchestrates the counterbalanced 2×2×2 experiment.

    Calls ``reset_ltm()`` exactly once per condition (not per run) so that
    the VectorDB accumulates knowledge within a condition while remaining
    isolated between conditions. Within each scenario/guardrail pair, matched
    seeds are executed consecutively and the memory-mode order alternates.

    Args:
        prepare_run: Called once per run before run_callback. Must perform
            run initialization (e.g., sandbox reset, target init_config).
        run_callback: Called once per run; receives a frozen ConfigSnapshot
            and a step_event_sink callback, then returns a RunResult. Tests
            inject a spy here.
        memory_gateway: Must implement ``reset_ltm()``.
        event_store: Where step_events and stop_events are persisted.
        model_cfg: Base LLM configuration merged into each ConfigSnapshot
            (seed is overridden per-run).
        tool_allowlist: Tool names forwarded to ConfigSnapshot.
        allow_incomplete_runs: If false, refuse to start a new experiment when
            run_events contains attempts without a terminal lifecycle marker.
        experiment_config: Additional fixed, secret-free configuration persisted
            for exact resume validation.
    """

    def __init__(self, prepare_run: PrepareRunCallback, run_callback: RunCallback, memory_gateway: _LTMResettable,
                 event_store: EventStore, model_cfg: dict[str, Any] | None = None,
                 tool_allowlist: list[str] | None = None, allow_incomplete_runs: bool = False,
                 experiment_config: dict[str, Any] | None = None) -> None:
        if prepare_run is None:
            raise ValueError("prepare_run callback is required")
        if run_callback is None:
            raise ValueError("run_callback callback is required")
        if memory_gateway is None:
            raise ValueError("memory_gateway is required")
        if event_store is None:
            raise ValueError("event_store is required")
        self._prepare_run = prepare_run
        self._run_callback = run_callback
        self._memory_gateway = memory_gateway
        self._event_store = event_store
        self._model_cfg: dict[str, Any] = dict(model_cfg or {})
        self._tool_allowlist: list[str] = list(tool_allowlist or [])
        self._allow_incomplete_runs = allow_incomplete_runs
        self._experiment_config: dict[str, Any] = dict(experiment_config or {})

    def run_experiment(self, runs_per_condition: int = 30, max_budget: int = 50,
                       conditions: tuple[tuple[str, str, str], ...] | None = None,
                       resume_experiment_id: str | None = None) -> str:
        """Execute conditions × ``runs_per_condition`` runs.

        Args:
            runs_per_condition: Number of seeded runs per condition (N=30 in the standard experiment).
            max_budget: Maximum attack steps per run (B=50 by default).
            conditions: Subset of conditions to run. If None, runs all 8 conditions from the canonical 2×2×2 matrix.

        Returns:
            The ``experiment_id`` assigned to this batch — use it to query
            the EventStore afterwards.

        Design-Entscheidungen (WARUM so):
        ------------------------------------
        • LTM-Reset pro Condition (nicht pro Run): LTM akkumuliert INNERHALB einer Condition
          über 30 Seeds, bleibt aber isoliert zwischen Conditions. Das ermöglicht:
          - Angreifer lernt über mehrere Versuche (Seed 1..30 bauen aufeinander auf)
          - Aber File-Scenario beeinflusst nicht SQL-Scenario (Kontaminations-Schutz)

        • Gegenbalancierte Reihenfolge pro Seed: Innerhalb jedes Szenario-Guardrail-Paars
          läuft bei ungeraden Seeds stm_only vor rag_ltm und bei geraden Seeds umgekehrt.
          Damit sind gepaarte Seeds zeitlich benachbart, und kein Gedächtnismodus wird
          systematisch früher von zeitabhängigen API-Bedingungen getroffen.

        • write_step_event sofort (via step_event_sink): Jeder Step wird unmittelbar nach
          Ausführung persistiert. Das ermöglicht:
          - Crash-Forensik: bei Prozess-Abbruch wissen wir exakt, wo wir waren
          - Checkpoint für recovery: last_step_t wird aktualisiert
          - Verhindert: unbemerkte offene Laufversuche

        • run_events als Kontroll-Register:
          status={started, completed, aborted}
          Technische Fehlversuche erhalten kein stop_event und gehen daher
          nicht als Beobachtung in die Statistik ein.

        • Resume: Bereits abgeschlossene condition/seed-Paare werden
          übersprungen. Fehlende Paare werden mit derselben experiment_id und
          derselben condition-spezifischen LTM-Collection fortgesetzt.
        """
        active_conditions = tuple(conditions if conditions is not None else CONDITIONS)
        manifest_config = self._build_manifest_config(
            runs_per_condition=runs_per_condition,
            max_budget=max_budget,
        )
        if resume_experiment_id:
            experiment_id = resume_experiment_id.strip()
            if not experiment_id:
                raise ValueError("resume_experiment_id must not be empty")
            self._validate_resume_state(
                experiment_id,
                active_conditions,
                runs_per_condition,
                manifest_config,
            )
            self._mark_open_resume_attempts(experiment_id)
        else:
            self._fail_on_open_runs_unless_allowed()
            experiment_id = generate_experiment_id()
            self._event_store.write_experiment_manifest(
                ExperimentManifest(
                    experiment_id=experiment_id,
                    config=manifest_config,
                    created_at_ms=wall_time_ms(),
                )
            )

        debug_path = self._event_store.db_path.parent / f"{experiment_id}_debug.jsonl"
        safe_touch_debug_file(debug_path)
        existing_run_events = self._event_store.get_run_events(experiment_id)
        run_counter = max((event.run_id for event in existing_run_events), default=0)
        completed_keys = {
            (event.condition, event.seed)
            for event in self._event_store.get_stop_events(experiment_id)
        }

        # Initialize and clear every condition-specific LTM collection exactly
        # once before the first run. Subsequent scope switches must not reset it,
        # because accumulation across rag_ltm seeds is the experimental treatment.
        persisted_conditions = {
            event.condition
            for event in (
                *self._event_store.get_run_events(experiment_id),
                *self._event_store.get_stop_events(experiment_id),
            )
        }
        conditions_to_initialize = active_conditions
        if resume_experiment_id:
            conditions_to_initialize = tuple(
                condition
                for condition in active_conditions
                if format_condition(*condition) not in persisted_conditions
            )
        for scenario, guardrail_level, memory_mode in conditions_to_initialize:
            condition = format_condition(scenario, guardrail_level, memory_mode)
            set_ltm_scope(self._memory_gateway, ltm_scope_name(experiment_id, condition))
            self._memory_gateway.reset_ltm()

        execution_schedule = build_execution_schedule(active_conditions, runs_per_condition)
        for scenario, guardrail_level, memory_mode, seed in execution_schedule:
            condition = format_condition(scenario, guardrail_level, memory_mode)
            if (condition, seed) in completed_keys:
                continue
            if memory_mode == "rag_ltm":
                set_ltm_scope(self._memory_gateway, ltm_scope_name(experiment_id, condition))

            config = ConfigSnapshot(experiment_id=experiment_id, seed=seed, scenario=scenario,
                                    guardrail_level=guardrail_level,
                                    memory_mode=memory_mode, model_cfg=dict(self._model_cfg),
                                    tool_allowlist=list(self._tool_allowlist),
                                    max_budget=max_budget)
            run_counter = self._execute_run(
                config=config,
                condition=condition,
                starting_run_id=run_counter,
                debug_path=debug_path,
            )

        return experiment_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_run(
            self,
            *,
            config: ConfigSnapshot,
            condition: str,
            starting_run_id: int,
            debug_path: Path,
    ) -> int:
        run_id = starting_run_id + 1
        last_step_t = 0
        self._write_run_event(
            experiment_id=config.experiment_id,
            run_id=run_id,
            condition=condition,
            seed=config.seed,
            status="started",
            last_step_t=0,
        )

        def step_event_sink(raw_ev: dict[str, Any]) -> None:
            nonlocal last_step_t
            step = build_step_event(
                raw_ev,
                experiment_id=config.experiment_id,
                run_id=run_id,
                condition=condition,
            )
            self._event_store.write_step_event(step)
            last_step_t = step.step_t

        try:
            self._prepare_run(config)
            result = self._run_callback(config, step_event_sink)
            self._persist_result(config.experiment_id, run_id, condition, config.seed, result)
            self._append_debug_entries(debug_path, run_id, result)
            self._write_run_event(
                experiment_id=config.experiment_id,
                run_id=run_id,
                condition=condition,
                seed=config.seed,
                status="completed",
                last_step_t=result.budget_used,
                stop_reason=result.stop_reason,
            )
            return run_id
        except Exception as exc:  # noqa: BLE001
            self._write_run_event(
                experiment_id=config.experiment_id,
                run_id=run_id,
                condition=condition,
                seed=config.seed,
                status="aborted",
                last_step_t=last_step_t,
                error=str(exc),
            )
            raise ExperimentInterruptedError(
                config.experiment_id,
                condition,
                config.seed,
                exc,
            ) from exc

    def _mark_open_resume_attempts(self, experiment_id: str) -> None:
        for event in self._event_store.get_open_run_events(experiment_id):
            self._write_run_event(
                experiment_id=experiment_id,
                run_id=event.run_id,
                condition=event.condition,
                seed=event.seed,
                status="aborted",
                last_step_t=event.last_step_t,
                stop_reason="resume_recovery",
                error="Process ended before a terminal run event; attempt superseded on resume.",
            )

    def _validate_resume_state(
            self,
            experiment_id: str,
            active_conditions: tuple[tuple[str, str, str], ...],
            runs_per_condition: int,
            manifest_config: dict[str, Any],
    ) -> None:
        """Reject resumes that change fixed configuration or corrupt seed space."""
        run_events = self._event_store.get_run_events(experiment_id)
        stop_events = self._event_store.get_stop_events(experiment_id)
        if not run_events and not stop_events:
            raise ValueError(
                f"Cannot resume unknown experiment_id {experiment_id!r}."
            )

        manifest = self._event_store.get_experiment_manifest(experiment_id)
        if manifest is None:
            raise ValueError(
                "Resume configuration cannot be verified because this experiment "
                "has no persisted manifest. Start a new experiment with the current "
                "version instead of resuming this legacy experiment."
            )
        differences = _manifest_differences(manifest.config, manifest_config)
        if differences:
            raise ValueError(
                "Resume configuration does not match the persisted experiment "
                f"manifest. Changed fields: {', '.join(differences[:8])}."
            )

        canonical_conditions = {
            format_condition(scenario, guardrail_level, memory_mode)
            for scenario, guardrail_level, memory_mode in CONDITIONS
        }
        requested_conditions = {
            format_condition(scenario, guardrail_level, memory_mode)
            for scenario, guardrail_level, memory_mode in active_conditions
        }
        if not requested_conditions <= canonical_conditions:
            unexpected = sorted(requested_conditions - canonical_conditions)
            raise ValueError(
                "Resume requested non-canonical conditions: "
                + ", ".join(unexpected)
            )

        persisted_keys = {
            (event.condition, event.seed)
            for event in (*run_events, *stop_events)
        }
        invalid_persisted = sorted(
            (condition, seed)
            for condition, seed in persisted_keys
            if condition not in canonical_conditions
            or seed not in range(1, runs_per_condition + 1)
        )
        if invalid_persisted:
            preview = ", ".join(
                f"{condition}/seed={seed}"
                for condition, seed in invalid_persisted[:5]
            )
            raise ValueError(
                "Persisted experiment data contains invalid condition/seed "
                f"entries: {preview}."
            )

        completed_counts: dict[tuple[str, int], int] = {}
        for event in stop_events:
            key = (event.condition, event.seed)
            completed_counts[key] = completed_counts.get(key, 0) + 1
        duplicates = sorted(
            key
            for key, count in completed_counts.items()
            if count > 1
        )
        if duplicates:
            preview = ", ".join(
                f"{condition}/seed={seed}"
                for condition, seed in duplicates[:5]
            )
            raise ValueError(
                "Cannot resume experiment with duplicate completed "
                f"condition/seed pairs: {preview}."
            )

    def _build_manifest_config(
        self,
        *,
        runs_per_condition: int,
        max_budget: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "runs_per_condition": runs_per_condition,
            "max_budget": max_budget,
            "target_model_cfg": dict(self._model_cfg),
            "tool_allowlist": list(self._tool_allowlist),
            "experiment_config": dict(self._experiment_config),
        }

    def _persist_result(self, experiment_id: str, run_id: int, condition: str, seed: int, result: RunResult) -> None:
        """Write all step_events and the stop_event for one run to EventStore."""
        if not result.step_events_persisted:
            for raw_ev in result.step_events:
                step = build_step_event(raw_ev, experiment_id=experiment_id, run_id=run_id, condition=condition)
                self._event_store.write_step_event(step)

        raw_stop = result.stop_event
        stop = StopEvent(experiment_id=experiment_id, run_id=run_id, condition=condition,
                         stop_reason=result.stop_reason, q2b=result.q2b,
                         censored_flag=result.censored_flag, timestamp_ms=int(raw_stop.get("timestamp_ms", 0)),
                         seed=seed)
        self._event_store.write_stop_event(stop)

    def _write_run_event(self, *, experiment_id: str, run_id: int, condition: str, seed: int, status: str,
                         last_step_t: int, stop_reason: str = "", error: str = "") -> None:
        self._event_store.write_run_event(
            RunEvent(experiment_id=experiment_id, run_id=run_id, condition=condition, seed=seed, status=status,
                     timestamp_ms=wall_time_ms(),
                     last_step_t=last_step_t, stop_reason=stop_reason, error=error)
        )

    def _fail_on_open_runs_unless_allowed(self) -> None:
        """Sicherheits-Check vor Experiment-Start: Blockiert bei offenen Runs aus vorherigen Crashes.

        WARUM das wichtig ist:
        - run_events.status='started' ohne zugehöriges stop_event deutet auf Crash hin
        - VectorDB (Chroma) könnte halbfertig gefüllt sein → Kontamination der LTM für nächsten Lauf
        - Ohne diesen Check könnten Episoden aus abgebrochenen Runs in neuen Lauf "zurück"fliessen
        - Das würde Experiment-Validität zerstören (nicht reproduzierbar, verunreinigte Daten)

        WARUM blockieren wir standardmäßig:
        - Conservative: Lieber sauber (neue DB) als fragwürdig (alter Zustand)
        - Mit --allow-incomplete-runs kann User explizit akzeptieren (nur nach manueller VectorDB-Prüfung!)
        """
        if self._allow_incomplete_runs:
            return
        open_runs = self._event_store.get_open_run_events()
        if not open_runs:
            return
        details = ", ".join(
            f"{ev.experiment_id}/run={ev.run_id}/{ev.condition}/seed={ev.seed}"
            for ev in open_runs[:3]
        )
        extra = "" if len(open_runs) <= 3 else f" (+{len(open_runs) - 3} more)"
        raise RuntimeError(
            "EventStore contains started runs without terminal lifecycle events: "
            f"{details}{extra}. Use a new --db-path, inspect run_events, or rerun "
            "with allow_incomplete_runs=True only after deciding how to handle the "
            "possibly contaminated VectorDB state."
        )

    def _append_debug_entries(self, debug_path: Path, run_id: int, result: RunResult) -> None:
        for raw_ev in result.step_events:
            step_t = int(raw_ev.get("step_index", 0))
            memory_context = raw_ev.get("memory_context", {})
            memory_entries = memory_context_length(memory_context)
            ltm_hits = ltm_hits_count(memory_context)
            record = {
                "run_id": run_id,
                "step_t": step_t,
                "planner_raw_output": str(
                    raw_ev.get("planner_raw_output", raw_ev.get("planner_raw_response", ""))
                ),
                "mutator_raw_output": str(
                    raw_ev.get("mutator_raw_output", raw_ev.get("mutator_raw_response", ""))
                ),
                "judge_raw_output": str(
                    raw_ev.get("judge_raw_output", raw_ev.get("judge_raw_response", ""))
                ),
                "memory_context_length": memory_entries,
                "ltm_hits_count": ltm_hits,
            }
            safe_append_jsonl(debug_path, record)


def _manifest_differences(
    persisted: dict[str, Any],
    requested: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    keys = sorted(set(persisted) | set(requested))
    for key in keys:
        path = f"{prefix}.{key}" if prefix else key
        left = persisted.get(key)
        right = requested.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(_manifest_differences(left, right, path))
        elif left != right:
            differences.append(path)
    return differences
