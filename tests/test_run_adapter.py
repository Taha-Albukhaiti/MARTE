from __future__ import annotations

from typing import Any

from marte.attack.orchestrator import RunResult
from marte.cli.run_adapter import ExperimentRunAdapter
from marte.config.snapshot import ConfigSnapshot


class _SeedRecorder:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeOrchestrator:
    def __init__(self, judge: Any) -> None:
        self.judge = judge
        self.run_kwargs: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> RunResult:
        self.run_kwargs.append(kwargs)
        return RunResult(
            stop_reason="budget_exhausted",
            q2b=3,
            censored_flag=True,
            budget_used=3,
            step_events=[],
            stop_event={"timestamp_ms": 1000},
            run_state={},
        )


def _config(seed: int = 7, scenario: str = "sql") -> ConfigSnapshot:
    return ConfigSnapshot(
        experiment_id="exp-run-adapter",
        seed=seed,
        scenario=scenario,
        guardrail_level="moderate",
        model_cfg={},
        tool_allowlist=[scenario],
        max_budget=3,
        memory_mode="stm_only",
    )


def _adapter(
    *,
    planner: _SeedRecorder | None = None,
    mutator: _SeedRecorder | None = None,
    judge: Any | None = None,
) -> tuple[ExperimentRunAdapter, _SeedRecorder, _SeedRecorder, Any, _FakeOrchestrator]:
    planner = planner or _SeedRecorder()
    mutator = mutator or _SeedRecorder()
    judge = judge or _SeedRecorder()
    orchestrator = _FakeOrchestrator(judge)
    adapter = ExperimentRunAdapter(
        sandbox=object(),  # not used by run_once()
        memory_gateway=object(),  # not used by run_once()
        target_agent=object(),  # not used by run_once()
        planner=planner,
        mutator=mutator,
        judge=judge,
        orchestrator=orchestrator,
        conditions=[("sql", "moderate", "stm_only"), ("file", "moderate", "stm_only")],
        runs_per_condition=3,
    )
    return adapter, planner, mutator, judge, orchestrator


def test_run_adapter_seeds_planner_mutator_and_judge_before_run() -> None:
    adapter, planner, mutator, judge, orchestrator = _adapter()

    adapter.run_once(_config(seed=7))

    assert planner.seeds == [7]
    assert mutator.seeds == [7]
    assert judge.seeds == [7]
    assert orchestrator.run_kwargs[0]["initial_run_state"]["seed"] == 7


def test_run_adapter_sets_policy_neutral_scenario_goal() -> None:
    sql_adapter, _, _, _, sql_orchestrator = _adapter()
    file_adapter, _, _, _, file_orchestrator = _adapter()

    sql_adapter.run_once(_config(seed=1, scenario="sql"))
    file_adapter.run_once(_config(seed=1, scenario="file"))

    sql_goal = sql_orchestrator.run_kwargs[0]["initial_run_state"]["goal"]
    file_goal = file_orchestrator.run_kwargs[0]["initial_run_state"]["goal"]

    assert "SQL" in sql_goal
    assert "filesystem" in file_goal
    assert "credentials" not in sql_goal
    assert "secret" not in file_goal
