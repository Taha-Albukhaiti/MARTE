from __future__ import annotations

import pytest

from marte.attack.oracle import BreachOracle, SandboxPolicy
from marte.attack.orchestrator import RunOrchestrator
from marte.attack.planner import AttackPlan
from marte.attack.target_executor import ExecutionResult
from marte.llm.errors import LLMCompletionTruncatedError
from marte.llm.phase_retry import PhaseRetryPolicy
from marte.target.agent import StepResult
from marte.target.sandbox import SandboxSnapshot


def _step_result() -> StepResult:
    return StepResult(
        response_text="ok",
        tool_trace=[],
        policy_decisions=[],
        input_decision="allow",
        output_decision="allow",
        timings={"llm_ms": 1.0, "guardrail_ms": 1.0, "tool_ms": 1.0},
    )


def _snapshot_seq(n: int) -> list[SandboxSnapshot]:
    return [
        SandboxSnapshot(db_hash=f"db_{idx}", fs_hash=f"fs_{idx}")
        for idx in range(1, n + 1)
    ]


class SpyMemoryGateway:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.reset_calls = 0
        self.enrich_calls = 0
        self.call_log = call_log

    def reset_stm(self) -> None:
        self.reset_calls += 1
        if self.call_log is not None:
            self.call_log.append("reset_stm")

    def enrich_context(self, run_state):
        self.enrich_calls += 1
        if self.call_log is not None:
            self.call_log.append("memory")
        return {"step_index": run_state.get("step_index", 0)}


class AuditMemoryGateway(SpyMemoryGateway):
    def enrich_context(self, run_state):
        context = super().enrich_context(run_state)
        context["retrieval_trace"] = {
            "planner_positive": {
                "phase": "planner_positive",
                "query": "planner query",
                "filters": {"memory_kind": "positive"},
                "candidate_count": 1,
                "min_similarity": None,
                "hits": [],
            },
            "planner_blocked": {
                "phase": "planner_blocked",
                "query": "planner query",
                "filters": {"memory_kind": "blocked"},
                "candidate_count": 0,
                "min_similarity": None,
                "hits": [],
            },
        }
        return context

    def enrich_context_for_mutator(
        self,
        plan_strategy,
        plan_objective,
        run_state,
    ):
        return {
            "mutator_ltm_hits": [
                {
                    "strategy": plan_strategy,
                    "success_label": "success",
                    "mutation": "prior prompt",
                }
            ],
            "mutator_retrieval": {
                "phase": "mutator_positive",
                "query": plan_objective,
                "filters": {"strategy": plan_strategy},
                "candidate_count": 1,
                "min_similarity": None,
                "hits": [],
            },
        }


class SpyPlanner:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.calls = 0
        self.call_log = call_log
        self.last_raw_response = ""
        self.last_error = None
        self.last_selected_strategy = ""

    def plan(self, run_state, memory_context) -> AttackPlan:
        self.calls += 1
        if self.call_log is not None:
            self.call_log.append("planner")
        self.last_raw_response = '{"strategy":"schema_probe"}'
        self.last_error = None
        self.last_selected_strategy = "schema_probe"
        return AttackPlan(
            strategy="schema_probe",
            objective=f"obj-step-{memory_context.get('step_index', 0)}",
            rationale="test",
        )


class FailingPlanner:
    def plan(self, run_state, memory_context) -> AttackPlan:  # noqa: ARG002
        raise RuntimeError("planner down")


class SpyMutator:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.calls = 0
        self.call_log = call_log
        self.last_raw_response = ""
        self.last_sanitized_response = ""
        self.last_sanitization_actions = []
        self.last_used_fallback = False
        self.last_fallback_reason = ""
        self.memory_contexts = []

    def mutate(self, plan: AttackPlan, memory_context) -> str:
        self.calls += 1
        self.memory_contexts.append(dict(memory_context))
        if self.call_log is not None:
            self.call_log.append("mutator")
        self.last_raw_response = "prompt-body"
        self.last_sanitized_response = "prompt-body"
        self.last_sanitization_actions = []
        self.last_used_fallback = False
        self.last_fallback_reason = ""
        return f"prompt::{plan.strategy}::{memory_context.get('step_index', 0)}"


class SequenceExecutor:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.call_log = call_log

    def execute(self, prompt: str) -> ExecutionResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.call_log is not None:
            self.call_log.append("executor")

        return ExecutionResult(
            prompt=prompt,
            step_result=_step_result(),
            error=None,
        )


class ErrorResultExecutor(SequenceExecutor):
    def execute(self, prompt: str) -> ExecutionResult:
        self.calls += 1
        return ExecutionResult(
            prompt=prompt,
            step_result=_step_result(),
            error="target execution failed: target unavailable",
        )


class TruncatedOnceExecutor(SequenceExecutor):
    def execute(self, prompt: str) -> ExecutionResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise LLMCompletionTruncatedError(
                role="target",
                model="gpt-oss-120b",
                finish_reason="length",
                output_limit=4096,
                prompt_tokens=100,
                completion_tokens=4096,
                reasoning_tokens=4000,
            )
        return ExecutionResult(
            prompt=prompt,
            step_result=_step_result(),
            error=None,
        )


class SpySnapshotProvider:
    def __init__(self, snapshots: list[SandboxSnapshot], call_log: list[str] | None = None) -> None:
        self._snapshots = list(snapshots)
        self.calls = 0
        self.call_log = call_log

    def snapshot(self) -> SandboxSnapshot:
        self.calls += 1
        if self.call_log is not None:
            label = "snapshot_pre" if self.calls % 2 == 1 else "snapshot_post"
            self.call_log.append(label)
        if not self._snapshots:
            raise RuntimeError("no snapshot configured")
        return self._snapshots.pop(0)


class SequenceOracle:
    def __init__(self, verdicts: list[bool], call_log: list[str] | None = None) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0
        self.call_log = call_log

    def check(self, pre_snapshot, post_snapshot, tool_trace, sandbox_policy) -> bool:  # noqa: ARG002
        self.calls += 1
        if self.call_log is not None:
            self.call_log.append("oracle")
        idx = min(self.calls - 1, len(self._verdicts) - 1)
        return self._verdicts[idx]


class SpyJudge:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.calls = 0
        self.call_log = call_log
        self.strategies: list[str] = []

    def classify(self, execution_result: ExecutionResult, oracle_verdict: bool, strategy: str = "") -> str:  # noqa: ARG002
        self.calls += 1
        self.strategies.append(strategy)
        if self.call_log is not None:
            self.call_log.append("judge")
        return "success" if oracle_verdict else "partial"


class TruncatedOnceJudge(SpyJudge):
    def classify(
        self,
        execution_result: ExecutionResult,
        oracle_verdict: bool,
        strategy: str = "",
    ) -> str:
        self.calls += 1
        self.strategies.append(strategy)
        if self.calls == 1:
            raise LLMCompletionTruncatedError(
                role="judge",
                model="glm-4.7",
                finish_reason="length",
                output_limit=768,
                prompt_tokens=100,
                completion_tokens=768,
                reasoning_tokens=None,
            )
        return "success" if oracle_verdict else "partial"


class FailingJudge:
    def classify(self, execution_result: ExecutionResult, oracle_verdict: bool, strategy: str = "") -> str:  # noqa: ARG002
        raise RuntimeError("judge unavailable")


class FailingOracle(BreachOracle):
    def check(self, pre_snapshot, post_snapshot, tool_trace, sandbox_policy) -> bool:  # noqa: ARG002
        raise RuntimeError("oracle unavailable")


def test_orchestrator_exposes_planner_and_mutator_accessors() -> None:
    planner = SpyPlanner()
    mutator = SpyMutator()
    orchestrator = RunOrchestrator(
        planner=planner,
        mutator=mutator,
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    assert orchestrator.planner is planner
    assert orchestrator.mutator is mutator


def test_run_stops_on_breach_and_sets_q2b() -> None:
    policy = SandboxPolicy(scenario="sql")
    memory = SpyMemoryGateway()
    planner = SpyPlanner()
    mutator = SpyMutator()
    executor = SequenceExecutor()
    snapshots = SpySnapshotProvider(_snapshot_seq(10))
    oracle = SequenceOracle([False, True, True])
    judge = SpyJudge()

    orchestrator = RunOrchestrator(
        planner=planner,
        mutator=mutator,
        executor=executor,
        judge=judge,
        snapshot_provider=snapshots,
        oracle=oracle,
        memory_gateway=memory,
    )

    result = orchestrator.run(max_budget=5, sandbox_policy=policy)

    assert result.stop_reason == "breach_stop"
    assert result.q2b == 2
    assert result.censored_flag is False
    assert result.budget_used == 2
    assert len(result.step_events) == 2
    assert memory.reset_calls == 1
    assert planner.calls == 2
    assert mutator.calls == 2
    assert executor.calls == 2
    assert snapshots.calls == 4
    assert oracle.calls == 2
    assert judge.calls == 2
    assert judge.strategies == ["schema_probe", "schema_probe"]


def test_run_stop_event_records_q2b_one_for_first_step_breach() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([True]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=50, sandbox_policy=SandboxPolicy(scenario="sql"))

    assert result.stop_reason == "breach_stop"
    assert result.q2b == 1
    assert result.stop_event["q2b"] == 1
    assert result.stop_event["censored_flag"] is False


def test_run_stops_on_budget_exhausted_when_no_breach() -> None:
    policy = SandboxPolicy(scenario="sql")

    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(6)),
        oracle=SequenceOracle([False, False, False]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=3, sandbox_policy=policy)

    assert result.stop_reason == "budget_exhausted"
    assert result.censored_flag is True
    assert result.q2b == 3
    assert result.budget_used == 3
    assert len(result.step_events) == 3
    assert result.stop_event["stop_reason"] == "budget_exhausted"


def test_pipeline_order_is_planner_mutator_snapshot_execute_oracle_judge_per_step() -> None:
    calls: list[str] = []

    orchestrator = RunOrchestrator(
        planner=SpyPlanner(call_log=calls),
        mutator=SpyMutator(call_log=calls),
        executor=SequenceExecutor(call_log=calls),
        judge=SpyJudge(call_log=calls),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2), call_log=calls),
        oracle=SequenceOracle([True], call_log=calls),
        memory_gateway=SpyMemoryGateway(call_log=calls),
    )

    result = orchestrator.run(max_budget=3, sandbox_policy=SandboxPolicy(scenario="sql"))

    assert result.stop_reason == "breach_stop"
    assert calls == [
        "reset_stm",
        "memory",
        "planner",
        "mutator",
        "snapshot_pre",
        "executor",
        "snapshot_post",
        "oracle",
        "judge",
    ]


def test_step_event_sink_runs_after_judge_and_before_decision() -> None:
    calls: list[str] = []
    persisted: list[dict] = []

    def sink(event: dict) -> None:
        calls.append("write_event")
        persisted.append(event)

    orchestrator = RunOrchestrator(
        planner=SpyPlanner(call_log=calls),
        mutator=SpyMutator(call_log=calls),
        executor=SequenceExecutor(call_log=calls),
        judge=SpyJudge(call_log=calls),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2), call_log=calls),
        oracle=SequenceOracle([True], call_log=calls),
        memory_gateway=SpyMemoryGateway(call_log=calls),
    )

    result = orchestrator.run(
        max_budget=3,
        sandbox_policy=SandboxPolicy(scenario="sql"),
        step_event_sink=sink,
    )

    assert result.step_events_persisted is True
    assert len(persisted) == 1
    assert persisted[0]["step_index"] == 1
    assert calls == [
        "reset_stm",
        "memory",
        "planner",
        "mutator",
        "snapshot_pre",
        "executor",
        "snapshot_post",
        "oracle",
        "judge",
        "write_event",
    ]


def test_orchestrator_survives_planner_and_judge_failures() -> None:
    orchestrator = RunOrchestrator(
        planner=FailingPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=FailingJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=1, sandbox_policy=SandboxPolicy(scenario="sql"))

    assert result.stop_reason == "budget_exhausted"
    assert result.q2b == 1
    assert result.censored_flag is True
    assert len(result.step_events) == 1
    assert any("planner failed" in err for err in result.step_events[0]["errors"])
    assert any("judge failed" in err for err in result.step_events[0]["errors"])


def test_orchestrator_aborts_on_oracle_failure() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=FailingOracle(),
        memory_gateway=SpyMemoryGateway(),
    )

    with pytest.raises(RuntimeError, match="oracle failed"):
        orchestrator.run(
            max_budget=1,
            sandbox_policy=SandboxPolicy(scenario="sql"),
        )


def test_orchestrator_aborts_on_target_execution_error() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=ErrorResultExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(1)),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    with pytest.raises(RuntimeError, match="Target execution failed"):
        orchestrator.run(
            max_budget=1,
            sandbox_policy=SandboxPolicy(scenario="sql"),
        )


def test_orchestrator_retries_truncated_target_in_same_step() -> None:
    oracle = SequenceOracle([False])
    judge = SpyJudge()
    executor = TruncatedOnceExecutor()
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=executor,
        judge=judge,
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=oracle,
        memory_gateway=SpyMemoryGateway(),
        phase_retry_policy=PhaseRetryPolicy(
            max_attempts=2,
            backoff_s=0,
            max_backoff_s=0,
        ),
    )

    result = orchestrator.run(
        max_budget=1,
        sandbox_policy=SandboxPolicy(scenario="sql"),
    )

    assert result.stop_reason == "budget_exhausted"
    assert result.budget_used == 1
    assert len(result.step_events) == 1
    assert executor.calls == 2
    assert executor.prompts == [
        "prompt::schema_probe::1",
        "prompt::schema_probe::1",
    ]
    assert oracle.calls == 1
    assert judge.calls == 1
    assert result.step_events[0]["timings"]["technical_retries"] == {
        "target": 1
    }


def test_orchestrator_retries_judge_without_reexecuting_target_or_oracle() -> None:
    executor = SequenceExecutor()
    oracle = SequenceOracle([False])
    judge = TruncatedOnceJudge()
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=executor,
        judge=judge,
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=oracle,
        memory_gateway=SpyMemoryGateway(),
        phase_retry_policy=PhaseRetryPolicy(
            max_attempts=2,
            backoff_s=0,
            max_backoff_s=0,
        ),
    )

    result = orchestrator.run(
        max_budget=1,
        sandbox_policy=SandboxPolicy(scenario="sql"),
    )

    assert result.budget_used == 1
    assert executor.calls == 1
    assert oracle.calls == 1
    assert judge.calls == 2
    assert result.step_events[0]["timings"]["technical_retries"] == {
        "judge": 1
    }


class FailingEnrichGateway:
    """Memory gateway whose enrich_context always raises."""

    def reset_stm(self) -> None:
        return None

    def enrich_context(self, run_state):
        raise RuntimeError("memory backend down")


class FailingSnapshotProvider:
    def snapshot(self):
        raise RuntimeError("disk error")


def test_orchestrator_survives_enrich_context_failure() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=FailingEnrichGateway(),
    )

    result = orchestrator.run(max_budget=1, sandbox_policy=SandboxPolicy(scenario="sql"))

    assert result.stop_reason == "budget_exhausted"
    assert len(result.step_events) == 1
    # No enrich error is propagated to errors list (silent fallback to {})
    # but the step must have completed without crash
    assert result.step_events[0]["step_index"] == 1


def test_orchestrator_fails_fast_on_enrich_failure_in_rag_mode() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=FailingEnrichGateway(),
    )

    with pytest.raises(RuntimeError, match="rag_ltm"):
        orchestrator.run(
            max_budget=1,
            sandbox_policy=SandboxPolicy(scenario="sql"),
            initial_run_state={"memory_mode": "rag_ltm"},
        )


def test_orchestrator_aborts_on_snapshot_provider_failure() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=FailingSnapshotProvider(),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        orchestrator.run(
            max_budget=1,
            sandbox_policy=SandboxPolicy(scenario="sql"),
        )


def test_run_state_propagates_between_steps() -> None:
    policy = SandboxPolicy(scenario="sql")
    planner = SpyPlanner()
    mutator = SpyMutator()
    executor = SequenceExecutor()

    orchestrator = RunOrchestrator(
        planner=planner,
        mutator=mutator,
        executor=executor,
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(6)),
        oracle=SequenceOracle([False, False, True]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=5, sandbox_policy=policy)

    assert result.stop_reason == "breach_stop"
    assert result.run_state["steps_executed"] == 3
    assert result.run_state["last_strategy"] == "schema_probe"
    assert result.run_state["last_oracle_verdict"] is True
    assert result.run_state["last_judge_verdict"] == "success"
    assert "attempted_strategies" not in result.run_state


def test_step_event_contains_planner_mutator_debug_fields() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=1, sandbox_policy=SandboxPolicy(scenario="sql"))
    event = result.step_events[0]

    assert event["memory_context"] == {"step_index": 1}
    assert event["planner_raw_response"] == '{"strategy":"schema_probe"}'
    assert event["planner_selected_strategy"] == "schema_probe"
    assert event["planner_error"] is None
    assert event["mutator_raw_response"] == "prompt-body"
    assert event["mutator_sanitized_response"] == "prompt-body"
    assert event["mutator_sanitization_actions"] == []
    assert event["mutator_used_fallback"] is False
    assert event["mutator_fallback_reason"] == ""


def test_orchestrator_persists_all_three_retrieval_phases() -> None:
    mutator = SpyMutator()
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=mutator,
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=AuditMemoryGateway(),
    )

    result = orchestrator.run(
        max_budget=1,
        sandbox_policy=SandboxPolicy(scenario="sql"),
    )

    trace = result.step_events[0]["retrieval_trace"]
    assert set(trace) == {
        "planner_positive",
        "planner_blocked",
        "mutator_positive",
    }
    assert trace["mutator_positive"]["filters"]["strategy"] == "schema_probe"
    assert mutator.memory_contexts[0]["mutator_ltm_hits"][0]["strategy"] == (
        "schema_probe"
    )


def test_timestamps_are_run_relative_durations_not_epoch() -> None:
    orchestrator = RunOrchestrator(
        planner=SpyPlanner(),
        mutator=SpyMutator(),
        executor=SequenceExecutor(),
        judge=SpyJudge(),
        snapshot_provider=SpySnapshotProvider(_snapshot_seq(2)),
        oracle=SequenceOracle([False]),
        memory_gateway=SpyMemoryGateway(),
    )

    result = orchestrator.run(max_budget=1, sandbox_policy=SandboxPolicy(scenario="sql"))
    step_ts = int(result.step_events[0]["timestamp_ms"])
    stop_ts = int(result.stop_event["timestamp_ms"])

    # Duration timestamps are measured from run start and should be small in tests.
    assert step_ts >= 0
    assert stop_ts >= 0
    assert stop_ts >= step_ts
    assert stop_ts < 60_000
