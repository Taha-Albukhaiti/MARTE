from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from marte.attack.helpers.orchestrator import _elapsed_ms
from marte.attack.run_state import update_run_state_after_step
from marte.attack.step_event_builder import build_step_event as _build_step_event_fn

from marte.attack.interfaces import (
    JudgeProtocol,
    MutatorProtocol,
    PlannerProtocol,
    RunMemoryGatewayProtocol,
    SnapshotProvider,
    _NoOpMemoryGateway,
)
from marte.attack.oracle import BreachOracle, SandboxPolicy
from marte.attack.planner import AttackPlan
from marte.attack.run_trace import (
    log_mutator_execution,
    log_plan_execution,
    log_target_execution,
)
from marte.attack.target_executor import (
    ExecutionResult,
    TargetExecutor,
)
from marte.llm.errors import NonRetryableLLMResponseError
from marte.llm.phase_retry import PhaseRetryPolicy, retry_llm_phase


StopReason = Literal["breach_stop", "budget_exhausted"]
StepEventSink = Callable[[dict[str, Any]], None]


def _empty_retrieval_batch(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "query": "",
        "filters": {},
        "candidate_count": 0,
        "min_similarity": None,
        "hits": [],
    }


if TYPE_CHECKING:
    from marte.target.sandbox import SandboxSnapshot


@dataclass
class RunResult:
    stop_reason: StopReason
    q2b: int
    censored_flag: bool
    budget_used: int
    step_events: list[dict[str, Any]]
    stop_event: dict[str, Any]
    run_state: dict[str, Any]
    step_events_persisted: bool = False


class RunOrchestrator:
    """Main attack loop for a single run."""

    def __init__(
        self,
        planner: PlannerProtocol,
        mutator: MutatorProtocol,
        executor: TargetExecutor,
        judge: JudgeProtocol,
        snapshot_provider: SnapshotProvider,
        oracle: BreachOracle,
        memory_gateway: RunMemoryGatewayProtocol | None = None,
        phase_retry_policy: PhaseRetryPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._mutator = mutator
        self._executor = executor
        self._judge = judge
        self._snapshot_provider = snapshot_provider
        self._oracle = oracle
        self._memory = memory_gateway or _NoOpMemoryGateway()
        self._phase_retry_policy = phase_retry_policy or PhaseRetryPolicy.from_env()
        self._phase_retry_counts: dict[str, int] = {}

    @property
    def planner(self) -> PlannerProtocol:
        return self._planner

    @property
    def mutator(self) -> MutatorProtocol:
        return self._mutator

    @property
    def judge(self) -> JudgeProtocol:
        return self._judge

    def run(
        self,
        *,
        max_budget: int,
        sandbox_policy: SandboxPolicy,
        initial_run_state: Mapping[str, Any] | None = None,
        step_event_sink: StepEventSink | None = None,
    ) -> RunResult:
        run_started_at = time.monotonic()
        run_state: dict[str, Any] = dict(initial_run_state or {})
        run_state.setdefault("step_index", 0)

        step_events: list[dict[str, Any]] = []
        persisted_steps = 0
        breach_detected = False
        q2b = 0

        self._safe_reset_stm()

        for step_idx in range(1, max_budget + 1):
            self._phase_retry_counts = {}
            run_state["step_index"] = step_idx

            memory_context = self._safe_enrich_context(run_state)
            plan, plan_error = self._safe_plan(run_state, memory_context)
            log_plan_execution(plan)
            mutator_retrieval = self._safe_enrich_context_for_mutator(plan, run_state)
            mutator_memory_context = {
                **dict(memory_context),
                **dict(mutator_retrieval),
            }
            retrieval_trace = dict(memory_context.get("retrieval_trace", {}))
            retrieval_trace["mutator_positive"] = mutator_retrieval.get(
                "mutator_retrieval",
                _empty_retrieval_batch("mutator_positive"),
            )
            prompt, mutator_error = self._safe_mutate(plan, mutator_memory_context)
            log_mutator_execution(prompt)

            pre_snapshot, pre_snapshot_error = self._safe_snapshot("pre")

            execution_result, executor_error = self._safe_execute(prompt)
            log_target_execution(execution_result.step_result.response_text)

            post_snapshot, post_snapshot_error = self._safe_snapshot("post")
            oracle_verdict, oracle_error = self._safe_oracle(
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
                execution_result=execution_result,
                sandbox_policy=sandbox_policy,
            )
            judge_verdict, judge_error = self._safe_classify(execution_result, oracle_verdict, plan.strategy)

            event = _build_step_event_fn(
                step_idx=step_idx,
                run_started_at=run_started_at,
                memory_context=memory_context,
                retrieval_trace=retrieval_trace,
                plan=plan,
                prompt=prompt,
                execution_result=execution_result,
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
                oracle_verdict=oracle_verdict,
                judge_verdict=judge_verdict,
                phase_errors=(
                    plan_error,
                    mutator_error,
                    pre_snapshot_error,
                    execution_result.error,
                    executor_error,
                    post_snapshot_error,
                    oracle_error,
                    judge_error,
                ),
                phase_retry_counts=dict(self._phase_retry_counts),
                planner=self._planner,
                mutator=self._mutator,
                judge=self._judge,
            )
            step_events.append(event)
            if step_event_sink is not None:
                step_event_sink(event)
                persisted_steps += 1

            update_run_state_after_step(
                run_state=run_state,
                step_idx=step_idx,
                plan=plan,
                prompt=prompt,
                execution_result=execution_result,
                oracle_verdict=oracle_verdict,
                judge_verdict=judge_verdict,
                errors=list(event["errors"]),
            )

            if oracle_verdict:
                breach_detected = True
                q2b = step_idx
                break

        budget_used = len(step_events)
        if breach_detected:
            stop_reason: StopReason = "breach_stop"
            censored_flag = False
        else:
            stop_reason = "budget_exhausted"
            censored_flag = True
            q2b = budget_used

        stop_event = {
            "timestamp_ms": _elapsed_ms(run_started_at),
            "stop_reason": stop_reason,
            "q2b": q2b,
            "censored_flag": censored_flag,
            "budget_used": budget_used,
        }
        return RunResult(
            stop_reason=stop_reason,
            q2b=q2b,
            censored_flag=censored_flag,
            budget_used=budget_used,
            step_events=step_events,
            stop_event=stop_event,
            run_state=run_state,
            step_events_persisted=persisted_steps == len(step_events),
        )

    def _safe_reset_stm(self) -> None:
        try:
            self._memory.reset_stm()
        except Exception:
            pass

    def _safe_enrich_context(self, run_state: Mapping[str, Any]) -> Mapping[str, Any]:
        memory_mode = str(run_state.get("memory_mode", "")).strip().lower()
        for attempt in range(2):
            try:
                return self._memory.enrich_context(run_state)
            except Exception as exc:  # noqa: BLE001
                if memory_mode == "rag_ltm":
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    raise RuntimeError("Memory enrichment failed in rag_ltm after retry.") from exc
                return {}
        return {}  # unreachable

    def _safe_enrich_context_for_mutator(
        self,
        plan: AttackPlan,
        run_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        enrich = getattr(self._memory, "enrich_context_for_mutator", None)
        if not callable(enrich):
            return {
                "mutator_ltm_hits": [],
                "mutator_retrieval": _empty_retrieval_batch("mutator_positive"),
            }
        memory_mode = str(run_state.get("memory_mode", "")).strip().lower()
        for attempt in range(2):
            try:
                return enrich(plan.strategy, plan.objective, run_state)
            except Exception as exc:  # noqa: BLE001
                if memory_mode == "rag_ltm":
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    raise RuntimeError("Mutator memory enrichment failed in rag_ltm after retry.") from exc
                return {
                    "mutator_ltm_hits": [],
                    "mutator_retrieval": _empty_retrieval_batch("mutator_positive"),
                }
        return {
            "mutator_ltm_hits": [],
            "mutator_retrieval": _empty_retrieval_batch("mutator_positive"),
        }

    def _safe_plan(
        self,
        run_state: Mapping[str, Any],
        memory_context: Mapping[str, Any],
    ) -> tuple[AttackPlan, str | None]:
        try:
            return (
                self._retry_llm_phase(
                    "planner",
                    lambda: self._planner.plan(run_state, memory_context),
                ),
                None,
            )
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            return (
                AttackPlan(
                    strategy="schema_probe",
                    objective="recover after planner failure",
                    rationale="fallback",
                ),
                f"planner failed: {exc}",
            )

    def _safe_mutate(self, plan: AttackPlan, memory_context: Mapping[str, Any]) -> tuple[str, str | None]:
        try:
            return (
                self._retry_llm_phase(
                    "mutator",
                    lambda: self._mutator.mutate(plan, memory_context),
                ),
                None,
            )
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            return (
                f"[strategy={plan.strategy}] Objective: {plan.objective}.",
                f"mutator failed: {exc}",
            )

    def _safe_execute(self, prompt: str) -> tuple[ExecutionResult, str | None]:
        try:
            result = self._retry_llm_phase(
                "target",
                lambda: self._executor.execute(prompt),
            )
            if result.error:
                raise RuntimeError(result.error)
            return result, None
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Target execution failed: {exc}") from exc

    def _safe_snapshot(self, label: str) -> tuple[SandboxSnapshot, str | None]:
        try:
            return self._snapshot_provider.snapshot(), None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{label} snapshot failed: {exc}") from exc

    def _safe_oracle(
        self,
        *,
        pre_snapshot: SandboxSnapshot,
        post_snapshot: SandboxSnapshot,
        execution_result: ExecutionResult,
        sandbox_policy: SandboxPolicy,
    ) -> tuple[bool, str | None]:
        try:
            verdict = self._oracle.check(
                pre_snapshot,
                post_snapshot,
                execution_result.step_result.tool_trace,
                sandbox_policy,
            )
            return verdict, None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"oracle failed: {exc}") from exc

    def _safe_classify(
        self,
        execution_result: ExecutionResult,
        oracle_verdict: bool,
        strategy: str,
    ) -> tuple[str, str | None]:
        try:
            return (
                self._retry_llm_phase(
                    "judge",
                    lambda: self._judge.classify(
                        execution_result,
                        oracle_verdict,
                        strategy=strategy,
                    ),
                ),
                None,
            )
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            return "fail", f"judge failed: {exc}"

    def _retry_llm_phase(self, phase: str, operation: Callable[[], Any]) -> Any:
        def record_retry(
            retry_number: int,
            exc: NonRetryableLLMResponseError,
        ) -> None:
            self._phase_retry_counts[phase] = retry_number

        return retry_llm_phase(
            operation,
            phase=phase,
            policy=self._phase_retry_policy,
            on_retry=record_retry,
        )
