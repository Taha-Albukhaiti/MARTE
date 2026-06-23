from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:
    from marte.attack.planner import AttackPlan
    from marte.attack.target_executor import ExecutionResult
    from marte.target.sandbox import SandboxSnapshot


class PlannerProtocol(Protocol):
    def plan(self, run_state: Mapping[str, Any] | None, memory_context: Mapping[str, Any] | None) -> AttackPlan:
        ...


class MutatorProtocol(Protocol):
    def mutate(self, plan: AttackPlan, memory_context: Mapping[str, Any] | None) -> str:
        ...


class JudgeProtocol(Protocol):
    def classify(self, execution_result: ExecutionResult, oracle_verdict: bool, strategy: str = "") -> str:
        ...


class RunMemoryGatewayProtocol(Protocol):
    """Memory gateway shape required by the RunOrchestrator."""

    def enrich_context(self, run_state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def enrich_context_for_mutator(
        self,
        plan_strategy: str,
        plan_objective: str,
        run_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def reset_stm(self) -> None:
        ...


class JudgeMemoryGatewayProtocol(Protocol):
    """Memory gateway shape required by Judge for episode persistence."""

    def upsert_episode(self, episode: dict[str, Any]) -> None:
        ...


class SnapshotProvider(Protocol):
    def snapshot(self) -> SandboxSnapshot:
        ...


class _NoOpMemoryGateway:
    """Satisfies both RunMemoryGatewayProtocol and JudgeMemoryGatewayProtocol."""

    def enrich_context(self, run_state: Mapping[str, Any]) -> Mapping[str, Any]:  # noqa: ARG002
        return {}

    def enrich_context_for_mutator(
        self,
        plan_strategy: str,  # noqa: ARG002
        plan_objective: str,  # noqa: ARG002
        run_state: Mapping[str, Any],  # noqa: ARG002
    ) -> Mapping[str, Any]:
        return {}

    def reset_stm(self) -> None:
        pass

    def upsert_episode(self, episode: dict[str, Any]) -> None:  # noqa: ARG002
        pass
