from __future__ import annotations

from typing import Any, Mapping

from marte.memory.gateway import MemoryGateway
from marte.memory.noop import NoOpStrategy
from marte.memory.strategy import MemoryStrategy


class SwitchingMemoryGateway:
    """Switch memory mode per run while preserving condition-level LTM semantics.

    - ``stm_only``: routes to a gateway with NoOpStrategy.
    - ``rag_ltm``: routes to a gateway with the injected real strategy.

    RunOrchestrator resets STM at run start, ComparativeRunner resets LTM at
    condition boundaries.
    """

    def __init__(self, rag_strategy: MemoryStrategy, stm_capacity: int = 5) -> None:
        self._rag_strategy = rag_strategy
        self._stm = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=stm_capacity)
        self._rag = MemoryGateway(strategy=rag_strategy, stm_capacity=stm_capacity)
        self._active = self._stm

    def set_mode(self, memory_mode: str) -> None:
        mode = memory_mode.strip().lower()
        if mode == "rag_ltm":
            self._assert_rag_ready()
            self._active = self._rag
            return
        self._active = self._stm

    def enrich_context(self, run_state: Mapping[str, Any]) -> dict[str, Any]:
        return self._active.enrich_context(run_state)

    def enrich_context_for_mutator(
        self,
        plan_strategy: str,
        plan_objective: str,
        run_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._active.enrich_context_for_mutator(
            plan_strategy,
            plan_objective,
            run_state,
        )

    def upsert_episode(self, episode: dict[str, Any]) -> None:
        self._active.upsert_episode(episode)

    def reset_stm(self) -> None:
        self._active.reset_stm()

    def reset_ltm(self) -> None:
        self._stm.reset_ltm()
        self._rag.reset_ltm()

    def set_ltm_scope(self, scope_name: str) -> None:
        set_collection_name = getattr(self._rag_strategy, "set_collection_name", None)
        if callable(set_collection_name):
            set_collection_name(scope_name)

    @property
    def active_mode(self) -> str:
        return "rag_ltm" if self._active is self._rag else "stm_only"

    def _assert_rag_ready(self) -> None:
        validate_config = getattr(
            self._rag_strategy,
            "validate_experiment_config",
            None,
        )
        if callable(validate_config):
            try:
                validate_config()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Cannot activate memory_mode='rag_ltm': "
                    f"RAG configuration is invalid. {exc}"
                ) from exc

        health_check = getattr(self._rag_strategy, "health_check", None)
        if not callable(health_check):
            return
        try:
            health_check()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Cannot activate memory_mode='rag_ltm': RAG backend health check failed."
            ) from exc
