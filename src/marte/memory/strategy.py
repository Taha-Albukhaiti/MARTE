from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from marte.memory.schemas import RetrievalBatch, RetrievalPhase
from marte.persistence.schemas import Episode


@runtime_checkable
class MemoryStrategy(Protocol):
    """DI contract for the LTM backend.

    Both experimental conditions share the same MemoryGateway code path.
    Only the injected MemoryStrategy differs:
    - ``stm_only`` → NoOpStrategy (no LTM, episodes discarded after append to STM)
    - ``rag_ltm``  → RAGAdapter  (episodes embedded and stored in VectorDB)
    """

    def enrich_context(self, run_state: Mapping[str, Any]) -> list[Episode]:
        """Return LTM hits relevant to the current run state.

        Args:
            run_state: Current run state dict (scenario, goal, last_prompt, …).

        Returns:
            Ordered list of retrieved LTM episodes (empty for NoOp).
        """
        ...

    def retrieve(
        self,
        *,
        query: str,
        phase: RetrievalPhase,
        filters: dict[str, Any],
        top_k: int | None = None,
        fetch_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalBatch:
        """Return an auditable retrieval batch for one pipeline phase."""
        ...

    def upsert_episode(self, episode: Episode) -> None:
        """Persist a labeled episode into the LTM store.

        Args:
            episode: Fully constructed Episode to store.
        """
        ...

    def upsert_episodes(self, episodes: list[Episode]) -> None:
        """Persist one completed run's episodes as a batch."""
        ...

    def reset_ltm(self) -> None:
        """Delete all LTM data (called at Condition boundary by ComparativeRunner)."""
        ...
