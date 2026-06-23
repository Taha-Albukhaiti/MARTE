from __future__ import annotations

from typing import Any, Mapping

from marte.memory.schemas import RetrievalBatch, RetrievalPhase
from marte.persistence.schemas import Episode


class NoOpStrategy:
    """MemoryStrategy for the ``stm_only`` condition.

    All LTM operations are no-ops: episodes are never embedded or stored
    beyond the STM ring buffer that the MemoryGateway manages directly.
    """

    def enrich_context(self, run_state: Mapping[str, Any]) -> list[Episode]:  # noqa: ARG002
        """Return an empty list — no LTM in stm_only condition."""
        return []

    def retrieve(
        self,
        *,
        query: str,
        phase: RetrievalPhase,
        filters: dict[str, Any],
        top_k: int | None = None,  # noqa: ARG002
        fetch_k: int | None = None,  # noqa: ARG002
        min_similarity: float | None = None,
    ) -> RetrievalBatch:
        """Return the same empty diagnostic shape used by the RAG condition."""
        return RetrievalBatch(
            phase=phase,
            query=query,
            filters=filters,
            min_similarity=min_similarity,
        )

    def upsert_episode(self, episode: Episode) -> None:  # noqa: ARG002
        """Discard episode — no persistent LTM in stm_only condition."""

    def upsert_episodes(self, episodes: list[Episode]) -> None:  # noqa: ARG002
        """Discard episodes — no persistent LTM in stm_only condition."""

    def reset_ltm(self) -> None:
        """No-op — there is no LTM to reset."""
