from __future__ import annotations

from typing import Any, Mapping

from marte.memory.retrieval_query import _current_memory_run_id
from marte.memory.retrieval_results import _deduplicate_hits
from marte.memory.schemas import RetrievalBatch, RetrievalHit, RetrievalPhase
from marte.persistence.schemas import Episode


class InMemoryEpisodeStrategy:
    """Simple in-memory LTM strategy for local CLI runs.

    This is intentionally lightweight and deterministic. It is not used for
    real experiments where RAGAdapter/VectorDB should be injected.
    """

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def enrich_context(self, run_state: Mapping[str, Any]) -> list[Episode]:
        scenario = str(run_state.get("scenario", "")).strip()
        guardrail_level = str(run_state.get("guardrail_level", "")).strip()
        memory_run_id = _current_memory_run_id(run_state)
        return [
            episode
            for episode in self._episodes
            if (not scenario or episode.scenario == scenario)
            and (not guardrail_level or episode.guardrail_level == guardrail_level)
            # Local LTM follows the same no intra-run retrieval rule as RAGAdapter.
            and (memory_run_id is None or episode.run_id != memory_run_id)
        ][-3:]

    def upsert_episode(self, episode: Episode) -> None:
        for index, current in enumerate(self._episodes):
            if current.document_id == episode.document_id:
                self._episodes[index] = episode
                return
        self._episodes.append(episode)

    def upsert_episodes(self, episodes: list[Episode]) -> None:
        for episode in episodes:
            self.upsert_episode(episode)

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
        candidates = [
            episode
            for episode in reversed(self._episodes)
            if _matches_filters(episode, filters)
        ][: fetch_k or len(self._episodes)]
        raw_hits = [
            RetrievalHit(
                episode=episode,
                distance=0.0,
                similarity=1.0,
                rank=index,
                phase=phase,
            )
            for index, episode in enumerate(candidates, start=1)
        ]
        retained = [
            hit
            for hit in raw_hits
            if min_similarity is None or hit.similarity >= min_similarity
        ]
        hits = _deduplicate_hits(retained, phase)[: top_k or len(retained)]
        return RetrievalBatch(
            phase=phase,
            query=query,
            filters=filters,
            candidate_count=len(raw_hits),
            min_similarity=min_similarity,
            hits=hits,
        )

    def reset_ltm(self) -> None:
        self._episodes.clear()


def _matches_filters(episode: Episode, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = getattr(episode, key, None)
        if isinstance(expected, dict):
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$eq" in expected and actual != expected["$eq"]:
                return False
        elif actual != expected:
            return False
    return True
