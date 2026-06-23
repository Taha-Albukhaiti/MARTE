from __future__ import annotations

from typing import Any

from marte.memory.schemas import RetrievalHit, RetrievalPhase
from marte.memory.text import _normalize_text
from marte.persistence.schemas import Episode


def _parse_query_results(
    results: dict[str, Any],
    phase: RetrievalPhase,
) -> list[RetrievalHit]:
    """Convert raw ChromaDB query results to ranked retrieval hits."""
    hits: list[RetrievalHit] = []
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for index, meta in enumerate(metadatas):
        if not meta:
            continue
        try:
            distance = float(distances[index]) if index < len(distances) else 1.0
            similarity = 1.0 - distance
            episode = Episode(
                document_id=str(meta.get("document_id", "")),
                run_id=int(meta.get("run_id", 0)),
                step_t=int(meta.get("step_t", 0)),
                scenario=str(meta.get("scenario", "")),
                guardrail_level=str(meta.get("guardrail_level", "")),
                strategy=str(meta.get("strategy", "")),
                mutation=str(meta.get("mutation", "")),
                response_summary=str(meta.get("response_summary", "")),
                tool_effect=str(meta.get("tool_effect", "")),
                success_label=str(meta.get("success_label", "fail")),
                memory_kind=str(meta.get("memory_kind", "positive")),
                policy_stage=str(meta.get("policy_stage", "")),
                policy_reason=str(meta.get("policy_reason", "")),
            )
            hits.append(
                RetrievalHit(
                    episode=episode,
                    distance=distance,
                    similarity=similarity,
                    rank=index + 1,
                    phase=phase,
                )
            )
        except (ValueError, TypeError, KeyError):
            continue
    return hits


def _deduplicate_hits(
    hits: list[RetrievalHit],
    phase: RetrievalPhase,
) -> list[RetrievalHit]:
    """Keep the highest-ranked deterministic representative per duplicate key."""
    deduplicated: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        episode = hit.episode
        if phase == "planner_blocked":
            key = "|".join(
                (
                    _normalize_text(episode.mutation),
                    _normalize_text(episode.policy_reason),
                )
            )
        elif phase == "planner_positive":
            key = _normalize_text(episode.strategy) or episode.document_id
        else:
            key = episode.document_id
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(hit)
    return deduplicated
