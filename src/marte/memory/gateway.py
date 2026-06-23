from __future__ import annotations

import uuid
from typing import Any, Mapping

from marte.memory.episode_codec import _blocked_document_id
from marte.memory.episode_policy import (
    _classify_ltm_episode,
    _summarize_tool_effect,
)
from marte.memory.retrieval_query import (
    _build_mutator_retrieval_query,
    _build_retrieval_filters,
    _build_retrieval_query,
)
from marte.memory.retrieval_results import _deduplicate_hits
from marte.memory.noop import NoOpStrategy
from marte.memory.schemas import RetrievalBatch, RetrievalHit, RetrievalPhase
from marte.memory.stm import STMRingBuffer
from marte.memory.strategy import MemoryStrategy
from marte.persistence.schemas import Episode

_DEFAULT_STM_CAPACITY = 5


class MemoryGateway:
    """Single access point for STM and the injected LTM strategy."""

    def __init__(
        self,
        strategy: MemoryStrategy | None = None,
        stm_capacity: int = _DEFAULT_STM_CAPACITY,
    ) -> None:
        self._strategy: MemoryStrategy = strategy or NoOpStrategy()
        self._stm = STMRingBuffer(capacity=stm_capacity)
        self._run_context: dict[str, Any] = {}

    def enrich_context(self, run_state: Mapping[str, Any]) -> dict[str, Any]:
        """Build the pre-planning context with separated LTM channels."""
        self._run_context = dict(run_state)
        query = _build_retrieval_query(run_state)
        positive = self._retrieve_batch(
            query=query,
            phase="planner_positive",
            filters=_build_retrieval_filters(run_state, memory_kind="positive"),
        )
        blocked = self._retrieve_batch(
            query=query,
            phase="planner_blocked",
            filters=_build_retrieval_filters(run_state, memory_kind="blocked"),
        )

        context = self._base_context(run_state)
        context.update(
            {
                "stm_recent": [
                    episode.model_dump(exclude={"embedding"})
                    for episode in self._stm.get_context()
                ],
                "positive_ltm_hits": _episodes_from_batch(positive),
                "blocked_patterns": _episodes_from_batch(blocked),
                # Compatibility alias for non-model callers.
                "ltm_hits": _episodes_from_batch(positive),
                "retrieval_trace": {
                    "planner_positive": positive.model_dump(),
                    "planner_blocked": blocked.model_dump(),
                },
            }
        )
        return context

    def enrich_context_for_mutator(
        self,
        plan_strategy: str,
        plan_objective: str,
        run_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Retrieve positive examples matching the selected strategy exactly."""
        self._run_context = dict(run_state)
        query = _build_mutator_retrieval_query(
            plan_strategy,
            plan_objective,
            run_state,
        )
        batch = self._retrieve_batch(
            query=query,
            phase="mutator_positive",
            filters=_build_retrieval_filters(
                run_state,
                memory_kind="positive",
                strategy=plan_strategy,
            ),
        )
        return {
            "mutator_ltm_hits": _episodes_from_batch(batch),
            "mutator_retrieval": batch.model_dump(),
        }

    def upsert_episode(self, episode_dict: dict[str, Any]) -> None:
        """Append every step to STM and selected evidence to the proper LTM."""
        classification = _classify_ltm_episode(episode_dict)
        episode = self._build_episode(episode_dict, classification)
        self._stm.append(episode)
        if classification is not None:
            self._strategy.upsert_episode(episode)

    def reset_stm(self) -> None:
        self._stm.reset()

    def reset_ltm(self) -> None:
        self._strategy.reset_ltm()

    def _base_context(self, run_state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "step_index": run_state.get("step_index", 0),
            "scenario": run_state.get("scenario", ""),
            "goal": run_state.get("goal", ""),
            "guardrail_level": run_state.get("guardrail_level", ""),
            "available_tools": run_state.get("available_tools", []),
            "execution_mode": run_state.get("execution_mode", "local"),
            "memory_run_id": run_state.get(
                "memory_run_id",
                run_state.get("run_id", 0),
            ),
            # Internal control metadata. Planner/Mutator payload builders
            # deliberately omit this field to keep treatment assignment blind.
            "memory_mode": (
                "stm_only"
                if isinstance(self._strategy, NoOpStrategy)
                else "rag_ltm"
            ),
        }

    def _retrieve_batch(
        self,
        *,
        query: str,
        phase: RetrievalPhase,
        filters: dict[str, Any],
    ) -> RetrievalBatch:
        retrieve = getattr(self._strategy, "retrieve", None)
        if callable(retrieve):
            return retrieve(query=query, phase=phase, filters=filters)

        # Compatibility for simple test/local strategies implementing the old
        # contract. Production RAG and NoOp strategies use retrieve().
        if phase == "planner_blocked":
            return RetrievalBatch(phase=phase, query=query, filters=filters)
        episodes = self._strategy.enrich_context(self._run_context)
        filtered = [
            episode
            for episode in episodes
            if episode.memory_kind == filters.get("memory_kind", episode.memory_kind)
            and (
                not filters.get("strategy")
                or episode.strategy == filters["strategy"]
            )
        ]
        raw_hits = [
            RetrievalHit(
                episode=episode,
                distance=0.0,
                similarity=1.0,
                rank=index,
                phase=phase,
            )
            for index, episode in enumerate(filtered, start=1)
        ]
        limit = 3
        return RetrievalBatch(
            phase=phase,
            query=query,
            filters=filters,
            candidate_count=len(raw_hits),
            hits=_deduplicate_hits(raw_hits, phase)[:limit],
        )

    def _build_episode(
        self,
        episode_dict: dict[str, Any],
        classification: tuple[str, str, str] | None,
    ) -> Episode:
        ctx = self._run_context
        tool_trace = episode_dict.get("tool_trace", [])
        memory_kind, policy_stage, policy_reason = classification or (
            "positive",
            "",
            "",
        )
        mutation = str(episode_dict.get("prompt", ""))[:500]
        scenario = str(ctx.get("scenario", "sql"))
        guardrail_level = str(ctx.get("guardrail_level", "moderate"))
        document_id = (
            _blocked_document_id(
                scenario=scenario,
                guardrail_level=guardrail_level,
                policy_stage=policy_stage,
                policy_reason=policy_reason,
                mutation=mutation,
            )
            if memory_kind == "blocked"
            else str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"marte:{scenario}:{guardrail_level}:"
                        f"{ctx.get('memory_run_id', ctx.get('run_id', 0))}:"
                        f"{ctx.get('step_index', 0)}:positive"
                    ),
                )
            )
        )
        return Episode(
            document_id=document_id,
            run_id=int(ctx.get("memory_run_id", ctx.get("run_id", 0))),
            step_t=int(ctx.get("step_index", 0)),
            scenario=scenario,
            guardrail_level=guardrail_level,
            strategy=str(episode_dict.get("strategy", ""))[:100],
            mutation=mutation,
            response_summary=str(
                episode_dict.get("memory_summary")
                or episode_dict.get("response_text", "")
            )[:1000],
            tool_effect=_summarize_tool_effect(tool_trace),
            success_label=(
                "success"
                if bool(episode_dict.get("oracle_verdict"))
                else str(episode_dict.get("judge_verdict", "fail"))
            ),
            memory_kind=memory_kind,
            policy_stage=policy_stage,
            policy_reason=policy_reason[:300],
        )


def _episodes_from_batch(batch: RetrievalBatch) -> list[dict[str, Any]]:
    return [
        hit.episode.model_dump(exclude={"embedding"})
        for hit in batch.hits
    ]
