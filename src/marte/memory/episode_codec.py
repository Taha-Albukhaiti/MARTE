from __future__ import annotations

import hashlib
from typing import Any

from marte.memory.text import _normalize_text
from marte.persistence.schemas import Episode


def _episode_to_text(episode: Episode) -> str:
    """Serialise an episode for embedding without leaking outcome labels."""
    parts = [
        f"strategy: {episode.strategy[:100]}",
        f"mutation: {episode.mutation[:300]}",
        f"response: {episode.response_summary[:200]}",
        f"tool_effect: {episode.tool_effect[:100]}",
    ]
    if episode.memory_kind == "blocked":
        parts.extend(
            [
                f"policy_stage: {episode.policy_stage[:20]}",
                f"policy_reason: {episode.policy_reason[:200]}",
            ]
        )
    return " | ".join(parts)


def _episode_to_metadata(episode: Episode) -> dict[str, Any]:
    """Extract flat metadata fields stored alongside the vector."""
    return {
        "document_id": episode.document_id,
        "run_id": episode.run_id,
        "step_t": episode.step_t,
        "scenario": episode.scenario,
        "guardrail_level": episode.guardrail_level,
        "strategy": episode.strategy[:100],
        "mutation": episode.mutation[:500],
        "response_summary": episode.response_summary[:300],
        "tool_effect": episode.tool_effect[:200],
        "success_label": episode.success_label,
        "memory_kind": episode.memory_kind,
        "policy_stage": episode.policy_stage[:20],
        "policy_reason": episode.policy_reason[:300],
    }


def _blocked_document_id(
    *,
    scenario: str,
    guardrail_level: str,
    policy_stage: str,
    policy_reason: str,
    mutation: str,
) -> str:
    """Return a stable ID so repeated concrete block patterns are upserted."""
    canonical = "|".join(
        _normalize_text(value)
        for value in (
            scenario,
            guardrail_level,
            policy_stage,
            policy_reason,
            mutation,
        )
    )
    return f"blocked-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
