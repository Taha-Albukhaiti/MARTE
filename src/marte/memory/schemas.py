from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from marte.persistence.schemas import Episode

RetrievalPhase = Literal["planner_positive", "planner_blocked", "mutator_positive"]


class RetrievalHit(BaseModel):
    """One auditable vector-retrieval result."""

    episode: Episode
    distance: float
    similarity: float
    rank: int = Field(ge=1)
    phase: RetrievalPhase


class RetrievalBatch(BaseModel):
    """Complete diagnostic record for one retrieval phase."""

    phase: RetrievalPhase
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    candidate_count: int = Field(default=0, ge=0)
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    hits: list[RetrievalHit] = Field(default_factory=list)

