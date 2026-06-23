from __future__ import annotations

from marte.memory.in_memory import InMemoryEpisodeStrategy
from marte.persistence.schemas import Episode


def _episode(document_id: str, run_id: int) -> Episode:
    return Episode(
        document_id=document_id,
        run_id=run_id,
        step_t=1,
        scenario="sql",
        guardrail_level="moderate",
        mutation="credential probe",
        response_summary="summary",
        tool_effect="SELECT:users",
        success_label="partial",
    )


def test_enrich_context_excludes_current_run_hits() -> None:
    strategy = InMemoryEpisodeStrategy()
    strategy.upsert_episode(_episode("current", run_id=7))
    strategy.upsert_episode(_episode("prior", run_id=6))

    hits = strategy.enrich_context({
        "scenario": "sql",
        "guardrail_level": "moderate",
        "run_id": 7,
    })

    ids = {episode.document_id for episode in hits}
    assert "prior" in ids
    assert "current" not in ids
