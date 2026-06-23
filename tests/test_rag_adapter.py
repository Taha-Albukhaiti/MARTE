from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
import pytest

from marte.memory.retrieval_results import _parse_query_results
from marte.memory.rag_adapter import RAGAdapter
from marte.memory.schemas import RetrievalBatch
from marte.memory.strategy import MemoryStrategy
from marte.persistence.schemas import Episode


class _StubEmbedder:
    """4-dim deterministic embedder; no model download required."""

    DIM = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hash(text) & 0xFFFF_FFFF
            vec = [(h >> (i * 4)) & 0xF for i in range(self.DIM)]
            results.append([float(v) / 15.0 for v in vec])
        return results


class _FailingEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        raise RuntimeError("embedding backend unavailable")


class _AxisEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text == "query-axis" or "aligned-example" in text:
                vectors.append([1.0, 0.0])
            elif "orthogonal-example" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 1.0])
        return vectors


def _make_adapter(top_k: int = 3) -> RAGAdapter:
    # Unique collection name per call prevents cross-test state leakage.
    return RAGAdapter(
        collection_name=f"test_{uuid.uuid4().hex}",
        top_k=top_k,
        embedding_backend=_StubEmbedder(),
        chroma_client=chromadb.EphemeralClient(),
    )


def _ep(
    doc_id: str,
    *,
    scenario: str = "sql",
    guardrail_level: str = "moderate",
    strategy: str = "schema_probe",
    label: str = "fail",
    mutation: str = "probe",
    step_t: int = 1,
    run_id: int = 1,
    memory_kind: str = "positive",
    policy_reason: str = "",
) -> Episode:
    return Episode(
        document_id=doc_id,
        run_id=run_id,
        step_t=step_t,
        scenario=scenario,
        guardrail_level=guardrail_level,
        strategy=strategy,
        mutation=mutation,
        response_summary="target response",
        tool_effect="SELECT:users",
        success_label=label,
        memory_kind=memory_kind,
        policy_stage="input" if memory_kind == "blocked" else "",
        policy_reason=policy_reason,
    )


def test_upsert_and_search_returns_stored_episode() -> None:
    adapter = _make_adapter(top_k=1)
    adapter.upsert(_ep("doc-1", strategy="credential_read", mutation="read credentials from database"))
    results = adapter.search("credential_read read credentials from database", top_k=1)
    assert len(results) == 1
    assert results[0].document_id == "doc-1"
    assert results[0].strategy == "credential_read"


def test_search_returns_up_to_top_k_results() -> None:
    adapter = _make_adapter(top_k=2)
    for i in range(5):
        adapter.upsert(_ep(f"doc-{i}", mutation=f"attack attempt {i}"))
    results = adapter.search("attack attempt", top_k=2)
    assert len(results) == 2


def test_search_on_empty_collection_returns_empty_list() -> None:
    adapter = _make_adapter()
    assert adapter.search("anything") == []


def test_upsert_idempotent_by_document_id() -> None:
    adapter = _make_adapter(top_k=10)
    ep = _ep("doc-x", mutation="same episode")
    adapter.upsert(ep)
    adapter.upsert(ep)
    results = adapter.search("same episode", top_k=10)
    ids = [r.document_id for r in results]
    assert ids.count("doc-x") == 1


def test_search_filters_by_scenario() -> None:
    adapter = _make_adapter(top_k=10)
    adapter.upsert(_ep("sql-1", scenario="sql", mutation="sql probe"))
    adapter.upsert(_ep("file-1", scenario="file", mutation="sql probe"))
    results = adapter.search("sql probe", filters={"scenario": "sql"})
    ids = {r.document_id for r in results}
    assert "sql-1" in ids
    assert "file-1" not in ids


def test_search_filters_by_guardrail_level() -> None:
    adapter = _make_adapter(top_k=10)
    adapter.upsert(_ep("mod-1", guardrail_level="moderate", mutation="bypass"))
    adapter.upsert(_ep("ent-1", guardrail_level="enterprise", mutation="bypass"))
    results = adapter.search("bypass", filters={"guardrail_level": "moderate"})
    ids = {r.document_id for r in results}
    assert "mod-1" in ids
    assert "ent-1" not in ids


def test_search_filters_by_scenario_and_guardrail_level() -> None:
    adapter = _make_adapter(top_k=10)
    adapter.upsert(_ep("match", scenario="sql", guardrail_level="moderate", mutation="target"))
    adapter.upsert(_ep("wrong-scenario", scenario="file", guardrail_level="moderate", mutation="target"))
    adapter.upsert(_ep("wrong-guard", scenario="sql", guardrail_level="enterprise", mutation="target"))
    results = adapter.search("target", filters={"scenario": "sql", "guardrail_level": "moderate"})
    ids = {r.document_id for r in results}
    assert "match" in ids
    assert "wrong-scenario" not in ids
    assert "wrong-guard" not in ids


def test_reset_ltm_clears_all_stored_episodes() -> None:
    adapter = _make_adapter(top_k=10)
    for i in range(3):
        adapter.upsert(_ep(f"ep-{i}", mutation=f"episode {i}"))
    adapter.reset_ltm()
    assert adapter.search("episode") == []


def test_reset_ltm_is_idempotent() -> None:
    adapter = _make_adapter()
    adapter.reset_ltm()
    adapter.reset_ltm()


def test_upsert_after_reset_stores_new_episodes() -> None:
    adapter = _make_adapter(top_k=10)
    adapter.upsert(_ep("old", mutation="old episode"))
    adapter.reset_ltm()
    adapter.upsert(_ep("new", mutation="new episode"))
    results = adapter.search("new episode")
    ids = {r.document_id for r in results}
    assert "new" in ids
    assert "old" not in ids


def test_episodes_from_multiple_runs_retrieved_in_later_run() -> None:
    adapter = _make_adapter(top_k=5)
    for run_id in range(1, 4):
        ep = _ep(f"run{run_id}-ep", mutation="credential_read probe", run_id=run_id)
        adapter.upsert(ep)
    results = adapter.search("credential_read probe", top_k=5)
    ids = {r.document_id for r in results}
    assert "run1-ep" in ids
    assert "run2-ep" in ids
    assert "run3-ep" in ids


def test_reset_ltm_between_conditions_removes_previous_condition_episodes() -> None:
    adapter = _make_adapter(top_k=10)
    adapter.upsert(_ep("cond1-ep", mutation="condition 1 probe"))
    adapter.reset_ltm()
    adapter.upsert(_ep("cond2-ep", mutation="condition 2 probe"))
    results = adapter.search("probe", top_k=10)
    ids = {r.document_id for r in results}
    assert "cond1-ep" not in ids
    assert "cond2-ep" in ids


def test_rag_adapter_satisfies_memory_strategy_protocol() -> None:
    adapter = _make_adapter()
    assert isinstance(adapter, MemoryStrategy)


def test_enrich_context_returns_episode_list() -> None:
    adapter = _make_adapter(top_k=3)
    adapter.upsert(_ep("ctx-ep", scenario="sql", guardrail_level="moderate", mutation="find credentials"))
    run_state = {
        "scenario": "sql",
        "guardrail_level": "moderate",
        "goal": "read protected credentials",
        "last_prompt": "find credentials",
    }
    results = adapter.enrich_context(run_state)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(ep, Episode) for ep in results)


def test_enrich_context_excludes_current_run_id() -> None:
    adapter = _make_adapter(top_k=5)
    adapter.upsert(_ep("current-run", run_id=7, mutation="credential probe"))
    adapter.upsert(_ep("prior-run", run_id=6, mutation="credential probe"))
    run_state = {
        "run_id": 7,
        "scenario": "sql",
        "guardrail_level": "moderate",
        "goal": "read protected credentials",
        "last_prompt": "credential probe",
    }

    results = adapter.enrich_context(run_state)
    ids = {episode.document_id for episode in results}

    assert "prior-run" in ids
    assert "current-run" not in ids


def test_health_check_passes_with_stub_embedder() -> None:
    adapter = _make_adapter(top_k=1)
    adapter.health_check()


def test_health_check_raises_on_embedding_failure() -> None:
    adapter = RAGAdapter(
        collection_name=f"test_fail_{uuid.uuid4().hex}",
        top_k=1,
        embedding_backend=_FailingEmbedder(),
        chroma_client=chromadb.EphemeralClient(),
    )

    with pytest.raises(RuntimeError, match="health check failed"):
        adapter.health_check()


def test_persistent_client_writes_to_disk_when_no_client_injected(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma_store"
    adapter = RAGAdapter(
        collection_name=f"persist_{uuid.uuid4().hex}",
        top_k=1,
        embedding_backend=_StubEmbedder(),
        persist_path=persist_dir,
    )

    adapter.upsert(_ep("doc-persist", mutation="disk persistence check"))
    results = adapter.search("disk persistence check", top_k=1)

    assert adapter.persist_path == persist_dir
    assert results and results[0].document_id == "doc-persist"
    # PersistentClient creates files in the target directory.
    assert persist_dir.exists()
    assert any(persist_dir.iterdir())


def test_parse_query_results_converts_cosine_distance_to_similarity() -> None:
    results = {
        "metadatas": [[_ep("doc").model_dump(exclude={"embedding"})]],
        "distances": [[0.25]],
    }

    hits = _parse_query_results(results, phase="planner_positive")

    assert hits[0].distance == pytest.approx(0.25)
    assert hits[0].similarity == pytest.approx(0.75)
    assert hits[0].rank == 1
    assert hits[0].phase == "planner_positive"


def test_retrieve_returns_auditable_empty_batch() -> None:
    adapter = _make_adapter()

    batch = adapter.retrieve(
        query="anything",
        phase="planner_blocked",
        filters={"memory_kind": "blocked"},
    )

    assert isinstance(batch, RetrievalBatch)
    assert batch.phase == "planner_blocked"
    assert batch.candidate_count == 0
    assert batch.hits == []


def test_retrieve_applies_positive_strategy_deduplication() -> None:
    adapter = _make_adapter(top_k=3)
    adapter.upsert(
        _ep(
            "same-a",
            strategy="credential_read",
            mutation="same relevant prompt",
            label="success",
        )
    )
    adapter.upsert(
        _ep(
            "same-b",
            strategy="credential_read",
            mutation="same relevant prompt",
            label="partial",
        )
    )
    adapter.upsert(
        _ep(
            "other",
            strategy="table_read",
            mutation="same relevant prompt",
            label="success",
        )
    )

    batch = adapter.retrieve(
        query="same relevant prompt",
        phase="planner_positive",
        filters={"memory_kind": "positive"},
        top_k=3,
        fetch_k=12,
        min_similarity=None,
    )

    assert batch.candidate_count == 3
    assert len(batch.hits) == 2
    assert {hit.episode.strategy for hit in batch.hits} == {
        "credential_read",
        "table_read",
    }


def test_retrieve_applies_similarity_threshold() -> None:
    adapter = RAGAdapter(
        collection_name=f"threshold_{uuid.uuid4().hex}",
        top_k=3,
        embedding_backend=_AxisEmbedder(),
        chroma_client=chromadb.EphemeralClient(),
    )
    adapter.upsert(
        _ep(
            "aligned",
            strategy="credential_read",
            mutation="aligned-example",
        )
    )
    adapter.upsert(
        _ep(
            "orthogonal",
            strategy="table_read",
            mutation="orthogonal-example",
        )
    )

    batch = adapter.retrieve(
        query="query-axis",
        phase="planner_positive",
        filters={"memory_kind": "positive"},
        min_similarity=0.5,
    )

    assert batch.candidate_count == 2
    assert [hit.episode.document_id for hit in batch.hits] == ["aligned"]


def test_retrieve_filters_exact_mutator_strategy() -> None:
    adapter = _make_adapter(top_k=3)
    adapter.upsert(
        _ep(
            "credential",
            strategy="credential_read",
            label="success",
            mutation="read protected rows",
        )
    )
    adapter.upsert(
        _ep(
            "credential-second",
            strategy="credential_read",
            label="partial",
            mutation="read protected rows with a different framing",
        )
    )
    adapter.upsert(
        _ep(
            "table",
            strategy="table_read",
            label="success",
            mutation="read protected rows",
        )
    )

    batch = adapter.retrieve(
        query="read protected rows",
        phase="mutator_positive",
        filters={
            "memory_kind": "positive",
            "strategy": "credential_read",
        },
    )

    ids = {hit.episode.document_id for hit in batch.hits}
    assert ids == {"credential", "credential-second"}
