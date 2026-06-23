from __future__ import annotations

import uuid
from typing import Any

import chromadb

from marte.memory.gateway import MemoryGateway
from marte.memory.noop import NoOpStrategy
from marte.memory.rag_adapter import RAGAdapter


class _StubEmbedder:
    DIM = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hash(text) & 0xFFFF_FFFF
            vec = [(h >> (i * 4)) & 0xF for i in range(self.DIM)]
            results.append([float(v) / 15.0 for v in vec])
        return results


def _rag_adapter() -> RAGAdapter:
    return RAGAdapter(
        collection_name=f"gw_test_{uuid.uuid4().hex}",
        top_k=3,
        embedding_backend=_StubEmbedder(),
        chroma_client=chromadb.EphemeralClient(),
    )


def _run_state(step: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "scenario": "sql",
        "guardrail_level": "moderate",
        "step_index": step,
        "run_id": 1,
        "goal": "read protected credentials",
        **extra,
    }


def _judge_episode(
    prompt: str = "test prompt",
    verdict: str = "fail",
    oracle_verdict: bool = False,
    strategy: str = "schema_probe",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "prompt": prompt,
        "response_text": "agent response",
        "tool_trace": [{"operation": "SELECT", "target": "users"}],
        "oracle_verdict": oracle_verdict,
        "judge_verdict": verdict,
        **extra,
    }


# ---------------------------------------------------------------------------
# enrich_context
# ---------------------------------------------------------------------------

def test_enrich_context_always_includes_stm_key() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    ctx = gw.enrich_context(_run_state(step=1))
    assert "stm_recent" in ctx
    assert "ltm_hits" in ctx


def test_enrich_context_stm_only_ltm_hits_are_empty() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.upsert_episode(_judge_episode())
    ctx = gw.enrich_context(_run_state(step=2))
    assert ctx["ltm_hits"] == []


def test_enrich_context_stm_recent_contains_upserted_episodes() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))  # seed run_context
    gw.upsert_episode(_judge_episode("first prompt", "partial"))
    ctx = gw.enrich_context(_run_state(step=2))
    assert len(ctx["stm_recent"]) == 1
    assert ctx["stm_recent"][0]["success_label"] == "partial"


def test_enrich_context_step_index_propagated() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    ctx = gw.enrich_context(_run_state(step=7))
    assert ctx["step_index"] == 7


def test_enrich_context_exposes_observations_only_through_stm() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(
        _run_state(
            step=2,
            guardrail_level="enterprise",
            available_tools=["sql"],
        )
    )
    gw.upsert_episode(
        _judge_episode(
            prompt="[strategy=credential_read] blocked prompt",
            verdict="fail",
            strategy="credential_read",
            input_decision="deny",
            input_reason="[DENIED] protected table",
            policy_decisions=[],
            tool_trace=[],
        )
    )
    ctx = gw.enrich_context(
        _run_state(
            step=3,
            guardrail_level="enterprise",
            available_tools=["sql"],
        )
    )

    assert ctx["guardrail_level"] == "enterprise"
    assert ctx["scenario"] == "sql"
    assert ctx["available_tools"] == ["sql"]
    assert "last_policy_decisions" not in ctx
    assert "last_tool_trace" not in ctx
    assert ctx["stm_recent"][0]["policy_stage"] == "input"
    assert ctx["stm_recent"][0]["policy_reason"] == "[DENIED] protected table"


def test_enrich_context_rag_ltm_excludes_current_run_hits() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)

    rs = _run_state(step=1, last_prompt="find credentials")
    gw.enrich_context(rs)  # seed run_context
    gw.upsert_episode(_judge_episode("find credentials in db", "success", oracle_verdict=True))

    ctx = gw.enrich_context({**rs, "step_index": 2, "last_prompt": "find credentials"})
    assert ctx["ltm_hits"] == []


def test_enrich_context_rag_ltm_hits_populated_from_prior_run() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)

    rs = _run_state(step=1, run_id=1, last_prompt="find credentials")
    gw.enrich_context(rs)  # seed run_context
    gw.upsert_episode(_judge_episode("find credentials in db", "success", oracle_verdict=True))

    ctx = gw.enrich_context({**rs, "run_id": 2, "step_index": 1, "last_prompt": "find credentials"})
    assert isinstance(ctx["ltm_hits"], list)
    assert len(ctx["ltm_hits"]) >= 1


# ---------------------------------------------------------------------------
# upsert_episode
# ---------------------------------------------------------------------------

def test_upsert_episode_adds_to_stm() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    assert len(gw._stm) == 0
    gw.upsert_episode(_judge_episode())
    assert len(gw._stm) == 1


def test_upsert_episode_maps_judge_verdict_to_success_label() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=3))
    gw.upsert_episode(_judge_episode(verdict="success"))
    ctx = gw.enrich_context(_run_state(step=4))
    assert ctx["stm_recent"][0]["success_label"] == "success"


def test_upsert_episode_maps_prompt_to_mutation() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(_judge_episode("my attacker prompt", "fail"))
    ctx = gw.enrich_context(_run_state(step=2))
    assert "my attacker prompt" in ctx["stm_recent"][0]["mutation"]


def test_upsert_episode_maps_strategy_to_episode() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(_judge_episode("my attacker prompt", "partial", strategy="credential_read"))
    ctx = gw.enrich_context(_run_state(step=2))
    assert ctx["stm_recent"][0]["strategy"] == "credential_read"


def test_upsert_episode_respects_stm_capacity() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=3)
    gw.enrich_context(_run_state(step=1))
    for i in range(5):
        gw.upsert_episode(_judge_episode(f"prompt {i}"))
    assert len(gw._stm) == 3


def test_upsert_episode_delegates_to_rag_strategy() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    rs = _run_state(step=1)
    gw.enrich_context(rs)
    gw.upsert_episode(_judge_episode("rag probe", "success", oracle_verdict=True))

    # Check that RAG now returns this episode on search
    results = rag.search("rag probe", top_k=5)
    assert any("rag probe" in ep.mutation for ep in results)


def test_upsert_episode_delegates_partial_to_rag_strategy() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    rs = _run_state(step=1)
    gw.enrich_context(rs)
    gw.upsert_episode(_judge_episode("partial probe", "partial", oracle_verdict=False))

    results = rag.search("partial probe", top_k=5)
    assert any("partial probe" in ep.mutation for ep in results)


def test_upsert_episode_does_not_delegate_fail_to_rag_strategy() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    rs = _run_state(step=1)
    gw.enrich_context(rs)
    before = rag._collection.count()
    gw.upsert_episode(_judge_episode("failed probe", "fail", oracle_verdict=False))

    assert rag._collection.count() == before


def test_upsert_episode_writes_directly_to_ltm() -> None:
    """Episodes are written to LTM immediately on upsert (no transaction needed)."""
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    gw.enrich_context(_run_state(step=1, run_id=1))

    gw.upsert_episode(
        _judge_episode("direct write probe", "success", oracle_verdict=True)
    )

    assert any(
        "direct write probe" in episode.mutation
        for episode in rag.search("direct write probe", top_k=5)
    )


def test_same_episode_upserted_twice_deduplicates() -> None:
    """Upserting the same episode twice results in exactly one LTM entry."""
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    run_state = _run_state(step=1, run_id=7, memory_run_id=7)
    episode = _judge_episode("stable dedup probe", "success", oracle_verdict=True)

    gw.enrich_context(run_state)
    gw.upsert_episode(episode)
    gw.upsert_episode(episode)

    assert rag._collection.count() == 1


def test_input_policy_block_is_stored_in_blocked_channel_with_stable_id() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    episode = _judge_episode(
        "blocked direct prompt",
        input_decision="deny",
        input_reason="direct tool JSON coercion",
    )

    gw.upsert_episode(episode)
    gw.upsert_episode(episode)

    results = rag.search(
        "direct tool JSON coercion",
        top_k=10,
        filters={"memory_kind": "blocked"},
    )
    assert len(results) == 1
    assert results[0].document_id.startswith("blocked-")
    assert results[0].policy_stage == "input"
    assert results[0].policy_reason == "direct tool JSON coercion"


def test_tool_policy_block_is_stored_but_output_only_block_is_not() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(
        _judge_episode(
            "tool blocked prompt",
            tool_trace=[
                {
                    "operation": "DENIED",
                    "target": "credentials",
                    "error": "enterprise policy: protected table",
                }
            ],
            policy_decisions=["deny"],
        )
    )
    after_tool_block = rag._collection.count()
    gw.upsert_episode(
        _judge_episode(
            "output blocked prompt",
            output_decision="deny",
            output_reason="response leaks password_hash",
        )
    )

    results = rag.search(
        "protected table",
        top_k=10,
        filters={"memory_kind": "blocked"},
    )
    assert after_tool_block == 1
    assert rag._collection.count() == after_tool_block
    assert results[0].policy_stage == "tool"


def test_infrastructure_denial_is_not_stored_as_blocked_memory() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    gw.enrich_context(_run_state(step=1))

    gw.upsert_episode(
        _judge_episode(
            "executor failure",
            input_decision="deny",
            input_reason="target execution failed: connection reset",
        )
    )

    assert rag._collection.count() == 0


def test_oracle_breach_has_priority_over_block_metadata() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(
        _judge_episode(
            "oracle-confirmed prompt",
            verdict="fail",
            oracle_verdict=True,
            input_decision="deny",
            input_reason="synthetic conflict",
        )
    )

    results = rag.search("oracle-confirmed", top_k=5)
    assert len(results) == 1
    assert results[0].memory_kind == "positive"
    assert results[0].success_label == "success"


def test_planner_and_mutator_retrieval_use_separate_filters() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    for run_id, strategy in ((1, "credential_read"), (2, "table_read")):
        gw.enrich_context(_run_state(step=1, run_id=run_id))
        gw.upsert_episode(
            _judge_episode(
                f"{strategy} prior prompt",
                "success",
                oracle_verdict=True,
                strategy=strategy,
            )
        )

    current = _run_state(step=1, run_id=3)
    planner_context = gw.enrich_context(current)
    mutator_context = gw.enrich_context_for_mutator(
        "credential_read",
        "read credentials",
        current,
    )

    assert all(
        episode["memory_kind"] == "positive"
        for episode in planner_context["positive_ltm_hits"]
    )
    assert all(
        episode["strategy"] == "credential_read"
        for episode in mutator_context["mutator_ltm_hits"]
    )
    assert (
        mutator_context["mutator_retrieval"]["filters"]["strategy"]
        == "credential_read"
    )


# ---------------------------------------------------------------------------
# reset_stm
# ---------------------------------------------------------------------------

def test_reset_stm_clears_stm_buffer() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(_judge_episode())
    gw.upsert_episode(_judge_episode())
    gw.reset_stm()
    assert len(gw._stm) == 0


def test_reset_stm_does_not_clear_ltm() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    rs = _run_state(step=1)
    gw.enrich_context(rs)
    gw.upsert_episode(_judge_episode("keep in ltm", "success", oracle_verdict=True))

    gw.reset_stm()  # STM cleared, LTM intact

    results = rag.search("keep in ltm", top_k=5)
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# reset_ltm
# ---------------------------------------------------------------------------

def test_reset_ltm_clears_rag_collection() -> None:
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=5)
    rs = _run_state(step=1)
    gw.enrich_context(rs)
    gw.upsert_episode(_judge_episode("condition 1 data", "success", oracle_verdict=True))

    gw.reset_ltm()

    results = rag.search("condition 1", top_k=5)
    assert results == []


def test_reset_ltm_does_not_clear_stm() -> None:
    gw = MemoryGateway(strategy=NoOpStrategy(), stm_capacity=5)
    gw.enrich_context(_run_state(step=1))
    gw.upsert_episode(_judge_episode())
    initial_len = len(gw._stm)

    gw.reset_ltm()

    assert len(gw._stm) == initial_len


# ---------------------------------------------------------------------------
# Isolation semantics
# ---------------------------------------------------------------------------

def test_stm_reset_between_runs_ltm_accumulates() -> None:
    """Simulate 3 runs: STM cleared each run, LTM grows across runs."""
    rag = _rag_adapter()
    gw = MemoryGateway(strategy=rag, stm_capacity=3)

    for run in range(1, 4):
        gw.reset_stm()  # RunOrchestrator resets STM at run start
        rs = _run_state(step=1, run_id=run)
        gw.enrich_context(rs)
        gw.upsert_episode(_judge_episode(f"run {run} probe", "success", oracle_verdict=True))

    # After 3 runs, LTM should have 3 episodes; STM has only the last run's
    assert len(gw._stm) == 1
    results = rag.search("run probe", top_k=10)
    assert len(results) == 3


def test_default_strategy_is_noop() -> None:
    gw = MemoryGateway()
    ctx = gw.enrich_context(_run_state())
    assert ctx["ltm_hits"] == []
