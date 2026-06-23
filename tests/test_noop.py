from __future__ import annotations

from marte.memory.noop import NoOpStrategy
from marte.persistence.schemas import Episode


def _ep(step_t: int = 1) -> Episode:
    return Episode(
        document_id=f"ep-{step_t}",
        run_id=1,
        step_t=step_t,
        scenario="sql",
        guardrail_level="moderate",
        mutation="probe",
        response_summary="response",
        tool_effect="",
        success_label="fail",
    )


def test_enrich_context_returns_empty_list() -> None:
    strategy = NoOpStrategy()
    result = strategy.enrich_context({"scenario": "sql", "step_index": 1})
    assert result == []


def test_enrich_context_always_empty_regardless_of_run_state() -> None:
    strategy = NoOpStrategy()
    assert strategy.enrich_context({}) == []
    assert strategy.enrich_context({"goal": "attack", "last_prompt": "try"}) == []


def test_upsert_episode_has_no_effect_and_no_error() -> None:
    strategy = NoOpStrategy()
    strategy.upsert_episode(_ep(1))
    strategy.upsert_episode(_ep(2))
    # No state to check — just verify it does not raise.


def test_reset_ltm_is_a_noop_and_does_not_raise() -> None:
    strategy = NoOpStrategy()
    strategy.reset_ltm()
    strategy.reset_ltm()  # idempotent


def test_noop_satisfies_memory_strategy_protocol() -> None:
    from marte.memory.strategy import MemoryStrategy
    strategy = NoOpStrategy()
    assert isinstance(strategy, MemoryStrategy)
