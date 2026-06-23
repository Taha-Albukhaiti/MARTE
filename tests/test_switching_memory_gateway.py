from __future__ import annotations

from typing import Any, Mapping

import pytest

from marte.memory.switching_gateway import SwitchingMemoryGateway


class SpyStrategy:
    def __init__(self) -> None:
        self.upsert_calls = 0
        self.reset_calls = 0

    def enrich_context(self, run_state: Mapping[str, Any]):  # noqa: ARG002
        return []

    def upsert_episode(self, episode):  # noqa: ANN001, ARG002
        self.upsert_calls += 1

    def reset_ltm(self) -> None:
        self.reset_calls += 1


class FailingHealthStrategy(SpyStrategy):
    def health_check(self) -> None:
        raise RuntimeError("embedder unavailable")


class MissingThresholdStrategy(SpyStrategy):
    def validate_experiment_config(self) -> None:
        raise ValueError("RAG-LTM in API mode requires calibrated thresholds")


def _episode_dict(oracle_verdict: bool = False, judge_verdict: str | None = None) -> dict[str, Any]:
    return {
        "prompt": "p",
        "response_text": "r",
        "judge_verdict": judge_verdict or ("success" if oracle_verdict else "fail"),
        "oracle_verdict": oracle_verdict,
        "tool_trace": [],
    }


def test_defaults_to_stm_mode() -> None:
    gateway = SwitchingMemoryGateway(rag_strategy=SpyStrategy())

    context = gateway.enrich_context({"step_index": 1})

    assert gateway.active_mode == "stm_only"
    assert context["memory_mode"] == "stm_only"


def test_set_mode_switches_to_rag_ltm() -> None:
    gateway = SwitchingMemoryGateway(rag_strategy=SpyStrategy())

    gateway.set_mode("rag_ltm")
    context = gateway.enrich_context({"step_index": 1})

    assert gateway.active_mode == "rag_ltm"
    assert context["memory_mode"] == "rag_ltm"


def test_upsert_routes_to_active_gateway_only() -> None:
    strategy = SpyStrategy()
    gateway = SwitchingMemoryGateway(rag_strategy=strategy)

    gateway.upsert_episode(_episode_dict())
    assert strategy.upsert_calls == 0

    gateway.set_mode("rag_ltm")
    gateway.upsert_episode(_episode_dict(oracle_verdict=True))
    assert strategy.upsert_calls == 1

    gateway.upsert_episode(_episode_dict(judge_verdict="partial"))
    assert strategy.upsert_calls == 2


def test_reset_ltm_delegates_to_rag_strategy() -> None:
    strategy = SpyStrategy()
    gateway = SwitchingMemoryGateway(rag_strategy=strategy)

    gateway.reset_ltm()

    assert strategy.reset_calls == 1


def test_set_mode_rag_ltm_fails_fast_when_health_check_fails() -> None:
    gateway = SwitchingMemoryGateway(rag_strategy=FailingHealthStrategy())

    with pytest.raises(RuntimeError, match="health check failed"):
        gateway.set_mode("rag_ltm")


def test_stm_mode_does_not_validate_unused_rag_configuration() -> None:
    gateway = SwitchingMemoryGateway(rag_strategy=MissingThresholdStrategy())

    gateway.set_mode("stm_only")

    assert gateway.active_mode == "stm_only"


def test_rag_mode_validates_thresholds_on_activation() -> None:
    gateway = SwitchingMemoryGateway(rag_strategy=MissingThresholdStrategy())

    with pytest.raises(RuntimeError, match="requires calibrated thresholds"):
        gateway.set_mode("rag_ltm")
