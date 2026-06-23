from __future__ import annotations

import pytest
from pydantic import ValidationError

from marte.config.snapshot import ConfigSnapshot


def _cfg(**overrides: object) -> ConfigSnapshot:
    """Build a valid ConfigSnapshot with sensible defaults."""
    defaults: dict[str, object] = dict(
        experiment_id="exp-2026-01",
        seed=7,
        scenario="sql",
        guardrail_level="moderate",
        model_cfg={},
        tool_allowlist=["SQLTool"],
        max_budget=50,
        memory_mode="stm_only",
    )
    defaults.update(overrides)
    return ConfigSnapshot(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_all_fields_present() -> None:
    cfg = _cfg()
    assert cfg.experiment_id == "exp-2026-01"
    assert cfg.seed == 7
    assert cfg.scenario == "sql"
    assert cfg.guardrail_level == "moderate"
    assert cfg.memory_mode == "stm_only"
    assert cfg.max_budget == 50
    assert cfg.tool_allowlist == ["SQLTool"]


def test_frozen_rejects_mutation() -> None:
    cfg = _cfg()
    with pytest.raises((ValidationError, TypeError)):
        cfg.scenario = "file"  # type: ignore[misc]


def test_invalid_scenario_raises() -> None:
    with pytest.raises(ValidationError, match="scenario"):
        _cfg(scenario="network")


def test_invalid_guardrail_level_raises() -> None:
    with pytest.raises(ValidationError, match="guardrail_level"):
        _cfg(guardrail_level="strict")


def test_invalid_memory_mode_raises() -> None:
    with pytest.raises(ValidationError, match="memory_mode"):
        _cfg(memory_mode="full_memory")


def test_seed_propagates_to_model_cfg() -> None:
    cfg = _cfg(seed=42, model_cfg={})
    assert cfg.model_cfg["seed"] == 42


def test_seed_propagates_overwrites_existing_seed_in_model_cfg() -> None:
    cfg = _cfg(seed=13, model_cfg={"seed": 99, "model": "gpt-4"})
    # Seed from ConfigSnapshot.seed must win
    assert cfg.model_cfg["seed"] == 13
    assert cfg.model_cfg["model"] == "gpt-4"


def test_file_scenario_accepted() -> None:
    cfg = _cfg(scenario="file")
    assert cfg.scenario == "file"


def test_enterprise_guardrail_accepted() -> None:
    cfg = _cfg(guardrail_level="enterprise")
    assert cfg.guardrail_level == "enterprise"


def test_rag_ltm_memory_mode_accepted() -> None:
    cfg = _cfg(memory_mode="rag_ltm")
    assert cfg.memory_mode == "rag_ltm"
