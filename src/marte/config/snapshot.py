from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_VALID_SCENARIOS: frozenset[str] = frozenset({"sql", "file"})
_VALID_GUARDRAIL_LEVELS: frozenset[str] = frozenset({"moderate", "enterprise"})
_VALID_MEMORY_MODES: frozenset[str] = frozenset({"stm_only", "rag_ltm"})


class ConfigSnapshot(BaseModel):
    """Frozen configuration for a single experiment run.

    Immutable after construction — no field may change during a run.
    The ComparativeRunner creates one ConfigSnapshot per run with the
    appropriate seed and condition parameters.

    Seed propagation: the ``seed`` value is automatically written into
    ``model_cfg["seed"]`` so that LLM API calls receive it without
    requiring callers to set it manually.

    Args:
        experiment_id: Unique identifier for the enclosing experiment batch.
        seed: RNG seed for this run (1-30 in the standard experiment).
        scenario: Attack scenario — one of ``{"sql", "file"}``.
        guardrail_level: Target guardrail strength — one of
            ``{"moderate", "enterprise"}``.
        model_cfg: LLM configuration dict (model name, temperature, …).
            ``model_cfg["seed"]`` is set automatically from ``seed``.
        tool_allowlist: Names of tools available to the target agent.
        max_budget: Maximum number of attack steps per run.
        memory_mode: Memory strategy — one of ``{"stm_only", "rag_ltm"}``.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    seed: int
    scenario: str
    guardrail_level: str
    model_cfg: dict[str, Any]
    tool_allowlist: list[str]
    max_budget: int
    memory_mode: str

    @model_validator(mode="before")
    @classmethod
    def _propagate_seed(cls, data: Any) -> Any:
        """Copy ``seed`` into ``model_cfg["seed"]`` before the model is frozen."""
        if isinstance(data, dict):
            seed = data.get("seed")
            if seed is not None:
                model_cfg: dict[str, Any] = dict(data.get("model_cfg") or {})
                model_cfg["seed"] = seed
                return {**data, "model_cfg": model_cfg}
        return data

    @field_validator("scenario")
    @classmethod
    def _validate_scenario(cls, v: str) -> str:
        if v not in _VALID_SCENARIOS:
            raise ValueError(
                f"scenario must be one of {sorted(_VALID_SCENARIOS)}, got {v!r}"
            )
        return v

    @field_validator("guardrail_level")
    @classmethod
    def _validate_guardrail_level(cls, v: str) -> str:
        if v not in _VALID_GUARDRAIL_LEVELS:
            raise ValueError(
                f"guardrail_level must be one of {sorted(_VALID_GUARDRAIL_LEVELS)}, got {v!r}"
            )
        return v

    @field_validator("memory_mode")
    @classmethod
    def _validate_memory_mode(cls, v: str) -> str:
        if v not in _VALID_MEMORY_MODES:
            raise ValueError(
                f"memory_mode must be one of {sorted(_VALID_MEMORY_MODES)}, got {v!r}"
            )
        return v
