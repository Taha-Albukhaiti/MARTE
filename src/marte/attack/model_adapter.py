from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AttackLLMBackend(Protocol):
    """Attack-layer local backend contract.

    Kept separate from target.model_adapter to avoid cross-layer coupling.
    """

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


class AttackModelAdapter:
    """Thin adapter for Planner/Mutator LLM calls."""

    def __init__(self, backend: AttackLLMBackend, model_cfg: dict[str, Any] | None = None) -> None:
        self._backend = backend
        self._model_cfg = model_cfg or {}

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._backend.complete(messages, **self._model_cfg)

    def update_model_cfg(self, updates: dict[str, Any]) -> None:
        self._model_cfg.update(updates)

    def set_seed(self, seed: int) -> None:
        self._model_cfg["seed"] = seed
