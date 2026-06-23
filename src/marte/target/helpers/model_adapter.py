from __future__ import annotations

import os
from typing import Any


def cast_provider(raw: str) -> str:
    provider = raw.strip().lower()
    allowed: set[str] = {"mock", "openai", "openai_compatible"}
    if provider not in allowed:
        raise ValueError(
            f"unsupported LLM_PROVIDER={raw!r}. Allowed: {sorted(allowed)}"
        )
    return provider


def _parse_float(raw: str | None, default: float | None = None) -> float | None:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return float(value)


def _parse_int(raw: str | None, default: int | None = None) -> int | None:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return int(value)


def _resolve_api_key(config: Any) -> str:
    if config.api_key:
        return config.api_key

    if config.api_key_env:
        if value := os.getenv(config.api_key_env):
            return value

    # Backward-compatible defaults for existing MARTE-thesis-worktree setup.
    if value := os.getenv("MYLAB_API_KEY"):
        return value
    if value := os.getenv("OPENAI_API_KEY"):
        return value

    if config.provider == "openai_compatible":
        return "ignored"

    raise ValueError(
        "No API key configured. Set LLM_API_KEY or LLM_API_KEY_ENV "
        "(or OPENAI_API_KEY for provider='openai')."
    )
