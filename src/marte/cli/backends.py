"""LLM backend and role-model configuration helpers for the MARTE-thesis-worktree CLI."""

from __future__ import annotations

import os
from typing import Any

from marte.cli.env import _first_non_empty
from marte.target.helpers.model_adapter import _parse_float, _parse_int
from marte.target.model_adapter import BackendConfig, build_backend as build_llm_backend

_OPTIONAL_EXTRAS_INSTALL_HINT = 'pip install -e ".[dev,llm,rag]"'
_OPTIONAL_DEPENDENCIES = {"chromadb", "openai", "sentence_transformers"}

_DEFAULT_ATTACK_MODEL = "qwen3.6-27b"
_DEFAULT_TARGET_MODEL = "gpt-oss-120b"
_DEFAULT_MODEL = "gpt-oss-120b"
_ROLE_DEFAULT_MODELS: dict[str, str] = {
    "ATTACK": _DEFAULT_ATTACK_MODEL,
    "TARGET": _DEFAULT_TARGET_MODEL,
    "JUDGE": _DEFAULT_ATTACK_MODEL,
}


def _default_role_model(prefix: str) -> str:
    """Resolve role model: role env → shared env → role default."""
    role_default = _ROLE_DEFAULT_MODELS.get(prefix, _DEFAULT_MODEL)
    return _first_non_empty(os.getenv(f"{prefix}_LLM_MODEL"), os.getenv("LLM_MODEL"), role_default) or role_default


def _role_model_cfg(prefix: str, model: str, default_temperature: float) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": model,
        "_marte_role": prefix.lower(),
    }

    temperature = _parse_float(_first_non_empty(os.getenv(f"{prefix}_LLM_TEMPERATURE"), os.getenv("LLM_TEMPERATURE")))
    cfg["temperature"] = temperature if temperature is not None else default_temperature

    max_tokens = _parse_int(_first_non_empty(os.getenv(f"{prefix}_LLM_MAX_TOKENS"), os.getenv("LLM_MAX_TOKENS")))
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens

    seed = _parse_int(_first_non_empty(os.getenv(f"{prefix}_LLM_SEED"), os.getenv("LLM_SEED")))
    if seed is not None:
        cfg["seed"] = seed

    return cfg


def _has_role_backend_config(prefix: str) -> bool:
    suffixes = (
        "PROVIDER",
        "BASE_URL",
        "API_KEY",
        "API_KEY_ENV",
        "TIMEOUT_S",
        "MAX_RETRIES",
        "RETRY_BACKOFF_S",
        "MIN_INTERVAL_S",
        "MOCK_RESPONSE",
    )
    return any(os.getenv(f"{prefix}_LLM_{suffix}") for suffix in suffixes)


def _build_role_backend_config(prefix: str) -> BackendConfig:
    provider = _first_non_empty(os.getenv(f"{prefix}_LLM_PROVIDER"), os.getenv("LLM_PROVIDER"), "openai_compatible")
    base_url = _first_non_empty(os.getenv(f"{prefix}_LLM_BASE_URL"), os.getenv("LLM_BASE_URL"))
    api_key = _first_non_empty(os.getenv(f"{prefix}_LLM_API_KEY"), os.getenv("LLM_API_KEY"))
    api_key_env = _first_non_empty(os.getenv(f"{prefix}_LLM_API_KEY_ENV"), os.getenv("LLM_API_KEY_ENV"))
    timeout_s = _parse_float(_first_non_empty(os.getenv(f"{prefix}_LLM_TIMEOUT_S"), os.getenv("LLM_TIMEOUT_S")))
    max_retries = _parse_int(_first_non_empty(os.getenv(f"{prefix}_LLM_MAX_RETRIES"), os.getenv("LLM_MAX_RETRIES")), default=0)
    retry_backoff_s = _parse_float(
        _first_non_empty(os.getenv(f"{prefix}_LLM_RETRY_BACKOFF_S"), os.getenv("LLM_RETRY_BACKOFF_S")),
        default=0.75,
    )
    min_interval_s = _parse_float(
        _first_non_empty(os.getenv(f"{prefix}_LLM_MIN_INTERVAL_S"), os.getenv("LLM_MIN_INTERVAL_S")),
        default=0.0,
    )

    return BackendConfig(provider=provider or "openai_compatible", base_url=base_url, api_key=api_key,
                         api_key_env=api_key_env, timeout_s=timeout_s, max_retries=max_retries or 0,
                         retry_backoff_s=retry_backoff_s or 0.75, min_interval_s=min_interval_s or 0.0,
                         mock_response=os.getenv(f"{prefix}_LLM_MOCK_RESPONSE", "mock response"))


def _build_api_backend(prefix: str):
    config = _build_role_backend_config(prefix)

    if config.provider == "mock":
        raise ValueError(f"--mode api requires a real provider for {prefix} (got provider='mock').")
    if config.provider in {"openai", "openai_compatible"} and not config.base_url:
        raise ValueError(f"Missing base URL for {prefix}. Set {prefix}_LLM_BASE_URL or LLM_BASE_URL.")
    if not _has_role_api_key(config, prefix):
        raise ValueError(
            f"Missing API key for {prefix}. Set {prefix}_LLM_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or MYLAB_API_KEY."
        )

    try:
        return build_llm_backend(config)
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise RuntimeError(
                "Missing optional LLM dependency 'openai'. "
                f"Install the full experiment extras with: {_OPTIONAL_EXTRAS_INSTALL_HINT}"
            ) from exc
        raise


def _has_role_api_key(config: BackendConfig, prefix: str) -> bool:
    direct = [
        config.api_key,
        os.getenv(f"{prefix}_LLM_API_KEY"),
        os.getenv("LLM_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("MYLAB_API_KEY"),
    ]
    if any(value and value.strip() for value in direct):
        return True
    return bool(config.api_key_env and os.getenv(config.api_key_env))


def _is_optional_dependency_error(exc: BaseException) -> bool:
    return "Missing optional" in str(exc) or _missing_optional_dependency_name(exc) is not None


def _missing_optional_dependency_name(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            missing = current.name or ""
            if missing in _OPTIONAL_DEPENDENCIES:
                return missing
        current = current.__cause__ or current.__context__
    return None


def _format_cli_error(exc: BaseException) -> str:
    message = str(exc)
    missing = _missing_optional_dependency_name(exc)
    if missing and _OPTIONAL_EXTRAS_INSTALL_HINT not in message:
        return (
            f"{message}\nMissing optional dependency '{missing}'. "
            f"Install the full experiment extras with: {_OPTIONAL_EXTRAS_INSTALL_HINT}"
        )
    return message
