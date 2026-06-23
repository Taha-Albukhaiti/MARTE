from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from marte.llm.errors import (
    LLMBackendUnavailableError,
    LLMCompletionTruncatedError,
    LLMEmptyResponseError,
    NonRetryableLLMResponseError,
)
from marte.target.helpers.model_adapter import cast_provider, _parse_float, _parse_int, _resolve_api_key

_GLOBAL_RATE_LIMIT_LOCK = threading.Lock()
_GLOBAL_NEXT_API_CALL_AT = 0.0


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


BackendProvider = Literal["mock", "openai", "openai_compatible"]


@dataclass(frozen=True)
class BackendConfig:
    """Provider config used to construct the concrete LLM backend.

    Notes:
    - `openai_compatible` covers myLab, local vLLM/TGI, and any OpenAI-style endpoint.
    - `openai` targets the official OpenAI API (base_url optional).
    """

    provider: BackendProvider = "openai_compatible"
    base_url: str | None = "https://models.mylab.th-luebeck.dev/v1"
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_s: float | None = None
    max_retries: int = 0
    retry_backoff_s: float = 0.75
    min_interval_s: float = 0.0
    mock_response: str = "mock response"


def backend_config_from_env(env: Mapping[str, str] | None = None) -> BackendConfig:
    """Create backend config from environment variables.

    Supported env vars:
    - `LLM_PROVIDER`: mock | openai | openai_compatible
    - `LLM_BASE_URL`: endpoint base url (for openai_compatible)
    - `LLM_API_KEY`: explicit API key
    - `LLM_API_KEY_ENV`: name of env var that stores the API key
    - `LLM_TIMEOUT_S`: float timeout in seconds
    - `LLM_MAX_RETRIES`: retry count for transient backend errors
    - `LLM_RETRY_BACKOFF_S`: base backoff in seconds (exponential)
    - `LLM_MIN_INTERVAL_S`: minimum seconds between API calls in this process
    - `LLM_MOCK_RESPONSE`: response for mock provider

    Backward compatibility:
    - If `USE_UNI_LLM=0|false`, provider becomes `mock`.
    - myLab defaults remain when no provider/base_url is supplied.
    """
    e = env or os.environ

    use_uni_llm = e.get("USE_UNI_LLM", "1") not in {"0", "false", "False"}
    if not use_uni_llm:
        return BackendConfig(provider="mock", mock_response=e.get("LLM_MOCK_RESPONSE", "mock response"))

    provider = cast_provider(e.get("LLM_PROVIDER", "openai_compatible"))
    base_url = e.get("LLM_BASE_URL")
    if provider == "openai_compatible" and not base_url:
        base_url = "https://models.mylab.th-luebeck.dev/v1"

    timeout_s = _parse_float(e.get("LLM_TIMEOUT_S"))
    max_retries = _parse_int(e.get("LLM_MAX_RETRIES"), default=0)
    retry_backoff_s = _parse_float(e.get("LLM_RETRY_BACKOFF_S"))
    if retry_backoff_s is None:
        retry_backoff_s = 0.75
    min_interval_s = _parse_float(e.get("LLM_MIN_INTERVAL_S"))
    if min_interval_s is None:
        min_interval_s = 0.0

    api_key = e.get("LLM_API_KEY")
    api_key_env = e.get("LLM_API_KEY_ENV")

    return BackendConfig(provider=provider, base_url=base_url, api_key=api_key, api_key_env=api_key_env, timeout_s=timeout_s,
                         max_retries=max_retries, retry_backoff_s=retry_backoff_s, min_interval_s=min_interval_s,
                         mock_response=e.get("LLM_MOCK_RESPONSE", "mock response"))



class OpenAIClientBackend:
    """OpenAI-compatible backend using the official OpenAI Python SDK.

    Intended for OpenAI-API-compatible providers (e.g., myLab) by setting base_url.
    The TargetAgent passes `model_cfg` via kwargs (e.g. model, temperature,
    max_tokens, seed).
    """

    def __init__(self, *, base_url: str | None = None, api_key: str = "ignored", timeout_s: float | None = None) -> None:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout_s is not None:
            client_kwargs["timeout"] = timeout_s
        self._client = OpenAI(**client_kwargs)

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        model = kwargs.pop("model", None)
        if not model:
            raise ValueError("OpenAIClientBackend requires model=... in model_cfg")

        role = str(kwargs.pop("_marte_role", "unknown")).strip().lower() or "unknown"
        output_limit = _coerce_optional_int(
            kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
        )
        resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        if not resp.choices:
            raise RuntimeError("OpenAI response did not contain any choices")

        choice = resp.choices[0]
        msg = choice.message
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        prompt_tokens = _coerce_optional_int(getattr(usage, "prompt_tokens", None))
        completion_tokens = _coerce_optional_int(getattr(usage, "completion_tokens", None))
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = _coerce_optional_int(
            getattr(completion_details, "reasoning_tokens", None)
        )

        if finish == "length":
            raise LLMCompletionTruncatedError(
                role=role,
                model=model,
                finish_reason=finish,
                output_limit=output_limit,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
            )

        # Native function/tool calling — convert to our internal JSON format so
        # the rest of the pipeline (TargetAgent._parse_tool_call) works unchanged.
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            try:
                args: dict[str, Any] = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            return json.dumps({"tool": tc.function.name, **args})

        content = msg.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
            raise LLMEmptyResponseError(
                role=role,
                model=model,
                finish_reason=finish,
            )

        # Defensive handling for SDK/provider variants that may return content parts.
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text: str | None = None
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            joined = "\n".join(parts).strip()
            if joined:
                return joined

        raise LLMEmptyResponseError(
            role=role,
            model=model,
            finish_reason=finish,
        )



class MockBackend:
    """Returns a fixed response string; records all calls for test inspection."""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(list(messages))
        return self.response


class RateLimitedBackend:
    """Global in-process throttle for shared API limits."""

    def __init__(self, backend: LLMBackend, *, min_interval_s: float) -> None:
        self._backend = backend
        self._min_interval_s = max(0.0, float(min_interval_s))

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        _wait_for_global_api_slot(self._min_interval_s)
        return self._backend.complete(messages, **kwargs)


class RetryingBackend:
    """Retry wrapper for transient backend failures."""

    def __init__(self, backend: LLMBackend, *, max_retries: int, retry_backoff_s: float) -> None:
        self._backend = backend
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        attempt = 0
        while True:
            try:
                return self._backend.complete(messages, **kwargs)
            except NonRetryableLLMResponseError:
                raise
            except Exception:
                if attempt >= self._max_retries:
                    raise
                sleep_s = self._retry_backoff_s * (2 ** attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                attempt += 1


class AbortOnBackendError:
    """Expose exhausted backend failures to the logical phase retry layer."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            return self._backend.complete(messages, **kwargs)
        except NonRetryableLLMResponseError:
            raise
        except Exception as exc:
            role = str(kwargs.get("_marte_role", "unknown")).strip() or "unknown"
            model = str(kwargs.get("model", "unknown")).strip() or "unknown"
            raise LLMBackendUnavailableError(
                role=role,
                model=model,
                cause=exc,
            ) from exc


def _wait_for_global_api_slot(min_interval_s: float) -> None:
    """Ensure all API-backed roles share one conservative request cadence."""
    if min_interval_s <= 0:
        return

    global _GLOBAL_NEXT_API_CALL_AT
    with _GLOBAL_RATE_LIMIT_LOCK:
        now = time.monotonic()
        wait_s = _GLOBAL_NEXT_API_CALL_AT - now
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        _GLOBAL_NEXT_API_CALL_AT = now + min_interval_s


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None



def build_backend(config: BackendConfig) -> LLMBackend:
    """Build a concrete backend from typed config."""
    backend: LLMBackend
    if config.provider == "mock":
        backend = MockBackend(config.mock_response)
    elif config.provider in {"openai", "openai_compatible"}:
        api_key = _resolve_api_key(config)
        backend = OpenAIClientBackend(base_url=config.base_url, api_key=api_key, timeout_s=config.timeout_s)
    else:
        raise ValueError(f"unsupported provider: {config.provider!r}")

    if config.min_interval_s > 0:
        backend = RateLimitedBackend(backend=backend, min_interval_s=config.min_interval_s)

    if config.max_retries > 0:
        backend = RetryingBackend(
            backend=backend,
            max_retries=config.max_retries,
            retry_backoff_s=config.retry_backoff_s,
        )

    if config.provider in {"openai", "openai_compatible"}:
        backend = AbortOnBackendError(backend)

    return backend



class ModelAdapter:
    """Abstracts LLM inference.

    Swap the backend at construction time to change provider:
        ModelAdapter(MockBackend("test"))              — offline / tests
        ModelAdapter(OpenAIClientBackend(...), cfg)    — production
    """

    def __init__(self, backend: LLMBackend, model_cfg: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.model_cfg: dict[str, Any] = model_cfg or {}

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> str:
        kwargs = dict(self.model_cfg)
        if tools:
            kwargs["tools"] = tools
            kwargs.setdefault("tool_choice", "auto")
        return self.backend.complete(messages, **kwargs)
