from __future__ import annotations

from types import SimpleNamespace

import pytest

from marte.llm.errors import (
    LLMBackendUnavailableError,
    LLMCompletionTruncatedError,
    LLMEmptyResponseError,
)
from marte.target.model_adapter import (
    AbortOnBackendError,
    BackendConfig,
    OpenAIClientBackend,
    RateLimitedBackend,
    RetryingBackend,
    backend_config_from_env,
    build_backend,
)


class FlakyBackend:
    def __init__(self, fail_times: int, response: str = "ok") -> None:
        self._remaining = fail_times
        self._response = response
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ARG002
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("temporary backend failure")
        return self._response


class StaticResponseClient:
    def __init__(self, response) -> None:  # noqa: ANN001
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )


def _openai_backend_with_response(response) -> OpenAIClientBackend:  # noqa: ANN001
    backend = object.__new__(OpenAIClientBackend)
    backend._client = StaticResponseClient(response)  # noqa: SLF001
    return backend


def _completion_response(
    *,
    content,
    finish_reason: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 4096,
    reasoning_tokens: int = 4000,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
    )


def test_retrying_backend_succeeds_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("marte.target.model_adapter.time.sleep", lambda sec: sleeps.append(sec))

    backend = FlakyBackend(fail_times=2, response="done")
    retrying = RetryingBackend(backend, max_retries=2, retry_backoff_s=0.5)

    result = retrying.complete([{"role": "user", "content": "hi"}])

    assert result == "done"
    assert backend.calls == 3
    assert sleeps == [0.5, 1.0]


def test_retrying_backend_raises_after_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("marte.target.model_adapter.time.sleep", lambda sec: None)

    backend = FlakyBackend(fail_times=3)
    retrying = RetryingBackend(backend, max_retries=1, retry_backoff_s=0.1)

    with pytest.raises(RuntimeError, match="temporary backend failure"):
        retrying.complete([{"role": "user", "content": "hi"}])
    assert backend.calls == 2


def test_exhausted_backend_error_is_converted_to_run_aborting_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("marte.target.model_adapter.time.sleep", lambda sec: None)
    backend = FlakyBackend(fail_times=3)
    retrying = RetryingBackend(
        backend,
        max_retries=1,
        retry_backoff_s=0.1,
    )
    strict = AbortOnBackendError(retrying)

    with pytest.raises(LLMBackendUnavailableError) as caught:
        strict.complete(
            [{"role": "user", "content": "hi"}],
            model="glm-4.7",
            _marte_role="planner",
        )

    assert backend.calls == 2
    assert caught.value.role == "planner"
    assert caught.value.model == "glm-4.7"


def test_completion_length_raises_diagnostic_nonretryable_error() -> None:
    backend = _openai_backend_with_response(
        _completion_response(content=None, finish_reason="length")
    )

    with pytest.raises(LLMCompletionTruncatedError) as caught:
        backend.complete(
            [{"role": "user", "content": "test"}],
            model="gpt-oss-120b",
            max_tokens=4096,
            _marte_role="target",
        )

    error = caught.value
    assert error.role == "target"
    assert error.model == "gpt-oss-120b"
    assert error.output_limit == 4096
    assert error.prompt_tokens == 100
    assert error.completion_tokens == 4096
    assert error.reasoning_tokens == 4000


def test_retrying_backend_does_not_retry_truncated_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "marte.target.model_adapter.time.sleep",
        lambda sec: sleeps.append(sec),
    )

    class TruncatedBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ARG002
            self.calls += 1
            raise LLMCompletionTruncatedError(
                role="target",
                model="gpt-oss-120b",
                finish_reason="length",
                output_limit=4096,
                prompt_tokens=100,
                completion_tokens=4096,
                reasoning_tokens=4000,
            )

    backend = TruncatedBackend()
    retrying = RetryingBackend(backend, max_retries=5, retry_backoff_s=1.0)

    with pytest.raises(LLMCompletionTruncatedError):
        retrying.complete([{"role": "user", "content": "test"}])

    assert backend.calls == 1
    assert sleeps == []


def test_empty_completed_response_raises_nonretryable_error() -> None:
    backend = _openai_backend_with_response(
        _completion_response(
            content="",
            finish_reason="stop",
            completion_tokens=0,
            reasoning_tokens=0,
        )
    )

    with pytest.raises(LLMEmptyResponseError, match="role=judge"):
        backend.complete(
            [{"role": "user", "content": "test"}],
            model="glm-4.7",
            _marte_role="judge",
        )


def test_backend_config_reads_retry_fields_from_env() -> None:
    cfg = backend_config_from_env(
        {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MAX_RETRIES": "4",
            "LLM_RETRY_BACKOFF_S": "1.25",
            "LLM_MIN_INTERVAL_S": "4.2",
        }
    )
    assert cfg.max_retries == 4
    assert cfg.retry_backoff_s == 1.25
    assert cfg.min_interval_s == 4.2


def test_build_backend_wraps_mock_with_retry_when_configured() -> None:
    backend = build_backend(
        BackendConfig(provider="mock", mock_response="x", max_retries=2, retry_backoff_s=0.2)
    )
    assert isinstance(backend, RetryingBackend)
    assert backend.complete([]) == "x"


def test_build_backend_wraps_mock_with_rate_limit_when_configured() -> None:
    backend = build_backend(
        BackendConfig(provider="mock", mock_response="x", min_interval_s=4.2)
    )
    assert isinstance(backend, RateLimitedBackend)
