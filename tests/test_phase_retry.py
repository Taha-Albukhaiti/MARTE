from __future__ import annotations

import pytest

from marte.llm.errors import LLMBackendUnavailableError
from marte.llm.phase_retry import PhaseRetryPolicy, retry_llm_phase


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _backend_error(status_code: int) -> LLMBackendUnavailableError:
    return LLMBackendUnavailableError(
        role="target",
        model="test-model",
        cause=_HTTPError(status_code),
    )


def test_retry_policy_defaults_to_finite_attempts(monkeypatch) -> None:
    monkeypatch.delenv("MARTE_STEP_MAX_ATTEMPTS", raising=False)
    assert PhaseRetryPolicy().max_attempts == 6
    assert PhaseRetryPolicy.from_env().max_attempts == 6


def test_zero_retry_limit_explicitly_enables_unlimited_attempts(monkeypatch) -> None:
    monkeypatch.setenv("MARTE_STEP_MAX_ATTEMPTS", "0")
    assert PhaseRetryPolicy.from_env().max_attempts is None


def test_rate_limit_is_retried_inside_same_phase() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _backend_error(429)
        return "ok"

    result = retry_llm_phase(
        operation,
        phase="target",
        policy=PhaseRetryPolicy(
            max_attempts=2,
            backoff_s=0,
            max_backoff_s=0,
        ),
    )

    assert result == "ok"
    assert calls == 2


def test_permanent_not_found_error_is_not_retried() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise _backend_error(404)

    with pytest.raises(LLMBackendUnavailableError):
        retry_llm_phase(
            operation,
            phase="target",
            policy=PhaseRetryPolicy(
                max_attempts=None,
                backoff_s=0,
                max_backoff_s=0,
            ),
        )

    assert calls == 1
