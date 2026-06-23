from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from marte.llm.errors import (
    LLMBackendUnavailableError,
    NonRetryableLLMResponseError,
)

_T = TypeVar("_T")
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseRetryPolicy:
    """Retry policy for one logical phase of an attack step.

    The default permits six attempts. ``max_attempts=None`` is still available
    when the environment explicitly sets ``MARTE_STEP_MAX_ATTEMPTS=0``.
    Permanent client errors such as an invalid model name or authentication
    failure fail fast because repeating them cannot recover the experiment.
    """

    max_attempts: int | None = 6
    backoff_s: float = 5.0
    max_backoff_s: float = 60.0

    @classmethod
    def from_env(cls) -> PhaseRetryPolicy:
        raw_attempts = str(os.getenv("MARTE_STEP_MAX_ATTEMPTS", "6")).strip()
        raw_backoff = str(os.getenv("MARTE_STEP_RETRY_BACKOFF_S", "5")).strip()
        raw_max_backoff = str(
            os.getenv("MARTE_STEP_RETRY_MAX_BACKOFF_S", "60")
        ).strip()

        attempts = int(raw_attempts or "6")
        return cls(
            max_attempts=None if attempts <= 0 else attempts,
            backoff_s=max(0.0, float(raw_backoff or "5")),
            max_backoff_s=max(0.0, float(raw_max_backoff or "60")),
        )


def retry_llm_phase(
    operation: Callable[[], _T],
    *,
    phase: str,
    policy: PhaseRetryPolicy,
    on_retry: Callable[[int, NonRetryableLLMResponseError], None] | None = None,
) -> _T:
    """Repeat one LLM phase without advancing the scientific step counter."""

    attempt = 1
    while True:
        try:
            return operation()
        except NonRetryableLLMResponseError as exc:
            if not _is_transient_phase_error(exc):
                raise
            if policy.max_attempts is not None and attempt >= policy.max_attempts:
                raise

            retry_number = attempt
            if on_retry is not None:
                on_retry(retry_number, exc)

            delay = min(
                policy.backoff_s * (2 ** (retry_number - 1)),
                policy.max_backoff_s,
            )
            _log.warning(
                "Retrying LLM phase %s after technical failure "
                "(retry=%d, delay=%.1fs): %s",
                phase,
                retry_number,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
            attempt += 1


def _is_transient_phase_error(exc: NonRetryableLLMResponseError) -> bool:
    """Return false only for clearly permanent provider/client failures."""

    if not isinstance(exc, LLMBackendUnavailableError):
        # Truncated and empty completed responses may vary on a repeated call.
        return True

    cause: BaseException | None = exc.cause
    while cause is not None:
        status = getattr(cause, "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 425, 429} or status >= 500
        cause = cause.__cause__ or cause.__context__

    # Network failures and provider exceptions without an HTTP status are
    # treated as transient.
    return True
