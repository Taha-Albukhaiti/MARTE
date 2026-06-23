from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CONDITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("sql_moderate_stm_only", "sql_moderate_rag_ltm"),
    ("sql_enterprise_stm_only", "sql_enterprise_rag_ltm"),
    ("file_moderate_stm_only", "file_moderate_rag_ltm"),
    ("file_enterprise_stm_only", "file_enterprise_rag_ltm"),
)

EXPECTED_CONDITIONS: tuple[str, ...] = tuple(
    condition
    for pair in CONDITION_PAIRS
    for condition in pair
)

_VALID_STOP_REASONS = {"breach_stop", "budget_exhausted", "error"}


@dataclass(frozen=True)
class Violation:
    """One integrity issue found before statistical evaluation."""

    level: Literal["error", "warning"]
    message: str
    code: str = "validation"
    condition: str | None = None
    run_id: int | None = None


def _looks_like_backend_failure(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(
        marker in lowered
        for marker in (
            "[llm error]",
            "[target error]",
            "error code: 429",
            "rate limit",
            "failed to process request",
            "service unavailable",
            "connection reset",
            "connection refused",
            "connection error",
            "timed out",
            "timeout",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
    )
