"""Shared LLM contracts used across MARTE-thesis-worktree layers."""

from marte.llm.errors import (
    LLMBackendUnavailableError,
    LLMCompletionTruncatedError,
    LLMEmptyResponseError,
    NonRetryableLLMResponseError,
)

__all__ = [
    "LLMCompletionTruncatedError",
    "LLMBackendUnavailableError",
    "LLMEmptyResponseError",
    "NonRetryableLLMResponseError",
]
