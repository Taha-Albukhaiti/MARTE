"""LLM response errors shared across attack and target layers."""

from __future__ import annotations


class NonRetryableLLMResponseError(RuntimeError):
    """Provider response error requiring an outer logical-phase decision."""


class LLMBackendUnavailableError(NonRetryableLLMResponseError):
    """Raised after an API/backend failure survives the configured retries."""

    def __init__(self, *, role: str, model: str, cause: BaseException) -> None:
        self.role = role
        self.model = model
        self.cause = cause
        super().__init__(
            "LLM backend unavailable after configured retries "
            f"(role={role}, model={model}): {cause}"
        )


class LLMCompletionTruncatedError(NonRetryableLLMResponseError):
    """Raised when the provider exhausts the completion-token budget."""

    def __init__(
        self,
        *,
        role: str,
        model: str,
        finish_reason: str,
        output_limit: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None,
    ) -> None:
        self.role = role
        self.model = model
        self.finish_reason = finish_reason
        self.output_limit = output_limit
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        super().__init__(
            "LLM completion truncated "
            f"(role={role}, model={model}, finish_reason={finish_reason}, "
            f"output_limit={output_limit}, prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens})"
        )


class LLMEmptyResponseError(NonRetryableLLMResponseError):
    """Raised when a completed provider response has neither text nor tools."""

    def __init__(self, *, role: str, model: str, finish_reason: str | None) -> None:
        self.role = role
        self.model = model
        self.finish_reason = finish_reason
        super().__init__(
            "LLM returned no text or tool call "
            f"(role={role}, model={model}, finish_reason={finish_reason})"
        )
