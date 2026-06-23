from __future__ import annotations

import json
from typing import Any, Mapping

from marte.attack.helpers.mutator import (
    _build_mutator_payload,
    _compose_final_prompt,
    _fallback_prompt,
    _looks_like_refusal,
    _needs_enterprise_fallback,
    _sanitize_generated_prompt_with_diagnostics,
)
from marte.attack.model_adapter import AttackLLMBackend, AttackModelAdapter
from marte.attack.planner import AttackPlan
from marte.llm.errors import NonRetryableLLMResponseError
from marte.prompts.loader import load_prompt


_MUTATOR_SYSTEM_PROMPT = load_prompt("attack/mutator_system.txt")


class Mutator:
    """Turns an attack plan into a concrete prompt candidate.

    Within-run adaptation is supplied exclusively through the bounded STM
    trajectory in ``memory_context``. The Mutator therefore has no private
    conversation history that could bypass the configured STM capacity.
    """

    def __init__(self, model_backend: AttackLLMBackend, model_cfg: dict[str, Any] | None = None) -> None:
        self._adapter = AttackModelAdapter(model_backend, model_cfg or {})
        self.last_raw_response: str = ""
        self.last_sanitized_response: str = ""
        self.last_sanitization_actions: list[str] = []
        self.last_used_fallback: bool = False
        self.last_fallback_reason: str = ""
        self.last_error: str | None = None
        self.last_final_prompt: str = ""

    def set_seed(self, seed: int) -> None:
        self._adapter.set_seed(seed)

    def mutate(self, plan: AttackPlan, memory_context: Mapping[str, Any] | None) -> str:
        memory = dict(memory_context or {})
        self.last_error = None

        generated_raw = self._complete_mutation_request(plan, memory).strip()
        self.last_raw_response = generated_raw
        generated, sanitization_actions = _sanitize_generated_prompt_with_diagnostics(
            generated_raw
        )
        self.last_sanitized_response = generated
        self.last_sanitization_actions = sanitization_actions
        self.last_used_fallback = False
        self.last_fallback_reason = ""

        if not generated:
            self.last_used_fallback = True
            self.last_fallback_reason = (
                "invalid_model_output"
                if self.last_error
                else (
                    "unsafe_generated_output"
                    if generated_raw and sanitization_actions
                    else "empty_response"
                )
            )
            generated = _fallback_prompt(plan, memory)
        elif _looks_like_refusal(generated):
            self.last_used_fallback = True
            self.last_fallback_reason = "refusal"
            generated = _fallback_prompt(plan, memory)
        elif _needs_enterprise_fallback(plan, generated, memory):
            self.last_used_fallback = True
            self.last_fallback_reason = "enterprise_unproductive_prompt"
            generated = _fallback_prompt(plan, memory)

        # Stable prefix guarantees traceability and yields different prompts
        # for different plans even if the backend emits the same body.
        final_prompt = _compose_final_prompt(plan.strategy, generated)
        self.last_final_prompt = final_prompt
        return final_prompt

    def _complete_mutation_request(self, plan: AttackPlan, memory_context: dict[str, Any]) -> str:
        payload = _build_mutator_payload(plan, memory_context)
        user_content = json.dumps(payload, ensure_ascii=True)
        messages = [
            {"role": "system", "content": _MUTATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            response = self._adapter.complete(messages)
            return response
        except NonRetryableLLMResponseError as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            return ""
