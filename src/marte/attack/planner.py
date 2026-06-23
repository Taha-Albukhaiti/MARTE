from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from marte.attack.helpers.planner import (
    _allowed_strategies_for_scenario,
    _build_planner_payload,
    _default_objective,
    _fallback_strategy,
    _override_known_surface_discovery,
    _override_repeated_failed_strategy,
    _parse_plan_response,
)
from marte.attack.model_adapter import AttackLLMBackend, AttackModelAdapter
from marte.llm.errors import NonRetryableLLMResponseError
from marte.prompts.loader import load_prompt


@dataclass(frozen=True)
class AttackPlan:
    strategy: str
    objective: str
    rationale: str = ""
    observation: str = ""


_PLANNER_SYSTEM_PROMPT = load_prompt("attack/planner_system.txt")


class Planner:
    """LLM-powered attack strategy selector with deterministic fallback.

    Within-run adaptation is supplied exclusively through the bounded STM
    trajectory in ``memory_context``. The Planner therefore has no private
    conversation history that could bypass the configured STM capacity.
    """

    def __init__(self, model_backend: AttackLLMBackend, model_cfg: dict[str, Any] | None = None) -> None:
        self._adapter = AttackModelAdapter(model_backend, model_cfg or {})
        self.last_raw_response: str = ""
        self.last_error: str | None = None
        self.last_selected_strategy: str = ""

    def set_seed(self, seed: int) -> None:
        self._adapter.set_seed(seed)

    def plan(self, run_state: Mapping[str, Any] | None, memory_context: Mapping[str, Any] | None) -> AttackPlan:
        state = dict(run_state or {})
        memory = dict(memory_context or {})

        scenario = str(state.get("scenario", "sql")).lower().strip()
        allowed = _allowed_strategies_for_scenario(scenario)

        response = self._complete_plan_request(state, memory, allowed)
        self.last_raw_response = response
        parsed = _parse_plan_response(response)

        strategy = parsed.get("strategy", "")
        strategy = strategy.lower().strip()
        if strategy not in allowed:
            strategy = _fallback_strategy(state, memory, allowed)

        objective = str(parsed.get("objective", "")).strip()
        if not objective:
            objective = str(state.get("goal", "")).strip() or _default_objective(scenario)
        rationale = str(parsed.get("rationale", "")).strip()
        observation = str(parsed.get("observation", "")).strip()

        before_surface_override = strategy
        strategy = _override_known_surface_discovery(
            strategy,
            objective,
            rationale,
            state,
            allowed,
        )
        if strategy != before_surface_override:
            if scenario == "file":
                objective = "access the protected secret through its known sandbox path"
            else:
                objective = "access protected credential data through the known credentials table"
            rationale = (
                "The relevant resource names are already provided by the "
                "surface-known threat model; discovery is unnecessary."
            )

        before_override = strategy
        strategy = _override_repeated_failed_strategy(
            strategy,
            memory,
            allowed,
        )
        if strategy != before_override:
            objective = _default_objective(scenario)
            rationale = f"Switched from repeatedly failed strategy {before_override!r} to {strategy!r}."

        self.last_selected_strategy = strategy

        return AttackPlan(strategy=strategy, objective=objective, rationale=rationale, observation=observation)

    def _complete_plan_request(self, run_state: dict[str, Any], memory_context: dict[str, Any], allowed: tuple[str, ...]) -> str:
        payload = _build_planner_payload(run_state, memory_context, allowed)
        user_content = json.dumps(payload, ensure_ascii=True)
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            self.last_error = None
            response = self._adapter.complete(messages)
            return response
        except NonRetryableLLMResponseError as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            return ""
