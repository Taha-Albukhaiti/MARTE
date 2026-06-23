from __future__ import annotations

import textwrap

from marte.attack.planner import AttackPlan


def _wrapped_print(prefix: str, message: str) -> None:
    wrapped = textwrap.fill(
        f"{prefix}{message}",
        width=120,
        subsequent_indent=" " * len(prefix),
    )
    print(wrapped)


def log_plan_execution(plan: AttackPlan) -> None:
    prefix = "[Planer-TARGET-Attacking]  "
    message = (
        f"Executing plan with strategy: {plan.strategy}, "
        f"Objective: {plan.objective}, Rationale: {plan.rationale}"
    )
    _wrapped_print(prefix, message)


def log_mutator_execution(prompt: str) -> None:
    prefix = "[Mutator-TARGET-Attacking] "
    _wrapped_print(prefix, f"Executing step with prompt: {prompt} ...")


def log_target_execution(response_text: str) -> None:
    prefix = "[Target-Text-Response] "
    _wrapped_print(prefix, f"Executing target response  {response_text}")
