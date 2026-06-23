"""Condition, scenario, and guardrail helpers for the MARTE-thesis-worktree CLI."""

from __future__ import annotations

from marte.attack.oracle import SandboxPolicy
from marte.experiment.runner import format_condition
from marte.target.guardrail import GuardrailLevel


def _sandbox_policy_for_scenario(scenario: str) -> SandboxPolicy:
    if scenario == "file":
        return SandboxPolicy(scenario="file", allowed_paths=("public/",))
    return SandboxPolicy(scenario="sql", protected_tables=("credentials",), allowed_operations={})


def _tool_allowlist_for_scenario(scenario: str) -> list[str]:
    return ["file"] if scenario == "file" else ["sql"]


def _guardrail_from_text(raw: str) -> GuardrailLevel:
    value = raw.strip().lower()
    if value == "moderate":
        return GuardrailLevel.MODERATE
    if value == "enterprise":
        return GuardrailLevel.ENTERPRISE
    raise ValueError(f"invalid guardrail level: {raw!r}")


def _filter_conditions(tokens: list[str], all_conds: tuple[tuple[str, str, str], ...]) -> tuple[tuple[str, str, str], ...]:
    """Filter the condition matrix by shorthand tokens or exact names."""
    if not tokens or tokens == ["all"]:
        return all_conds

    shorthands = {
        "stm_only": lambda s, g, m: m == "stm_only",
        "rag_ltm": lambda s, g, m: m == "rag_ltm",
        "sql": lambda s, g, m: s == "sql",
        "file": lambda s, g, m: s == "file",
        "moderate": lambda s, g, m: g == "moderate",
        "enterprise": lambda s, g, m: g == "enterprise",
        "all": lambda s, g, m: True,
    }

    matched: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for cond in all_conds:
        scenario, guardrail, memory = cond
        cond_name = format_condition(scenario, guardrail, memory)
        for token in tokens:
            predicate = shorthands.get(token)
            include = predicate(scenario, guardrail, memory) if predicate else token == cond_name
            if include and cond not in seen:
                matched.append(cond)
                seen.add(cond)
                break

    if not matched:
        known = sorted(shorthands) + [format_condition(s, g, m) for s, g, m in all_conds]
        raise ValueError(f"No conditions matched {tokens!r}. Valid tokens: {', '.join(known)}")
    return tuple(matched)
