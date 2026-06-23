from __future__ import annotations

import json
from typing import Any


def _try_parse_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _has_local_progress(payload: dict[str, Any]) -> bool:
    if str(payload.get("input_decision", "deny")).lower().strip() != "allow":
        return False
    if str(payload.get("output_decision", "deny")).lower().strip() != "allow":
        return False
    tool_trace = payload.get("tool_trace", [])
    policy_decisions = payload.get("policy_decisions", [])
    if not isinstance(tool_trace, list):
        return False
    policies = policy_decisions if isinstance(policy_decisions, list) else []
    for idx, call in enumerate(tool_trace):
        if not isinstance(call, dict):
            continue
        decision = str(policies[idx] if idx < len(policies) else "deny").lower().strip()
        if decision == "allow" and not call.get("error"):
            return True
    return False
