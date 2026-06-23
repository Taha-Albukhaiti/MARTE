from __future__ import annotations

import hashlib
import re
from typing import Any


def memory_context_length(memory_context: Any) -> int:
    if not isinstance(memory_context, dict):
        return 0
    stm = memory_context.get("stm_recent", [])
    ltm = memory_context.get("ltm_hits", [])
    stm_len = len(stm) if isinstance(stm, list) else 0
    ltm_len = len(ltm) if isinstance(ltm, list) else 0
    return stm_len + ltm_len


def ltm_hits_count(memory_context: Any) -> int:
    if not isinstance(memory_context, dict):
        return 0
    ltm = memory_context.get("ltm_hits", [])
    return len(ltm) if isinstance(ltm, list) else 0


def set_ltm_scope(memory_gateway: Any, scope_name: str) -> None:
    set_scope = getattr(memory_gateway, "set_ltm_scope", None)
    if callable(set_scope):
        set_scope(scope_name)


def ltm_scope_name(experiment_id: str, condition: str) -> str:
    digest = hashlib.sha1(f"{experiment_id}:{condition}".encode("utf-8")).hexdigest()[:12]
    condition_token = collection_token(condition)[:32] or "condition"
    return f"marte_{condition_token}_{digest}"


def collection_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not token:
        return "scope"
    if not token[0].isalnum():
        token = f"s{token}"
    if not token[-1].isalnum():
        token = f"{token}0"
    return token
