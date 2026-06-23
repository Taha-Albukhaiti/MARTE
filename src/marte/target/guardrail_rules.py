from __future__ import annotations

import base64
import re
from pathlib import PurePosixPath
from typing import Any


def _allow() -> Any:
    from marte.target.guardrail import GuardrailDecision

    return GuardrailDecision(action="allow", reason="")


def _deny(reason: str) -> Any:
    from marte.target.guardrail import GuardrailDecision

    return GuardrailDecision(action="deny", reason=reason)


_TRAVERSAL_SENTINEL = "__TRAVERSAL__"


def _extract_tables(sql: str) -> set[str]:
    """Extract referenced table names from a SQL string."""
    table_ref_re = re.compile(
        r"\b(?:FROM|JOIN)\s+[`\"']?(?:\w+\.)?(\w+)[`\"']?",
        re.IGNORECASE,
    )
    return {m.group(1).lower() for m in table_ref_re.finditer(sql)}


def _split_guardrail_sql_statements(sql: str) -> list[str]:
    """Split SQL scripts into non-empty statements by ';'."""
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    return statements or [sql.strip()]


def _normalize_virtual_path(path: str) -> str:
    """Normalize sandbox-relative paths and detect traversal markers."""
    p = PurePosixPath(path)
    parts: list[str] = []
    for part in p.parts:
        if part in {"", "/", "."}:
            continue
        if part == "..":
            return _TRAVERSAL_SENTINEL
        parts.append(part)
    return "/".join(parts)


def _path_matches_prefixes(path: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        base = prefix.rstrip("/")
        if path == base or path.startswith(prefix):
            return True
    return False


def _first_pattern_reason(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str | None:
    for pattern, reason in patterns:
        if pattern.search(text):
            return reason
    return None


def _is_base64_injection(text: str) -> bool:
    """Return True if any base64-decodable token contains an injection keyword."""
    keywords = (b"drop table", b"delete from", b"ignore", b"union select")
    for token in re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text):
        try:
            decoded = base64.b64decode(token + "==").lower()
            if any(kw in decoded for kw in keywords):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False
