from __future__ import annotations

import re


_OPERATION_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*SELECT\b", re.IGNORECASE), "SELECT"),
    (re.compile(r"^\s*PRAGMA\b", re.IGNORECASE), "PRAGMA"),
    (re.compile(r"^\s*INSERT\s+INTO\b", re.IGNORECASE), "INSERT"),
    (re.compile(r"^\s*UPDATE\b", re.IGNORECASE), "UPDATE"),
    (re.compile(r"^\s*DELETE\b", re.IGNORECASE), "DELETE"),
    (re.compile(r"^\s*(CREATE|DROP|ALTER)\b", re.IGNORECASE), "DDL"),
]

_TABLE_RE: dict[str, re.Pattern[str]] = {
    "SELECT": re.compile(r"\bFROM\s+[`\"\[]?(\w+)[`\"\]]?", re.IGNORECASE),
    "PRAGMA": re.compile(
        r"\bPRAGMA\s+(?:\w+\.)?table_info\s*\(\s*[`\"'\[]?(\w+)[`\"'\]]?\s*\)",
        re.IGNORECASE,
    ),
    "INSERT": re.compile(r"\bINSERT\s+INTO\s+[`\"\[]?(\w+)[`\"\]]?", re.IGNORECASE),
    "UPDATE": re.compile(r"\bUPDATE\s+[`\"\[]?(\w+)[`\"\]]?", re.IGNORECASE),
    "DELETE": re.compile(r"\bDELETE\s+FROM\s+[`\"\[]?(\w+)[`\"\]]?", re.IGNORECASE),
    "DDL": re.compile(r"\b(?:TABLE|INDEX)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?", re.IGNORECASE),
}


def _parse_operation(query: str) -> str:
    for pattern, op in _OPERATION_RE:
        if pattern.match(query):
            return op
    return "UNKNOWN"


def _parse_target(query: str, operation: str) -> str:
    pattern = _TABLE_RE.get(operation)
    if pattern:
        m = pattern.search(query)
        if m:
            return m.group(1)
    return ""


def _split_sql_tool_statements(query: str) -> list[str]:
    return [part.strip() for part in query.split(";") if part.strip()]


def _parse_multi_metadata(query: str) -> tuple[list[str], list[str]]:
    operations: list[str] = []
    targets: list[str] = []
    for statement in _split_sql_tool_statements(query):
        op = _parse_operation(statement)
        operations.append(op)
        target = _parse_target(statement, op)
        if target:
            targets.append(target)
    return operations, targets
