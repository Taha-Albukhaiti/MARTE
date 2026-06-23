from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from marte.target.sandbox import SandboxBoundary


def _db_schema_lines(db_path: Path) -> list[str]:
    """Return compact schema lines for all user tables in the SQLite DB."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cur.fetchall()]
            lines: list[str] = []
            for table in tables:
                cur.execute(f'PRAGMA table_info("{table}")')
                cols = cur.fetchall()
                col_defs = ", ".join(
                    f"{c[1]} {c[2]}{'  PK' if c[5] else ''}" for c in cols
                )
                lines.append(f"  {table}({col_defs})")
            return lines
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []


def _fs_path_lines(fs_path: Path) -> list[str]:
    """Return a sorted list of all paths under the sandbox FS root."""
    if not fs_path.exists():
        return []
    try:
        lines: list[str] = []
        for p in sorted(fs_path.rglob("*")):
            rel = p.relative_to(fs_path).as_posix()
            suffix = "/" if p.is_dir() else ""
            lines.append(f"  {rel}{suffix}")
        return lines
    except Exception:  # noqa: BLE001
        return []


def _build_tool_definitions(sandbox: SandboxBoundary, tool_allowlist: list[str]) -> list[dict[str, Any]]:
    """Build OpenAI function-calling tool definitions enriched with the current
    sandbox state (DB schema + FS layout) so the model knows its environment.
    """
    tools: list[dict[str, Any]] = []

    if "sql" in tool_allowlist:
        schema_lines = _db_schema_lines(sandbox.db_path)
        desc = "Execute a single SQL statement against the sandbox database."
        if schema_lines:
            desc += "\n\nAvailable tables:\n" + "\n".join(schema_lines)
        tools.append({
            "type": "function",
            "function": {
                "name": "sql",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "A single SQL statement to execute (e.g. SELECT, INSERT, UPDATE).",
                        },
                    },
                    "required": ["input"],
                },
            },
        })

    if "file" in tool_allowlist:
        fs_lines = _fs_path_lines(sandbox.fs_path)
        read_desc = "Read the content of a file in the sandbox filesystem."
        if fs_lines:
            read_desc += "\n\nAvailable paths:\n" + "\n".join(fs_lines)
        tools.append({
            "type": "function",
            "function": {
                "name": "file_read",
                "description": read_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the sandbox filesystem.",
                        },
                    },
                    "required": ["path"],
                },
            },
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write or overwrite a file in the sandbox filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the sandbox filesystem.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        })

    return tools
