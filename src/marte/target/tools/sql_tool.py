from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from marte.target.helpers.sql_tool import (
    _parse_multi_metadata,
    _parse_operation,
    _parse_target,
    _split_sql_tool_statements,
)
from marte.target.tools.tool_call import ToolCall

# ---------------------------------------------------------------------------
# SQLTool
# ---------------------------------------------------------------------------


class SQLTool:
    """Executes a single SQL statement against the sandbox database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def execute(self, query: str) -> ToolCall: # Check if multiple statements
        statements = _split_sql_tool_statements(query)
        is_multi = len(statements) > 1

        if is_multi:
            operation = "MULTI"
            ops, targets = _parse_multi_metadata(query)
            target = ",".join(targets)
        else:
            operation = _parse_operation(query)
            target = _parse_target(query, operation)

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.cursor()
                if is_multi:
                    conn.executescript(query)
                    op_text = ", ".join(ops) if ops else "UNKNOWN"
                    output = f"Executed multiple statements (ops: {op_text})"
                else:
                    cur.execute(query)
                    conn.commit()

                    if operation in {"SELECT", "PRAGMA"}:
                        rows = cur.fetchall()
                        output = json.dumps(rows, default=str)
                    else:
                        output = f"{cur.rowcount} row(s) affected"
            finally:
                conn.close()

            return ToolCall(tool="sql", operation=operation, target=target, raw_input=query, output=output)

        except Exception as exc:  # noqa: BLE001
            return ToolCall(tool="sql", operation=operation, target=target, raw_input=query, output="", error=str(exc))
