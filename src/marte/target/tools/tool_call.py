from __future__ import annotations

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Record of a single tool invocation, used by the BreachOracle."""

    tool: str        # "sql" | "file"
    operation: str   # SELECT | PRAGMA | INSERT | UPDATE | DELETE | DDL  (sql)
                     # read | write                             (file)
    target: str      # table name (sql) or relative file path (file)
    raw_input: str   # full query or path argument passed by the agent
    output: str      # serialised result, or "" on error
    error: str | None = None
