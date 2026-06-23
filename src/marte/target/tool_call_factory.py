from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marte.target.tools.tool_call import ToolCall


def _denied_call(tool: str, operation: str, raw_input: str, reason: str) -> ToolCall:
    return _blocked_call(tool, operation, raw_input, reason, tag="DENIED")


def _blocked_call(tool: str, operation: str, raw_input: str, reason: str, tag: str) -> ToolCall:
    from marte.target.tools.tool_call import ToolCall

    return ToolCall(tool=tool, operation=operation, target="", raw_input=raw_input, output="", error=f"[{tag}] {reason}")


def _execution_error_call(tool: str, operation: str, raw_input: str, reason: str) -> ToolCall:
    from marte.target.tools.tool_call import ToolCall

    return ToolCall(tool=tool, operation=operation, target="", raw_input=raw_input, output="", error=f"[ERROR] {reason}")


def _decision_tag(action: str) -> str:
    return "SANITIZE" if action == "sanitize" else "DENIED"
