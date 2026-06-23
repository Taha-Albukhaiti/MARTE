from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from marte.target.guardrail import GuardrailDecision, GuardrailEngine
from marte.target.tool_call_factory import (
    _blocked_call,
    _decision_tag,
    _denied_call,
    _execution_error_call,
)
from marte.target.tools import FileTool, SQLTool, ToolCall


@dataclass
class DispatchResult:
    """Outcome of a single tool dispatch.

    The TargetAgent uses this to build StepResult.tool_trace and
    StepResult.policy_decisions.
    """

    tool_call: ToolCall
    policy_decision: GuardrailDecision


class ToolBroker:
    """Routes tool requests through GuardrailEngine, then to SQLTool or FileTool.

    Responsibilities:
    - Check guardrail policy before any tool execution
    - If denied: return a ToolCall recording the denial, without touching the sandbox
    - If allowed: delegate to the appropriate tool and return its ToolCall
    - Never raise; all outcomes (including policy violations) are captured in DispatchResult
    """

    def __init__(self, guardrail: GuardrailEngine, sql_tool: SQLTool | None = None, file_tool: FileTool | None = None) -> None:
        self.guardrail = guardrail
        self._sql = sql_tool
        self._file = file_tool

    def dispatch_sql(self, query: str) -> DispatchResult:
        decision = self.guardrail.check_tool_call("sql", query)
        executor: Callable[[], ToolCall] | None = None
        if self._sql is not None:
            executor = partial(self._sql.execute, query)
        return self._dispatch(tool="sql", denied_operation="DENIED", error_operation="ERROR", raw_input=query, decision=decision,
                              executor=executor, missing_reason="ToolBroker: SQLTool not configured")

    def dispatch_file_read(self, path: str) -> DispatchResult:
        decision = self.guardrail.check_tool_call("file", path, operation="read")
        executor: Callable[[], ToolCall] | None = None
        if self._file is not None:
            executor = partial(self._file.read, path)
        return self._dispatch(tool="file", denied_operation="read", error_operation="read", raw_input=path, decision=decision,
                              executor=executor, missing_reason="ToolBroker: FileTool not configured")

    def dispatch_file_write(self, path: str, content: str) -> DispatchResult:
        decision = self.guardrail.check_tool_call("file", path, operation="write")
        executor: Callable[[], ToolCall] | None = None
        if self._file is not None:
            executor = partial(self._file.write, path, content)
        return self._dispatch(tool="file", denied_operation="write", error_operation="write", raw_input=path, decision=decision,
                              executor=executor, missing_reason="ToolBroker: FileTool not configured")

    def _dispatch(self, *, tool: str, denied_operation: str, error_operation: str,
                  raw_input: str, decision: GuardrailDecision, executor: Callable[[], ToolCall] | None, missing_reason: str) -> DispatchResult:
        if decision.action in {"deny", "sanitize"}:
            return DispatchResult(
                tool_call=_blocked_call(tool=tool, operation=denied_operation, raw_input=raw_input, reason=decision.reason, tag=_decision_tag(decision.action)),
                policy_decision=decision,
            )

        if executor is None:
            missing = GuardrailDecision(action="deny", reason=missing_reason)
            return DispatchResult(
                tool_call=_denied_call(tool=tool, operation=denied_operation, raw_input=raw_input, reason=missing.reason),
                policy_decision=missing,
            )

        try:
            tool_call = executor()
        except Exception as exc:  # noqa: BLE001
            tool_call = _execution_error_call(tool, error_operation, raw_input, str(exc))

        return DispatchResult(tool_call=tool_call, policy_decision=decision)
