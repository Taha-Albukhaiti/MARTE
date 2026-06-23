from __future__ import annotations

from pathlib import Path, PurePosixPath

from marte.target.tools.tool_call import ToolCall


class FileTool:
    """Read/write files within the sandbox filesystem.

    All paths are resolved relative to *fs_path*.  Any attempt to escape the
    sandbox root via path traversal raises a ``ValueError`` and is recorded in
    the returned ``ToolCall`` as an error.
    """

    def __init__(self, fs_path: Path) -> None:
        self.fs_path = fs_path.resolve()

    def read(self, path: str) -> ToolCall:
        try:
            resolved = self._resolve(path)
            content = resolved.read_text(encoding="utf-8")
            return ToolCall(tool="file", operation="read", target=str(resolved.relative_to(self.fs_path)), raw_input=path, output=content)
        except Exception as exc:  # noqa: BLE001
            return ToolCall(tool="file", operation="read", target=path, raw_input=path, output="", error=str(exc))

    def write(self, path: str, content: str) -> ToolCall:
        try:
            resolved = self._resolve(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return ToolCall(tool="file", operation="write", target=str(resolved.relative_to(self.fs_path)), raw_input=path, output="ok")
        except Exception as exc:  # noqa: BLE001
            return ToolCall(tool="file", operation="write", target=path, raw_input=path, output="", error=str(exc))

    def _resolve(self, path: str) -> Path:
        p = PurePosixPath(path)

        # "/protected/secret.txt" -> "protected/secret.txt"
        if p.is_absolute():
            p = PurePosixPath(*p.parts[1:])

        if str(p) in {"", "."}:
            raise ValueError("Empty/invalid sandbox path")

        candidate = (self.fs_path / p).resolve()
        if not candidate.is_relative_to(self.fs_path):
            raise ValueError(f"Path traversal attempt blocked: {path!r}")

        return candidate

