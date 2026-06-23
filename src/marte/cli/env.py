"""Environment and path helpers for the MARTE-thesis-worktree CLI."""

from __future__ import annotations

import os
from pathlib import Path

from marte.config.logging import setup_logging

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_LOADED = False


def _configure_logging(log_level: str) -> None:
    """Activate terminal logging for a MARTE-thesis-worktree run at the requested level."""
    setup_logging(log_level=log_level.upper(), console=True, json_format=False)


def _autoload_project_env() -> None:
    """Load .env from project root once without overriding process env."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        _DOTENV_LOADED = True
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        _load_env_file_without_dotenv(env_path)

    _DOTENV_LOADED = True


def _load_env_file_without_dotenv(env_path: Path) -> None:
    """Small fallback parser for KEY=VALUE lines when python-dotenv is absent."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_unquoted_comment(value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _strip_unquoted_comment(value: str) -> str:
    """Remove comments introduced by whitespace outside quoted values."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def _resolve_project_path(path: Path) -> Path:
    """Resolve relative paths against the project root."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (_PROJECT_ROOT / expanded).resolve()


def _seed_paths() -> tuple[Path, Path]:
    seed_sql_raw = _first_non_empty(
        os.getenv("MARTE_SEED_SQL_PATH"),
        "sandbox_data/sql/seed.sql",
    ) or "sandbox_data/sql/seed.sql"
    seed_fs_raw = _first_non_empty(
        os.getenv("MARTE_SEED_FS_PATH"),
        "sandbox_data/fs/seed",
    ) or "sandbox_data/fs/seed"

    return _resolve_project_path(Path(seed_sql_raw)), _resolve_project_path(Path(seed_fs_raw))


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
