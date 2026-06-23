from __future__ import annotations

import os
from pathlib import Path

from marte.cli.env import _load_env_file_without_dotenv


def test_fallback_env_loader_removes_unquoted_inline_comment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'NUMERIC_VALUE=60 # seconds\nQUOTED_VALUE="value # retained"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("NUMERIC_VALUE", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    _load_env_file_without_dotenv(env_path)

    assert os.environ["NUMERIC_VALUE"] == "60"
    assert os.environ["QUOTED_VALUE"] == "value # retained"
