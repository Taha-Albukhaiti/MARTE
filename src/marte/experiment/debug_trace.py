from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_touch_debug_file(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except Exception:
        pass


def safe_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")
    except Exception:
        pass
