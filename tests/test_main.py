from __future__ import annotations

import json
import math
from pathlib import Path

from main import _json_sanitize, _path_fingerprint


def test_json_sanitize_replaces_nan_and_infinity_with_none() -> None:
    payload = {
        "finite": 0.5,
        "nan": float("nan"),
        "pos_inf": float("inf"),
        "nested": [{"neg_inf": -math.inf}],
    }

    sanitized = _json_sanitize(payload)

    assert sanitized == {
        "finite": 0.5,
        "nan": None,
        "pos_inf": None,
        "nested": [{"neg_inf": None}],
    }
    json.dumps(sanitized, allow_nan=False)


def test_path_fingerprint_changes_with_seed_content(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    seed_file = seed_dir / "fixture.txt"
    seed_file.write_text("first", encoding="utf-8")
    first = _path_fingerprint(seed_dir)

    seed_file.write_text("second", encoding="utf-8")

    assert _path_fingerprint(seed_dir) != first
