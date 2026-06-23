from __future__ import annotations

import json
import time
from typing import Any


def _parse_tool_call(response: str) -> dict[str, Any] | None:
    """Find and decode the first JSON object that contains a 'tool' key.

    Uses json.JSONDecoder.raw_decode() so it correctly handles nested objects
    and JSON embedded in surrounding prose.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(response):
        try:
            start = response.index("{", idx)
        except ValueError:
            break
        try:
            obj, _ = decoder.raw_decode(response, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj
        idx = start + 1
    return None


def _ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000
