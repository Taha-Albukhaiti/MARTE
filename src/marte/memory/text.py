from __future__ import annotations

import re
from typing import Any


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
