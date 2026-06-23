from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

_PROMPT_PACKAGE = "marte.prompts"


@cache
def load_prompt(relative_path: str) -> str:
    """Load a prompt resource from ``src/marte/prompts``."""
    resource = resources.files(_PROMPT_PACKAGE).joinpath(*relative_path.split("/"))
    return resource.read_text(encoding="utf-8").strip()


@cache
def load_prompt_json(relative_path: str) -> dict[str, Any]:
    """Load structured prompt variants from ``src/marte/prompts``."""
    resource = resources.files(_PROMPT_PACKAGE).joinpath(*relative_path.split("/"))
    return json.loads(resource.read_text(encoding="utf-8"))
