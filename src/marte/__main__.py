"""Entry point for ``python -m marte``.

Delegates to the top-level ``main.py`` CLI so that both invocation styles work:

    python main.py --mode api --runs-per-condition 30
    python -m marte  --mode api --runs-per-condition 30
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path so that ``import main`` works
# regardless of how the package was installed.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
