from __future__ import annotations

import os
from pathlib import Path


DEFAULT_RUNS_DIR = Path.home() / "Documents" / "autoresearch-runs"


def runs_dir() -> Path:
    """Return the durable run directory used by the runner and dashboard."""
    configured = os.getenv("AUTORESEARCH_RUNS_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_RUNS_DIR
