"""Filesystem locations."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Directory holding registry.yaml, countries/, names/, global/.

    Resolution order: ``JAHAN_DATA`` env var, then the repository's ``data/`` folder.
    """
    env = os.environ.get("JAHAN_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"
