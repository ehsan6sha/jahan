from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.world import World

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenarios" / "launch_product_x.yaml"


def small_config(n: int = 800, seed: int = 3) -> RunConfig:
    return RunConfig(n_agents=n, countries=["USA"], seed=seed, start_date="2026-10-13")


@pytest.fixture(scope="session")
def small_world_factory():
    def make(n: int = 800, seed: int = 3) -> World:
        return generate_world(small_config(n, seed))

    return make


def cli_env(env_extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["JAHAN_OFFLINE"] = "1"  # never reach a real model from tests, even if an API key is set
    # Plain text: Typer forces colour codes into its error boxes on GitHub Actions (and with FORCE_COLOR).
    for var in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
        env.pop(var, None)
    env["NO_COLOR"] = "1"
    env.update(env_extra or {})
    return env


def cli(*args: str, env_extra: dict[str, str] | None = None, cwd: Path = ROOT, stdin: str | None = None) -> str:
    """Run the CLI in a subprocess; ``stdin`` is fed as input (otherwise stdin is empty)."""
    env = cli_env(env_extra)
    res = subprocess.run(
        [sys.executable, "-m", "jahan.cli", *args],
        capture_output=True,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
        check=False,
    )
    if res.returncode != 0:
        raise AssertionError(f"jahan {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout
