"""Performance budget: 10k agents x 365 days with a live product on a 4-core laptop.

Target < 60 s, hard fail > 180 s.  Run with ``uv run pytest -m perf``.
"""

import time
from pathlib import Path

import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

SCEN = Path(__file__).resolve().parents[2] / "examples" / "scenarios" / "launch_product_x.yaml"


@pytest.mark.perf
def test_ten_thousand_agents_one_year():
    world = generate_world(RunConfig(n_agents=10_000, countries=["USA"], seed=1, start_date="2026-10-13"))
    apply_scenario(world, Scenario.load(SCEN), EventLog(), 1)
    t0 = time.perf_counter()
    run_in_memory(world, 365)
    elapsed = time.perf_counter() - t0
    print(f"10k x 365: {elapsed:.1f}s")
    assert elapsed < 180
