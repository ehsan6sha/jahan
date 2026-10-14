"""Quick calibration harness for the product-launch dynamics.

Runs a world in memory: warm-up, inject the example scenario, simulate, and print
awareness/adoption curves and channel mix.  Parameter overrides: key=value args.

    uv run python scripts/calibrate_launch.py n=2000 share_b0=-5 adopt_b0=-9

Target bands (PROVISIONAL judgement, not sourced; to be replaced by empirical
launch data in M2): awareness 30-50% at day 30 and 50-80% at day 180; adult
adoption 2-8% at day 180; 20-50% of first exposures via people; 40-75% of
adopters last advised by a person.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> None:
    kv = dict(a.split("=", 1) for a in argv)
    n = int(kv.pop("n", 2000))
    seed = int(kv.pop("seed", 7))
    cfg = RunConfig(n_agents=n, countries=["USA"], seed=seed)
    for k, v in kv.items():
        setattr(cfg.params, k, float(v))
    world = generate_world(cfg)
    run_in_memory(world, 14)
    log = EventLog()
    apply_scenario(world, Scenario.load(ROOT / "examples/scenarios/launch_product_x.yaml"), log, 1)
    d0 = world.day
    rec = run_in_memory(world, 180)
    t = rec.metrics.table().to_pydict()
    series: dict[str, dict[int, float]] = {}
    for day, c, it, m, v in zip(t["day"], t["country"], t["item"], t["metric"], t["value"], strict=True):
        if it == 0 and c == 0:
            series.setdefault(m, {})[day - d0 + 1] = v
    for cp in (1, 7, 30, 90, 180):
        print(
            f"day {cp:>3}: aware {100 * series['aware_share'].get(cp, 0):5.1f}%  "
            f"adopted(adults) {100 * series['adopted_share_adults'].get(cp, 0):5.2f}%  "
            f"mean belief {series['mean_belief_aware'].get(cp, float('nan')):.2f}"
        )
    ev = rec.events.table().to_pydict()
    kind = np.array(ev["kind"])
    other = np.array(ev["other"])
    exp = kind == 7
    ad = kind == 8
    print(
        f"first exposures via people: {100 * (other[exp] >= 0).mean():.0f}%  "
        f"adopters advised by a person: {100 * (other[ad] >= 0).mean() if ad.any() else float('nan'):.0f}%"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
