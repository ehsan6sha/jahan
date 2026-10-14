"""Compare simulated vital rates with the country inputs over one year.

uv run python scripts/vital_check.py n=10000 seed=1 days=365
"""

from __future__ import annotations

import sys
import time

import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EmpStatus, EventKind
from jahan.sim.runner import run_in_memory


def main(argv: list[str]) -> None:
    kv = dict(a.split("=", 1) for a in argv)
    n, seed, days = int(kv.get("n", 10000)), int(kv.get("seed", 1)), int(kv.get("days", 365))
    countries = kv.get("countries", "USA").split(",")
    world = generate_world(RunConfig(n_agents=n, countries=countries, seed=seed, start_date="2026-10-13"))
    t0 = time.perf_counter()
    rec = run_in_memory(world, days)
    dt_ = time.perf_counter() - t0
    ev = rec.events.table().to_pydict()
    kind = np.array(ev["kind"])
    country = np.array(ev["country"])
    pop = world.pop
    print(f"{n} agents, {days} days in {dt_:.1f}s ({1000 * dt_ / days:.1f} ms/day); final alive {pop['alive'].sum()}")
    for c, spec in enumerate(world.countries):
        start_pop = int((pop["country"] == c).sum() - ((kind == EventKind.BORN) & (country == c)).sum())
        scale = 1000.0 / start_pop * 365.25 / days
        ind = spec.indicators
        rows = [
            ("births /1000", EventKind.BORN, ind.get("crude_birth_rate_per_1000")),
            ("deaths /1000", EventKind.DIED, ind.get("crude_death_rate_per_1000")),
            ("marriages /1000", EventKind.MARRIED, spec.vital.marriage_rate_per_1000 if spec.vital else None),
            ("divorces /1000", EventKind.DIVORCED, spec.vital.divorce_rate_per_1000 if spec.vital else None),
        ]
        print(f"[{spec.iso3}]")
        for label, k, target in rows:
            obs = ((kind == k) & (country == c)).sum() * scale
            print(f"  {label:<16} simulated {obs:6.2f}   input {target if target is not None else 'n/a'}")
        alive_c = pop["alive"] & (pop["country"] == c)
        age = world.ages()
        lf = alive_c & (age >= 25) & (age < 55)
        emp = pop["emp_status"][lf]
        sizes = np.bincount(pop["household_id"][alive_c])
        print(
            f"  employed 25-54   {np.mean(emp == EmpStatus.EMPLOYED):.3f}; "
            f"unemployed 25-54 {np.mean(emp == EmpStatus.UNEMPLOYED):.3f}; "
            f"mean household size {sizes[sizes > 0].mean():.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
