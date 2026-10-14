"""Travel-lite: short trips between the simulated countries.

Trip propensity per country comes from World Bank international departures per person (latest
pre-pandemic year), scaled by the share of trips assumed to go to the other simulated countries
(``travel_in_world_share``; most real trips go to neighbours outside the eight-country world, which
are not modelled).  Richer adults travel more.  First-generation migrants visit their country of birth
more often.  While away, an agent's location and time zone switch, so they see the destination's
media in languages they speak and keep in touch with home only when hours overlap.
"""

from __future__ import annotations

import numpy as np

from jahan.model.enums import EventKind
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.sim.decisions import income_basis
from jahan.sim.params import SimParams
from jahan.store.eventlog import EventLog

DEFAULT_TRIPS_PER_PERSON = 0.02  # used when the World Bank series is missing (HEURISTIC)


def trips_per_person(world: World) -> np.ndarray:
    out = np.full(world.n_countries, DEFAULT_TRIPS_PER_PERSON)
    for c, spec in enumerate(world.countries):
        dep = spec.wdi.get("international_departures")
        tot = spec.wdi.get("population_total") or spec.population.value
        if dep and tot:
            out[c] = dep / tot
    return out


def travel(world: World, params: SimParams, log: EventLog) -> dict[str, int]:
    if world.n_countries < 2:
        return {"trips_started": 0, "trips_ended": 0}
    pop = world.pop
    day = world.day
    ids = pop["id"]
    # returns
    back = pop["alive"] & (pop["trip_end_day"] >= 0) & (pop["trip_end_day"] <= day)
    bi = np.flatnonzero(back)
    if bi.size:
        pop["location_country"][bi] = pop["country"][bi]
        pop["tz_offset_min"][bi] = pop["home_tz_offset_min"][bi]
        pop["trip_end_day"][bi] = -1
        log.add(day, EventKind.TRAVEL_END, ids[bi], -1, -1, 0.0, pop["country"][bi])
    # departures
    age = world.ages()
    elig = (
        pop["alive"]
        & ~pop["is_notable"]
        & (pop["controlled_by"] < 0)
        & (pop["trip_end_day"] < 0)
        & (age >= 18)
        & (age < 80)
    )
    if not elig.any():
        return {"trips_started": 0, "trips_ended": int(bi.size)}
    rate = trips_per_person(world)[pop["country"].astype(np.int64)] * params.travel_in_world_share
    basis = income_basis(world)
    rel_income = np.zeros(pop.n)
    for c in range(world.n_countries):
        m = elig & (pop["country"] == c)
        if m.any():
            med = float(np.median(basis[m])) or 1.0
            rel_income[m] = np.clip(basis[m] / med, 0.1, 5.0)
    migrant = pop["birth_country"] != pop["country"]
    p = rate / 365.25 * rel_income * np.where(migrant, 3.0, 1.0)  # HEURISTIC income and diaspora multipliers
    start = np.flatnonzero(elig & (u01(world.sim_seed, Stream.TRAVEL, day, ids) < p))
    if start.size == 0:
        return {"trips_started": 0, "trips_ended": int(bi.size)}
    weights = np.sqrt(np.array([s.population.value for s in world.countries], dtype=np.float64))
    main_tz = np.array([max(s.time_zones.rows, key=lambda r: r.share).offset_min for s in world.countries])
    u_dest = u01(world.sim_seed, Stream.TRAVEL, day, ids[start], k=1)
    u_home = u01(world.sim_seed, Stream.TRAVEL, day, ids[start], k=2)
    u_len = u01(world.sim_seed, Stream.TRAVEL, day, ids[start], k=3)
    for j, a in enumerate(start):
        home = int(pop["country"][a])
        if migrant[a] and u_home[j] < 0.6 and pop["birth_country"][a] >= 0:
            dest = int(pop["birth_country"][a])
        else:
            w = weights.copy()
            w[home] = 0.0
            cum = np.cumsum(w / w.sum())
            dest = int(np.searchsorted(cum, u_dest[j], side="right"))
            dest = min(dest, world.n_countries - 1)
        if dest == home:
            continue
        pop["location_country"][a] = dest
        pop["tz_offset_min"][a] = main_tz[dest]
        pop["trip_end_day"][a] = day + 5 + int(16 * u_len[j])  # 5-20 days
        log.add(day, EventKind.TRAVEL_START, int(ids[a]), dest, -1, float(pop["trip_end_day"][a] - day), home)
    return {"trips_started": int(start.size), "trips_ended": int(bi.size)}
