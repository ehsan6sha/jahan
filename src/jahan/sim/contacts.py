"""Daily contact sampling per tie, gated by shared free (or work) hours."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EdgeKind, Marital
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.sim.availability import Masks, popcount
from jahan.sim.params import SimParams


def social_time_factor(world: World, params: SimParams) -> NDArray[np.float64]:
    """Share of 'outside' social time an agent has left after family obligations (1 = unconstrained)."""
    pop = world.pop
    f = np.where(pop["marital"] == Marital.MARRIED, params.married_social_time, 1.0)
    toddlers = pop["alive"] & (world.ages() < 5)
    parents = np.r_[pop["mother_id"][toddlers], pop["father_id"][toddlers]]
    parents = parents[parents >= 0]
    has_toddler = np.zeros(pop.n, dtype=bool)
    has_toddler[parents] = True
    return f * np.where(has_toddler, params.young_child_social_time, 1.0)


def sample_contacts(world: World, masks: Masks, params: SimParams) -> NDArray[np.bool_]:
    e = world.edges
    if e.n == 0:
        return np.zeros(0, dtype=bool)
    src = e["src"]
    dst = e["dst"]
    alive = world.pop["alive"]
    active = e["active"] & alive[src] & alive[dst]
    kind = e["kind"]
    work_tie = (kind == EdgeKind.COLLEAGUE) | (kind == EdgeKind.CLASSMATE)
    hh = kind == EdgeKind.HOUSEHOLD
    ov = np.where(
        hh,
        popcount(masks.awake_utc[src] & masks.awake_utc[dst]),
        np.where(
            work_tie,
            popcount(masks.work_utc[src] & masks.work_utc[dst]),
            popcount(masks.free_utc[src] & masks.free_utc[dst]),
        ),
    ).astype(np.float64)
    f = np.clip(ov / params.overlap_hours_full, 0.0, 1.0)
    p = e["strength"] * f
    # time allocation: marriage and young children take time away from ties outside the household
    social = social_time_factor(world, params)
    p = np.where(hh, p, p * np.sqrt(social[src] * social[dst]))
    u = u01(world.sim_seed, Stream.CONTACT, world.day, e["uid"] >> 1)  # same draw for both directions
    contacted = active & (u < p)
    e["last_contact_day"][contacted] = world.day
    return contacted
