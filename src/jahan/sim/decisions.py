"""Product adoption decisions and realised outcomes."""

from __future__ import annotations

import numpy as np

from jahan.gen.sampler import sigmoid
from jahan.model.enums import N_TOPICS, N_VALUES, EventKind
from jahan.model.world import World
from jahan.rng import Stream, normal, u01
from jahan.sim.params import SimParams
from jahan.sim.trust import apply_outcomes
from jahan.store.eventlog import EventLog


def income_basis(world: World) -> np.ndarray:
    """Own income if positive, else household income per member (local currency / month)."""
    pop = world.pop
    own = pop["income_month"].astype(np.float64)
    hh = pop["household_id"]
    hh_inc = world.households["income_pooled"].astype(np.float64)
    hh_n = np.maximum(world.households["n_members"].astype(np.float64), 1)
    per_cap = np.where(hh >= 0, hh_inc[np.maximum(hh, 0)] / hh_n[np.maximum(hh, 0)], 0.0)
    return np.where(own > 0, own, per_cap)


def value_fit(values: np.ndarray, attributes: np.ndarray) -> np.ndarray:
    """0 when values are uniform; positive when the agent's priorities match the product."""
    a = attributes.astype(np.float64)
    s = a.sum()
    if s <= 0:
        return np.zeros(values.shape[0])
    return values.astype(np.float64) @ (a / s) * N_VALUES - 1.0


def decide_and_realise(world: World, params: SimParams, log: EventLog) -> dict[str, int]:
    stats = {"adoptions": 0, "outcomes": 0, "trust_updates": 0}
    pr = world.products
    if pr.n == 0:
        return stats
    pop = world.pop
    n = pop.n
    day = world.day
    ids = pop["id"]
    alive = pop["alive"]
    age = world.ages()
    bel = world.beliefs.m
    basis = np.maximum(income_basis(world), 1.0)
    loc = pop["location_country"].astype(np.int64)
    e = world.edges
    for p in range(pr.n):
        item = int(pr["item_id"][p])
        slot = int(world.items["slot"][item])
        if slot < 0:
            continue
        adopted_day = bel["adopted_day"][:n, slot]
        avail_from = world.product_available[p][loc]
        available = (avail_from >= 0) & (avail_from <= day)
        elig = alive & (bel["aware"][:n, slot] == 1) & (adopted_day < 0) & (age >= params.min_adopt_age) & available
        elig &= pop["controlled_by"] < 0  # a person decides for the agents they control
        if elig.any():
            afford = world.product_price[p][loc] / basis
            adopted_any = (adopted_day >= 0).astype(np.float64)
            act = e["active"]
            num = np.bincount(e["src"][act], weights=adopted_any[e["dst"][act]], minlength=n)
            deg = np.bincount(e["src"][act], minlength=n)
            peer = num / np.maximum(deg, 1)
            fit = value_fit(pop["values"], pr["attributes"][p])
            topic = int(pr["topic"][p])
            h = sigmoid(
                params.adopt_b0
                + params.adopt_belief * (bel["belief"][:n, slot] - 0.5) * 2
                - params.adopt_afford * np.minimum(afford, 3.0)
                + params.adopt_peer * peer
                + params.adopt_risk * (pop["risk_tolerance"] - 0.5)
                + params.adopt_fit * fit
                + params.adopt_interest * (pop["interests"][:, topic] * N_TOPICS - 1.0)
                + params.adopt_sentiment * pop["sentiment"]
            )
            adopt = elig & (u01(world.sim_seed, Stream.DECIDE, day, ids, k=p) < h)
            if adopt.any():
                ai = np.flatnonzero(adopt)
                adopted_day[ai] = day
                log.add(day, EventKind.ADOPTED, ids[ai], bel["heard_from"][ai, slot], item, afford[ai], loc[ai])
                stats["adoptions"] += int(ai.size)

        due = np.flatnonzero(alive & (adopted_day >= 0) & (adopted_day == day - params.outcome_lag_days))
        if due.size:
            fit_d = value_fit(pop["values"][due], pr["attributes"][p])
            noise = normal(world.sim_seed, Stream.OUTCOME, day, ids[due], k=p)
            s = np.clip(
                float(pr["true_quality"][p]) + params.outcome_fit_weight * fit_d + params.outcome_noise_sd * noise,
                0.0,
                1.0,
            )
            bel["satisfaction"][due, slot] = s.astype(np.float32)
            old = bel["belief"][due, slot].astype(np.float64)
            bel["belief"][due, slot] = (0.3 * old + 0.7 * s).astype(np.float32)
            advisers = bel["heard_from"][due, slot].astype(np.int64)
            log.add(day, EventKind.OUTCOME_REALIZED, ids[due], advisers, item, s, loc[due])
            stats["outcomes"] += int(due.size)
            stats["trust_updates"] += apply_outcomes(
                world, ids[due], advisers, bel["heard_val"][due, slot].astype(np.float64), s, params
            )
    return stats
