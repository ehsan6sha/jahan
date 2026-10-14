"""Directional trust: Beta posterior over a halo-adjusted prior.

trust = (alpha + K0 * p) / (alpha + beta + K0),  p = sigmoid(logit(prior_p) + halo * exp(-n_out / H))

Evidence (alpha, beta) decays toward the prior lazily, only when a new outcome
arrives: dormant ties keep their baseline trust.  Negative outcomes weigh more
than positive ones (negativity bias).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jahan.gen.sampler import logit, sigmoid
from jahan.model.country import CAMP_TAU
from jahan.model.enums import ValueDim
from jahan.model.world import World
from jahan.sim.params import SimParams

_VALUE_KEYS = [e.name.lower() for e in ValueDim]


def effective_trust(world: World, rows: NDArray[np.int64] | slice, params: SimParams) -> NDArray[np.float64]:
    e = world.edges
    p = sigmoid(
        logit(e["prior_p"][rows].astype(np.float64))
        + e["halo"][rows].astype(np.float64) * np.exp(-e["n_out"][rows] / params.halo_fade_outcomes)
    )
    a = e["alpha"][rows].astype(np.float64)
    b = e["beta"][rows].astype(np.float64)
    return (a + params.trust_k0 * p) / (a + b + params.trust_k0)


def apply_outcomes(
    world: World,
    agents: NDArray[np.int64],
    advisers: NDArray[np.int64],
    advised_val: NDArray[np.float64],
    satisfaction: NDArray[np.float64],
    params: SimParams,
) -> int:
    """Update agent->adviser trust from realised outcomes.  Returns the number of ties updated.

    Advisers who are notable public figures update the agent's parasocial trust instead of an edge.
    """
    m = advisers >= 0
    if not m.any():
        return 0
    notable = np.zeros(m.shape[0], dtype=bool)
    if world.notables:
        notable[m] = world.pop["is_notable"][advisers[m]]
    updated = 0
    if notable.any():
        updated += _apply_parasocial(
            world, agents[notable], advisers[notable], advised_val[notable], satisfaction[notable], params
        )
    m &= ~notable
    if not m.any():
        return updated
    rows = world.find_edges(agents[m], advisers[m])
    ok = rows >= 0
    rows = rows[ok]
    if rows.size == 0:
        return 0
    val = advised_val[m][ok]
    s = satisfaction[m][ok]
    good = (val - 0.5) * (s - 0.5) > 0
    mag = np.abs(s - 0.5) * 2.0
    e = world.edges
    uniq = np.unique(rows)
    last = e["last_update_day"][uniq]
    dt = np.where(last >= 0, world.day - last, 0)
    decay = params.evidence_decay_per_day ** dt.astype(np.float64)
    e["alpha"][uniq] = (e["alpha"][uniq] * decay).astype(np.float32)
    e["beta"][uniq] = (e["beta"][uniq] * decay).astype(np.float32)
    alpha = e["alpha"]
    beta = e["beta"]
    np.add.at(alpha, rows, np.where(good, mag, 0.0).astype(np.float32))
    np.add.at(beta, rows, np.where(good, 0.0, mag * params.negativity_weight).astype(np.float32))
    n_out = e["n_out"]
    np.add.at(n_out, rows, 1)
    e["last_update_day"][uniq] = world.day
    return updated + int(uniq.size)


def _apply_parasocial(
    world: World,
    agents: NDArray[np.int64],
    notables: NDArray[np.int64],
    advised_val: NDArray[np.float64],
    satisfaction: NDArray[np.float64],
    params: SimParams,
) -> int:
    cols = world.pop["role_id"][notables].astype(np.int64)
    good = (advised_val - 0.5) * (satisfaction - 0.5) > 0
    mag = np.abs(satisfaction - 0.5) * 2.0
    pt = world.ptrust
    np.add.at(pt.alpha, (agents, cols), np.where(good, mag, 0.0).astype(np.float32))
    np.add.at(pt.beta, (agents, cols), np.where(good, 0.0, mag * params.negativity_weight).astype(np.float32))
    return int(agents.size)


def parasocial_prior(world: World, rows: NDArray[np.int64], col: int) -> NDArray[np.float64]:
    """Prior trust of agents in notable ``col``.

    Base: the figure's approval within the agent's political camp for residents of the figure's country
    (``camp_prior`` from model/notables.py::camp_priors, so the camp-weighted mean is the topline), else
    the topline approval.  Plus a value-similarity spread, centred on the agent's country mean so it
    averages out (HEURISTIC weight 1.5), minus a foreign-figure discount (HEURISTIC 0.8 on the logit).
    """
    entry = world.notables[col]
    j = int(entry["agent_id"])
    pop = world.pop
    home = pop["country"][rows].astype(np.int64)
    values = pop["values"][rows].astype(np.float64)
    mu = np.array([[s.values.mean[k] for k in _VALUE_KEYS] for s in world.countries], np.float64)
    spread = (values - mu[home]) @ pop["values"][j].astype(np.float64) * values.shape[1]
    foreign = home != int(entry["country"])
    base = np.full(home.shape[0], float(logit(float(entry["approval"]["value"]))))
    prior = entry.get("camp_prior")
    if prior and entry.get("camp_status") != "non_partisan" and abs(float(entry.get("alignment", 0.0))) >= CAMP_TAU:
        rel = pop["alignment"][rows].astype(np.float64) * (1.0 if float(entry["alignment"]) > 0 else -1.0)
        camp_p = np.where(
            rel > CAMP_TAU, prior["aligned"], np.where(rel < -CAMP_TAU, prior["opposed"], prior["neutral"])
        )
        base = np.where(foreign, base, logit(camp_p))
    return sigmoid(base + 1.5 * spread - 0.8 * foreign.astype(np.float64))


def parasocial_trust(world: World, rows: NDArray[np.int64], col: int, params: SimParams) -> NDArray[np.float64]:
    p = parasocial_prior(world, rows, col)
    a = world.ptrust.alpha[rows, col].astype(np.float64)
    b = world.ptrust.beta[rows, col].astype(np.float64)
    return (a + params.trust_k0 * p) / (a + b + params.trust_k0)
