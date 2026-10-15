"""Word-of-mouth diffusion of tracked items along contacted ties.

Edge src -> dst means src listens to dst, so on a contacted edge the sharer is
``dst`` and the receiver is ``src``; acceptance uses that edge's trust.

Sharing probability rises with the sharer's conviction, extraversion, interest
in the topic and own adoption.  Dishonest sharers exaggerate.  Receivers accept
within a bounded-confidence window, more readily from trusted sharers and when
undecided; well-informed receivers pull the conveyed value toward the truth.
Each receiver takes at most one accepted message per item per day (the most
trusted one; ties broken by edge uid).  Any share, accepted or not, makes the
receiver aware.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jahan.gen.sampler import sigmoid
from jahan.model.enums import N_TOPICS, Big5, EventKind
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.sim.params import SimParams
from jahan.sim.trust import effective_trust
from jahan.store.eventlog import EventLog


def _first_per_receiver(
    recv: NDArray[np.int64], trust: NDArray[np.float64], uid: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Indices (into the inputs) of the best message per receiver: highest trust, then lowest uid."""
    order = np.lexsort((uid, -trust, recv))
    r = recv[order]
    first = np.r_[True, r[1:] != r[:-1]]
    return order[first]


def acceptance(
    world: World,
    params: SimParams,
    r: NDArray[np.int64],
    conveyed: NDArray[np.float64],
    tr: NDArray[np.float64],
    b_r: NDArray[np.float64],
    first_aware: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Bounded-confidence window and acceptance probability for receivers ``r`` (shared with actor delivery)."""
    openness = world.pop["big5"][r, Big5.OPENNESS]
    eps = params.bounded_eps_base + params.bounded_eps_openness * sigmoid(openness)
    within = first_aware | (np.abs(conveyed - b_r) < eps)
    undecided = 1.0 - np.abs(b_r - 0.5) * 2
    p_acc = sigmoid(
        params.accept_b0
        + params.accept_trust * (tr - 0.5) * 2
        + params.accept_openness * openness
        + params.accept_undecided * undecided
    )
    return within, p_acc


def diffuse(
    world: World,
    contacted: NDArray[np.bool_],
    params: SimParams,
    log: EventLog,
    trace: set[int] | None = None,
    trace_rows: list[dict] | None = None,
) -> dict[str, int]:
    stats = {"shares": 0, "accepts": 0, "wom_aware": 0}
    idx = np.flatnonzero(contacted)
    if idx.size == 0:
        return stats
    e = world.edges
    pop = world.pop
    day = world.day
    # agents controlled by a person share only through their own actions, never autonomously
    autonomous = pop["controlled_by"][e["dst"][idx]] < 0
    idx = idx[autonomous]
    if idx.size == 0:
        return stats
    recv_all = e["src"][idx]
    shr_all = e["dst"][idx]
    uid_all = e["uid"][idx]
    trust_all = effective_trust(world, idx, params)
    bel = world.beliefs.m
    n = pop.n
    big5 = pop["big5"]
    age = world.ages()
    maturity = np.clip((age - params.share_min_age) / (params.share_full_age - params.share_min_age), 0.0, 1.0)
    for slot in np.flatnonzero(world.beliefs.slot_item >= 0):
        item = int(world.beliefs.slot_item[slot])
        if int(world.items["active_until"][item]) < day:
            continue
        aware = bel["aware"][:n, slot]
        cand = aware[shr_all] == 1
        if not cand.any():
            continue
        r = recv_all[cand]
        s = shr_all[cand]
        uid = uid_all[cand]
        tr = trust_all[cand]
        b_s = bel["belief"][s, slot].astype(np.float64)
        conviction = np.abs(b_s - 0.5) * 2
        topic = int(world.items["topic"][item])
        interest = pop["interests"][s, topic] * N_TOPICS
        adopted = bel["adopted_day"][s, slot] >= 0
        fresh_since = np.maximum(bel["first_day"][s, slot], bel["adopted_day"][s, slot])
        novelty = np.exp(-(day - fresh_since) / params.share_novelty_days)
        p_share = (
            maturity[s]
            * novelty
            * sigmoid(
                params.share_b0
                + params.share_conviction * conviction
                + params.share_extraversion * big5[s, Big5.EXTRAVERSION]
                + params.share_interest * (interest - 1.0)
                + params.share_adopted * adopted
            )
        )
        share = u01(world.sim_seed, Stream.SHARE, day, uid, k=item) < p_share
        if not share.any():
            continue
        r, s, uid, tr, b_s = r[share], s[share], uid[share], tr[share], b_s[share]
        stats["shares"] += int(r.shape[0])
        dishonest = u01(world.sim_seed, Stream.DISTORT, day, uid, k=item) > pop["honesty"][s]
        conveyed = np.where(dishonest, np.clip(0.5 + (b_s - 0.5) * params.distort_exaggeration, 0, 1), b_s)

        first_aware = aware[r] == 0
        b_r = np.where(first_aware, 0.5, bel["belief"][r, slot].astype(np.float64))
        within, p_acc = acceptance(world, params, r, conveyed, tr, b_r, first_aware)
        acc = within & (u01(world.sim_seed, Stream.ACCEPT, day, uid, k=item) < p_acc)

        # awareness from any share (first exposure event credits the most trusted sharer)
        fa = np.flatnonzero(first_aware)
        if fa.size:
            best = fa[_first_per_receiver(r[fa], tr[fa], uid[fa])]
            rr = r[best]
            aware[rr] = 1
            bel["first_day"][rr, slot] = day
            bel["belief"][rr, slot] = 0.5
            stats["wom_aware"] += int(rr.shape[0])
            log.add(day, EventKind.INFO_EXPOSED, rr, s[best], item, conveyed[best], pop["location_country"][rr])

        ai = np.flatnonzero(acc)
        if ai.size:
            best = ai[_first_per_receiver(r[ai], tr[ai], uid[ai])]
            rr = r[best]
            d = pop["discernment"][rr].astype(np.float64)
            truth = float(world.items["truth"][item])
            target = (
                conveyed[best]
                if np.isnan(truth)  # a watched question: no true answer to pull toward
                else conveyed[best] * (1 - params.discernment_pull * d) + params.discernment_pull * d * truth
            )
            cur = bel["belief"][rr, slot].astype(np.float64)
            bel["belief"][rr, slot] = (cur + params.k_wom * tr[best] * (target - cur)).astype(np.float32)
            bel["heard_from"][rr, slot] = s[best]
            bel["heard_val"][rr, slot] = conveyed[best].astype(np.float32)
            stats["accepts"] += int(rr.shape[0])
        if trace and trace_rows is not None:
            for j in np.flatnonzero(np.isin(r, list(trace)) | np.isin(s, list(trace))):
                trace_rows.append(
                    {
                        "type": "share",
                        "day": day,
                        "item": item,
                        "receiver": int(r[j]),
                        "sharer": int(s[j]),
                        "trust": float(tr[j]),
                        "sharer_belief": float(b_s[j]),
                        "conveyed": float(conveyed[j]),
                        "distorted": bool(dishonest[j]),
                        "receiver_belief_before": float(b_r[j]),
                        "within_window": bool(within[j]),
                        "accepted": bool(acc[j]),
                    }
                )
    return stats
