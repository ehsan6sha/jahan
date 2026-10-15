"""Media exposure: channel pushes reach agents by tier, language, internet and awake hours."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import N_TOPICS, ChannelKind, EventKind, OriginKind
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.sim.availability import Masks, utc_hour_to_local, window_mask
from jahan.sim.params import SimParams
from jahan.sim.trust import parasocial_trust
from jahan.store.eventlog import EventLog

INTERNET_KINDS = {int(ChannelKind.ONLINE), int(ChannelKind.SOCIAL)}
SOCIAL_FEED_ORIGINS = {int(OriginKind.SOCIAL), int(OriginKind.NOTABLE)}


def conveyed_value(valence: float, credibility: float) -> float:
    """Belief value a source conveys: 0.5 neutral, toward 1 for positive claims."""
    return float(np.clip(0.5 + 0.5 * valence * credibility, 0.0, 1.0))


def social_feed_weight(
    world: World, item: int, cred: NDArray[np.float64] | float, origin: int, params: SimParams
) -> NDArray[np.float64]:
    """How much more (or less) likely each person is to see a post in their social feed.

    Interest: people see more posts on topics they care about (mean-preserving, since interests average
    1/N_TOPICS).  Follow affinity: people see more of a public figure they trust (``cred`` is their trust
    in the author).  Applied only to social-channel pushes of real posts and public figures' posts, so
    news and product launches on social channels are unchanged.
    """
    pop = world.pop
    topic = int(world.items["topic"][item])
    interest = pop["interests"][:, topic].astype(np.float64) * N_TOPICS
    w = np.clip(1.0 + params.social_interest_weight * (interest - 1.0), 0.25, 3.0)
    if origin == OriginKind.NOTABLE and params.social_affinity_base > 0:
        w = w * np.clip(params.social_affinity_base + np.asarray(cred, dtype=np.float64), 0.5, 1.5)
    return np.asarray(w, dtype=np.float64)


def media_exposure(world: World, masks: Masks, params: SimParams, log: EventLog) -> dict[str, int]:
    day = world.day
    pu = world.pushes
    if pu.n == 0:
        return {"media_exposures": 0}
    active = np.flatnonzero((pu["start_day"] <= day) & (day < pu["end_day"]))
    pop = world.pop
    alive = pop["alive"]
    ids = pop["id"]
    bel = world.beliefs.m
    exposures = 0
    for pi in active:
        item = int(pu["item_id"][pi])
        ch = int(pu["channel_id"][pi])
        slot = int(world.items["slot"][item])
        if slot < 0:
            continue  # ambient items are not tracked per agent in M1
        ch_country = int(world.channels["country"][ch])
        ch_lang = int(world.channels["lang"][ch])
        kind = int(world.channels["kind"][ch])
        elig = alive & (pop["location_country"] == ch_country) & (pop["lang"] == ch_lang).any(axis=1)
        if kind in INTERNET_KINDS:
            elig &= pop["internet"]
        hours = np.uint32(world.channels["hours_mask_local"][ch])
        window = masks.awake_local & hours
        if day == int(world.items["created_day"][item]):
            rel = utc_hour_to_local(int(world.items["release_utc_hour"][item]), pop["tz_offset_min"])
            window &= window_mask(rel, np.full(pop.n, 24, np.int64))
        elig &= window != 0
        reach = world.channels["reach_by_tier"][ch][pop["info_tier"].astype(np.int64)]
        attention = params.media_attention * float(world.items["salience"][item]) ** params.media_salience_power
        p = reach * params.media_use_days_per_week * attention * pu["intensity"][pi]
        origin = int(world.items["origin_kind"][item])
        source = -(ch + 2)
        if origin == OriginKind.NOTABLE and world.notables:
            # a public figure's statement: believed according to each agent's trust in that figure
            speaker = int(world.items["origin_id"][item])
            col = int(pop["role_id"][world.agent_row(speaker)])
            cred: NDArray[np.float64] | float = parasocial_trust(world, np.arange(pop.n), col, params)
            source = speaker
        else:
            cred = float(world.channels["credibility"][ch])
        if kind == ChannelKind.SOCIAL and origin in SOCIAL_FEED_ORIGINS:
            p = p * social_feed_weight(world, item, cred, origin, params)
        draw = u01(world.sim_seed, Stream.MEDIA, day, ids, k=item * 4096 + ch)
        hit = elig & (draw < p)
        if not hit.any():
            continue
        exposures += int(hit.sum())
        valence = float(world.items["valence"][item])
        val = np.clip(0.5 + 0.5 * valence * np.asarray(cred, dtype=np.float64), 0.0, 1.0)
        val = np.broadcast_to(val, (pop.n,))
        truth = float(world.items["truth"][item])
        disc = pop["discernment"].astype(np.float64)
        target = (
            val
            if np.isnan(truth)
            else val * (1 - params.discernment_pull * disc) + params.discernment_pull * disc * truth
        )
        cred_v = np.broadcast_to(np.asarray(cred, dtype=np.float64), (pop.n,))
        first = hit & (bel["aware"][: pop.n, slot] == 0)
        again = hit & ~first
        b = bel["belief"][: pop.n, slot]
        b[first] = target[first]
        b[again] += (params.k_media * cred_v[again] * (target[again] - b[again])).astype(np.float32)
        bel["aware"][: pop.n, slot][first] = 1
        bel["first_day"][: pop.n, slot][first] = day
        no_adviser = first & (bel["heard_from"][: pop.n, slot] < 0)
        bel["heard_from"][: pop.n, slot][no_adviser] = source
        bel["heard_val"][: pop.n, slot][no_adviser] = val[no_adviser]
        fi = np.flatnonzero(first)
        log.add(day, EventKind.INFO_EXPOSED, ids[fi], source, item, val[fi], pop["location_country"][fi])
    return {"media_exposures": exposures}
