"""Phase 4b: ordinary people repost what they believe, which strengthens a story on social media.

A repost does not create a new item (that would use up the tracked-item slots).  It raises the story's push on
the social channels of the reposter's country, in the languages the reposter reads, from tomorrow on, by the
reposter's audience as a share of the country's online population, and it is logged (``SOCIAL_SHARED``) and
counted per item.  All coefficients are HEURISTIC (``SimParams.repost_*``); none is sourced.

Who may repost an item: alive, not a public figure, not controlled by a person, online, old enough (the same
ramp as word of mouth), aware of the item, convinced (``|belief - 0.5| * 2 >= repost_min_conviction``) and on
the item's side (belief above 0.5 for good or neutral news, below for bad).  Each (person, item) pair gets one
chance, on a day drawn once in the first ``repost_window_days`` after the person heard it, so no new state is
needed; on that day the person reposts with probability

    maturity * sigmoid(b0 + share_conviction * conviction + share_extraversion * E
                       + share_interest * (12 * interest[topic] - 1) + repost_visibility * (visibility - 1))

At most one repost per person per day (the one they care most about).  Caps: a push's intensity never exceeds 1,
it rises by at most ``repost_bump_cap`` per day, and reposts keep a story tracked at most ``repost_max_life_days``
after it was created.  Products are never reposted here (launch calibration is unchanged).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.gen.sampler import sigmoid
from jahan.model.enums import N_TOPICS, Big5, ChannelKind, EventKind
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.sim.params import SimParams
from jahan.sim.questions import question_items
from jahan.store.eventlog import EventLog

KEEP_DAYS = 60  # per-day repost counts kept in world.macro


def social_audience(world: World) -> NDArray[np.float64]:
    """Real people online per country (population weights of alive, online people located there)."""
    pop = world.pop
    online = pop["alive"] & pop["internet"]
    w = np.where(online, pop["pop_weight"].astype(np.float64), 0.0)
    return np.bincount(pop["location_country"].astype(np.int64), weights=w, minlength=world.n_countries)


def reposts(world: World, params: SimParams, log: EventLog) -> dict[str, int]:
    if not params.social_reposts or world.items.n == 0:
        return {"reposts": 0}
    day = world.day
    pop = world.pop
    n = pop.n
    bel = world.beliefs.m
    ids = pop["id"][:n]
    base = pop["alive"][:n] & ~pop["is_notable"][:n] & (pop["controlled_by"][:n] < 0) & pop["internet"][:n]
    age = world.ages()[:n]
    maturity = np.clip((age - params.share_min_age) / (params.share_full_age - params.share_min_age), 0.0, 1.0)
    base &= maturity > 0
    products = {int(x) for x in world.products["item_id"]} | question_items(world)  # questions are not stories
    extra = pop["big5"][:n, Big5.EXTRAVERSION].astype(np.float64)
    vis = pop["visibility"][:n].astype(np.float64)
    cand_rows: list[NDArray[np.int64]] = []
    cand_items: list[NDArray[np.int64]] = []
    cand_score: list[NDArray[np.float64]] = []
    for slot in np.flatnonzero(world.beliefs.slot_item >= 0):
        item = int(world.beliefs.slot_item[slot])
        if item in products or int(world.items["active_until"][item]) < day:
            continue
        first = bel["first_day"][:n, slot].astype(np.int64)
        aware = base & (bel["aware"][:n, slot] == 1) & (first >= 0)
        if not aware.any():
            continue
        window = first + np.floor(
            u01(world.sim_seed, Stream.SOCIAL, 0, ids, k=2 * item) * params.repost_window_days
        ).astype(np.int64)
        due = aware & (window == day)
        if not due.any():
            continue
        r = np.flatnonzero(due)
        b = bel["belief"][r, slot].astype(np.float64)
        side = 1.0 if float(world.items["valence"][item]) >= 0 else -1.0
        conviction = np.abs(b - 0.5) * 2
        ok = (conviction >= params.repost_min_conviction) & ((b - 0.5) * side > 0)
        r, conviction = r[ok], conviction[ok]
        if r.size == 0:
            continue
        interest = pop["interests"][r, int(world.items["topic"][item])].astype(np.float64) * N_TOPICS
        p = maturity[r] * sigmoid(
            params.repost_b0
            + params.share_conviction * conviction
            + params.share_extraversion * extra[r]
            + params.share_interest * (interest - 1.0)
            + params.repost_visibility * (vis[r] - 1.0)
        )
        hit = u01(world.sim_seed, Stream.SOCIAL, day, ids[r], k=2 * item + 1) < p
        if hit.any():
            cand_rows.append(r[hit])
            cand_items.append(np.full(int(hit.sum()), item, np.int64))
            cand_score.append(conviction[hit] * interest[hit])
    if not cand_rows:
        return {"reposts": 0}
    rows = np.concatenate(cand_rows)
    items = np.concatenate(cand_items)
    score = np.concatenate(cand_score)
    order = np.lexsort((items, -score, rows))  # per person: highest score first, ties by item id
    rows, items = rows[order], items[order]
    first_of = np.ones(rows.size, bool)
    first_of[1:] = rows[1:] != rows[:-1]
    rows, items = rows[first_of], items[first_of]  # one repost per person per day
    return _amplify(world, rows, items, params, log)


def _amplify(
    world: World, rows: NDArray[np.int64], items: NDArray[np.int64], params: SimParams, log: EventLog
) -> dict[str, int]:
    day = world.day
    pop = world.pop
    ch = world.channels
    audience = social_audience(world)
    social = [c for c in range(ch.n) if int(ch["kind"][c]) == ChannelKind.SOCIAL]
    loc = pop["location_country"][rows].astype(np.int64)
    delta = pop["pop_weight"][rows].astype(np.float64) * 10.0 ** pop["visibility"][rows].astype(np.float64)
    delta = delta / np.maximum(audience[loc], 1.0)
    bumps: dict[tuple[int, int], float] = {}
    for k in range(rows.size):
        langs = pop["lang"][rows[k]]
        chans = [c for c in social if int(ch["country"][c]) == loc[k] and int(ch["lang"][c]) in langs[langs >= 0]]
        if not chans:
            continue
        for c in chans:
            key = (int(items[k]), c)
            bumps[key] = bumps.get(key, 0.0) + float(delta[k])
        log.add(
            day,
            EventKind.SOCIAL_SHARED,
            int(pop["id"][rows[k]]),
            -(chans[0] + 2),
            int(items[k]),
            float(delta[k]),
            int(loc[k]),
        )
    pu = world.pushes
    tail_end = day + 1 + params.repost_tail_days
    for (item, c), total in sorted(bumps.items()):
        bump = min(total, params.repost_bump_cap)
        live = np.flatnonzero((pu["item_id"] == item) & (pu["channel_id"] == c) & (pu["end_day"] > day))
        start = day + 1
        if live.size:
            # a push still running gets stronger; the days after it ends carry only the reposts' own strength
            r = int(live[-1])
            pu["intensity"][r] = np.float32(min(1.0, float(pu["intensity"][r]) + bump))
            start = int(pu["end_day"][r])
        if start < tail_end:
            pu.append(1, item_id=item, channel_id=c, start_day=start, end_day=tail_end, intensity=min(1.0, bump))
        limit = int(world.items["created_day"][item]) + params.repost_max_life_days
        world.items["active_until"][item] = max(int(world.items["active_until"][item]), min(tail_end, limit))
    _count(world, rows, items)
    return {"reposts": int(rows.size)}


def _count(world: World, rows: NDArray[np.int64], items: NDArray[np.int64]) -> None:
    """Repost totals per item and per day in world.macro; folded into the post record the item belongs to."""
    st = world.macro.setdefault("social", {})
    w = world.pop["pop_weight"][rows].astype(np.float64)
    per_item: dict[str, dict[str, Any]] = st.setdefault("reposts", {})
    for item in np.unique(items):
        m = items == item
        rec = per_item.setdefault(str(int(item)), {"n": 0, "people": 0.0})
        rec["n"] += int(m.sum())
        rec["people"] = round(rec["people"] + float(w[m].sum()), 1)
    by_day: dict[str, int] = st.setdefault("reposts_by_day", {})
    by_day[str(world.day)] = by_day.get(str(world.day), 0) + int(rows.size)
    for key in [k for k in by_day if int(k) < world.day - KEEP_DAYS]:
        del by_day[key]
    for post in world.macro.get("posts", {}).values():
        r = per_item.get(str(post["item"]))
        if r is not None:
            post["reposts"], post["repost_people"] = r["n"], r["people"]
