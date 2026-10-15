"""Watched questions: event odds from the simulated people's beliefs (tick phases 0c and 4c, and a metric).

A question ("Will X happen by date Y?") is a tracked item in a belief slot; each person's belief in that slot
is their probability that the answer is Yes.  Traders, a deterministic draw of internet-using adults
(``data/global/odds.yaml``, copied into ``world.macro["odds"]`` when the first question is watched), start
aware of it with beliefs spread around the real market price; everyone else may hear of it by word of mouth.

The simulated odds are the population-weighted mean belief of the traders who know the question (Wolfers and
Zitzewitz 2006: with log utility and beliefs unrelated to wealth a prediction market's price equals the mean
belief).  It is a belief average, not a market: no order book, no liquidity, no time decay toward the deadline,
and no pull toward the real market's price.

News moves beliefs through "claims": a (news item, question) pair with a direction in -1..1 (+1 makes Yes more
likely), scripted in a prompt (``bears_on``) or rated by Claude outside the core (recorded for replay).  A
person who knows the question and learns the item shifts their belief by k_odds x direction on the log-odds
scale.  Everything here reads and writes only ``world.macro["odds"]`` and the belief matrices.
"""

from __future__ import annotations

import datetime as dt
import zlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EventKind, Sex
from jahan.model.world import World
from jahan.rng import Stream, normal, u01
from jahan.sim.params import SimParams
from jahan.store.eventlog import EventLog, MetricsLog

MAX_OPEN_QUESTIONS = 8  # HEURISTIC: each open question holds one tracked-item slot until it closes
P_MIN, P_MAX = 0.01, 0.99  # beliefs stay inside this range (log-odds updates never reach 0 or 1)


def state(world: World) -> dict[str, Any] | None:
    return world.macro.get("odds")


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.log(p / (1.0 - p)), dtype=np.float64)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _skew(world: World, st: dict[str, Any]) -> NDArray[np.float64]:
    pop = world.pop
    young = world.ages() < 35
    male = pop["sex"] == int(Sex.M)
    sk = st["skew"]
    return np.select(
        [male & young, male & ~young, ~male & young],
        [sk["male_under_35"], sk["male_35_plus"], sk["female_under_35"]],
        sk["female_35_plus"],
    ).astype(np.float64)


def _eligible(world: World, st: dict[str, Any]) -> NDArray[np.bool_]:
    pop = world.pop
    return (
        pop["alive"]
        & pop["internet"]
        & (world.ages() >= float(st["adult_age"]))
        & ~pop["is_notable"]
        & (pop["pop_weight"] > 0)
    )


def ensure_state(
    world: World, trader_share: float, skew: dict[str, float], adult_age: float, sources: list[str]
) -> dict[str, Any]:
    """Create ``world.macro["odds"]`` on the first watched question (the trader draw is fixed from then on)."""
    st = state(world)
    if st is not None:
        return st
    st = {
        "trader_share": float(trader_share),
        "adult_age": float(adult_age),
        "skew": {k: float(v) for k, v in skew.items()},
        "sources": list(sources),
        "trader_norm": [],
        "questions": {},
    }
    elig = _eligible(world, st)
    sk = _skew(world, st)
    w = world.pop["pop_weight"].astype(np.float64)
    norm = []
    for c in range(world.n_countries):
        m = elig & (world.pop["country"] == c)
        norm.append(float(np.average(sk[m], weights=w[m])) if w[m].sum() > 0 else 1.0)
    st["trader_norm"] = norm
    world.macro["odds"] = st
    return st


def trader_mask(world: World, countries: list[int] | None = None) -> NDArray[np.bool_]:
    """Traders: a fixed draw of internet-using adults, a share of them in each country, skewed by age and sex."""
    st = state(world)
    pop = world.pop
    if st is None:
        return np.zeros(pop.n, bool)
    norm = np.asarray(st["trader_norm"], dtype=np.float64)
    home = pop["country"].astype(np.int64)
    p = float(st["trader_share"]) * _skew(world, st) / norm[home]
    u = u01(world.sim_seed, Stream.ODDS, 0, pop["id"], k=0)
    mask = _eligible(world, st) & (u < p)
    if countries is not None:
        mask &= np.isin(home, countries)
    return mask


def open_questions(world: World) -> list[tuple[str, dict[str, Any]]]:
    st = state(world)
    if st is None:
        return []
    return [(qid, q) for qid, q in st["questions"].items() if q["status"] == "open"]


def question_items(world: World) -> set[int]:
    st = state(world)
    return set() if st is None else {int(q["item"]) for q in st["questions"].values()}


def _countries(world: World, q: dict[str, Any]) -> list[int] | None:
    if not q.get("countries"):
        return None
    codes = [c.iso3 for c in world.countries]
    return [codes.index(c) for c in q["countries"] if c in codes]


def seed_question(world: World, qid: str, params: SimParams) -> int:
    """Traders learn the question with beliefs spread around its starting price.  Returns how many."""
    q = world.macro["odds"]["questions"][qid]
    slot = int(world.items["slot"][int(q["item"])])
    traders = trader_mask(world, _countries(world, q))
    rows = np.flatnonzero(traders)
    if rows.size == 0 or slot < 0:
        return 0
    pop = world.pop
    p0 = float(np.clip(q["prior"], 0.02, 0.98))
    k = 1 + zlib.crc32(qid.encode()) % (1 << 30)
    z = normal(world.sim_seed, Stream.ODDS, int(q["added_day"]), pop["id"][rows], k=k)
    lo = np.log(p0 / (1.0 - p0)) + params.odds_initial_sd * z
    lo += params.odds_partisan_shift * float(q.get("partisan", 0.0)) * pop["alignment"][rows].astype(np.float64)
    # centre the spread so the traders' average belief starts at the market price (spreading on the log-odds
    # scale alone would pull it toward 0.5)
    w = pop["pop_weight"][rows].astype(np.float64)
    lo_c, hi_c = -6.0, 6.0
    for _ in range(50):
        mid = 0.5 * (lo_c + hi_c)
        if float(np.average(np.clip(_sigmoid(lo + mid), P_MIN, P_MAX), weights=w)) < p0:
            lo_c = mid
        else:
            hi_c = mid
    b = np.clip(_sigmoid(lo + 0.5 * (lo_c + hi_c)), P_MIN, P_MAX)
    bel = world.beliefs.m
    bel["aware"][rows, slot] = 1
    bel["first_day"][rows, slot] = world.day
    bel["belief"][rows, slot] = b.astype(np.float32)
    bel["heard_val"][rows, slot] = b.astype(np.float32)
    return int(rows.size)


def apply_bearings(world: World, params: SimParams, log: EventLog) -> int:
    """Phase 4c: people who know a question and learn a news item that bears on it update their belief.

    The first time a claim is applied it reaches everyone who already knows both; afterwards only those who
    learned the item today.  Returns how many beliefs moved."""
    moved = 0
    if params.k_odds == 0:
        return 0
    day = world.day
    n = world.pop.n
    bel = world.beliefs.m
    w = world.pop["pop_weight"].astype(np.float64)
    for _qid, q in open_questions(world):
        qslot = int(world.items["slot"][int(q["item"])])
        if qslot < 0:
            continue
        aware_q = bel["aware"][:n, qslot] == 1
        tm: NDArray[np.bool_] | None = None
        for key, claim in sorted(q["claims"].items(), key=lambda kv: int(kv[0])):
            first = claim.get("applied_day") is None
            if first:
                claim["applied_day"] = day
            d = float(claim["direction"])
            item = int(key)
            if d == 0.0 or item >= world.items.n:
                continue
            s_item = int(world.items["slot"][item])
            if s_item < 0:
                continue
            knows = bel["aware"][:n, s_item] == 1
            fd = bel["first_day"][:n, s_item]
            learners = aware_q & knows & ((fd <= day) if first else (fd == day))
            if not learners.any():
                continue
            if tm is None:
                tm = trader_mask(world, _countries(world, q))
            b = np.clip(bel["belief"][:n, qslot][learners].astype(np.float64), P_MIN, P_MAX)
            b2 = np.clip(_sigmoid(_logit(b) + params.k_odds * d), P_MIN, P_MAX)
            denom = float((w * (tm & aware_q)).sum())
            if denom > 0:
                claim["moved"] = (
                    float(claim.get("moved", 0.0)) + float((w[learners] * tm[learners] * (b2 - b)).sum()) / denom
                )
            col = bel["belief"][:n, qslot]
            col[learners] = b2.astype(np.float32)
            moved += int(learners.sum())
    return moved


def current_odds(world: World) -> dict[str, dict[str, Any]]:
    """Per open question: the mean belief of traders who know it, how many they are, and per country."""
    out: dict[str, dict[str, Any]] = {}
    qs = open_questions(world)
    if not qs:
        return out
    n = world.pop.n
    bel = world.beliefs.m
    w = world.pop["pop_weight"].astype(np.float64)
    home = world.pop["country"]
    for qid, q in qs:
        slot = int(world.items["slot"][int(q["item"])])
        if slot < 0:
            continue
        m = trader_mask(world, _countries(world, q)) & (bel["aware"][:n, slot] == 1)
        b = bel["belief"][:n, slot].astype(np.float64)
        tot = float(w[m].sum())
        by_country = {}
        for c in range(world.n_countries):
            mc = m & (home == c)
            if w[mc].sum() > 0:
                by_country[world.countries[c].iso3] = float(np.average(b[mc], weights=w[mc]))
        out[qid] = {
            "odds": float(np.average(b[m], weights=w[m])) if tot > 0 else float("nan"),
            "n_traders": int(m.sum()),
            "represents": round(tot, 1),
            "by_country": by_country,
        }
    return out


def record_odds(world: World, out: MetricsLog) -> None:
    """Daily metrics: odds.<id> (country -1 and per country) and odds_n_traders.<id>, item = the question."""
    st = state(world)
    if st is None:
        return
    codes = [c.iso3 for c in world.countries]
    for qid, row in current_odds(world).items():
        item = int(st["questions"][qid]["item"])
        if row["n_traders"] == 0:
            continue
        out.add(world.day, -1, item, f"odds.{qid}", row["odds"])
        out.add(world.day, -1, item, f"odds_n_traders.{qid}", float(row["n_traders"]))
        for iso, v in row["by_country"].items():
            out.add(world.day, codes.index(iso), item, f"odds.{qid}", v)


def close_question(world: World, qid: str, log: EventLog, reason: str) -> None:
    """Stop following a question: freeze its odds and free its slot (what people believed is forgotten)."""
    st = state(world)
    assert st is not None
    q = st["questions"][qid]
    if q["status"] != "open":
        return
    now = current_odds(world).get(qid, {})
    item = int(q["item"])
    slot = int(world.items["slot"][item])
    if slot >= 0:
        world.beliefs.free_slot(slot)
        world.items["slot"][item] = -1
    world.items["active_until"][item] = min(int(world.items["active_until"][item]), world.day - 1)
    q.update(status="closed", closed_day=world.day, closed_reason=reason, final_odds=now.get("odds"))
    final = now.get("odds")
    log.add(world.day, EventKind.QUESTION_CLOSED, -1, -1, item, float("nan") if final is None else final, -1, reason)


def resolve_due(world: World, log: EventLog) -> dict[str, int]:
    """Phase 0c: questions whose resolve date has passed stop (Jahan does not know how they resolved)."""
    closed = 0
    today = world.date()
    for qid, q in open_questions(world):
        by = q.get("resolves_by")
        if by and today > dt.date.fromisoformat(str(by)):
            close_question(world, qid, log, "resolve date passed")
            closed += 1
    return {"questions_closed": closed} if closed else {}
