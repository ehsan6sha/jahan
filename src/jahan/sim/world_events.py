"""Tick phase 0 (scheduled world events, item expiry) and the news-driven parts of the mind.

* Scheduled events come from scenarios: news that starts later, and macro steps (exchange rates,
  gold, sentiment).  Large macro moves create automatic headlines in the affected countries.
* Economic sentiment is an exponential moving average of how good the news an agent knows is,
  weighted by each item's ``sentiment_weight``; it shifts purchase decisions.
* Claims about a product (e.g. a rumour) move agents' views of that product when they learn them.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np

from jahan.actors import engine
from jahan.model.enums import CHANNEL_KIND_NAMES, EventKind, OriginKind, Topic
from jahan.model.world import World
from jahan.scenario.inject import create_item, retire_item, start_news, start_social, start_statement
from jahan.scenario.spec import NewsEvent, SocialPostEvent, Statement
from jahan.sim.params import SimParams
from jahan.store.eventlog import EventLog

AUTO_NEWS_MIN_PCT = 2.0  # cumulative move since the last headline that makes news (HEURISTIC)
AUTO_NEWS_REFRESH_DAYS = 3
AUTO_NEWS_KINDS = {int(CHANNEL_KIND_NAMES[k]) for k in ("tv", "online", "social", "radio", "newspaper")}


def apply_due(world: World, log: EventLog) -> dict[str, int]:
    # messages, posts and next-day deliveries wait for the availability masks (actors.engine.deliver_actions)
    later_in_tick = (engine.KIND_DEFERRED, engine.KIND_DELIVER)
    due = [e for e in world.scheduled if e["day"] <= world.day and e["kind"] not in later_in_tick]
    world.scheduled = [e for e in world.scheduled if e["day"] > world.day or e["kind"] in later_in_tick]
    for e in due:
        if e["kind"] == "news":
            start_news(world, NewsEvent.model_validate(e["spec"]), log, int(e.get("prompt_id", -1)))
        elif e["kind"] == "statement":
            start_statement(world, Statement.model_validate(e["spec"]), log, int(e.get("prompt_id", -1)))
        elif e["kind"] == "social":
            start_social(world, SocialPostEvent.model_validate(e["spec"]), log, int(e.get("prompt_id", -1)))
        elif e["kind"] == "macro":
            apply_macro(world, e, log)
        elif e["kind"] == engine.KIND_LATER:
            engine.apply_now(world, engine.parse(e["spec"]), log, int(e.get("prompt_id", -1)))
        else:
            raise ValueError(f"unknown scheduled event kind {e['kind']!r}")
    freed = expire_items(world)
    return {"scheduled_events": len(due), "items_expired": freed}


def _targets(world: World, variable: str, country: str | None) -> list[int]:
    codes = [c.iso3 for c in world.countries]
    if country is not None:
        return [codes.index(country)]
    if variable == "fx_local_per_usd":
        return [i for i, c in enumerate(world.countries) if c.currency != "USD"]
    return list(range(len(codes)))


def _fx_headline(currency: str) -> Callable[[float], str]:
    def make(p: float) -> str:
        return f"{currency} {'falls' if p > 0 else 'rises'} {abs(p):.1f}% against the US dollar"

    return make


def _newsworthy(world: World, key: str, before: float, after: float) -> float | None:
    """Cumulative % move since the last headline on this key, if it reached the news threshold.

    Slow grinds become news once they add up (a 0.65%/day slide makes headlines every few days).
    """
    refs = world.macro.setdefault("news_ref", {})
    ref = float(refs.get(key, before))
    refs.setdefault(key, before)
    cum = 100.0 * (after / ref - 1.0)
    if abs(cum) < AUTO_NEWS_MIN_PCT:
        return None
    refs[key] = after
    return cum


def apply_macro(world: World, e: dict, log: EventLog) -> None:
    var, pct = e["variable"], float(e["pct"])
    targets = _targets(world, var, e.get("country"))
    if var == "fx_local_per_usd":
        fx = world.macro["fx"]
        for c in targets:
            before = fx[c]
            fx[c] *= 1.0 + pct / 100.0
            log.add(world.day, EventKind.MACRO_UPDATE, -1, c, -1, fx[c], c, "fx_local_per_usd")
            cum = _newsworthy(world, f"fx:{c}", before, fx[c]) if e.get("auto_news", True) else None
            if cum is not None:
                cur = world.countries[c].currency
                _auto_news(
                    world,
                    ("fx", c),
                    [c],
                    valence=-float(np.sign(cum)) * min(1.0, 0.3 + abs(cum) / 10),
                    pct=cum,
                    weight=1.0,
                    headline=_fx_headline(cur),
                    log=log,
                )
        world.refresh_usd_prices()
    elif var == "gold":
        before = world.macro["gold_index"]
        world.macro["gold_index"] *= 1.0 + pct / 100.0
        log.add(world.day, EventKind.MACRO_UPDATE, -1, -1, -1, world.macro["gold_index"], -1, "gold_index")
        cum = _newsworthy(world, "gold", before, world.macro["gold_index"]) if e.get("auto_news", True) else None
        if cum is not None:
            # rapid gold rises read as a sign of fear; sharp falls as relief (HEURISTIC sign)
            _auto_news(
                world,
                ("gold", -1),
                targets,
                valence=-0.5 * float(np.sign(cum)),
                pct=cum,
                weight=0.5,
                headline=lambda p: f"Gold price {'jumps' if p > 0 else 'drops'} {abs(p):.1f}%",
                log=log,
            )
    elif var == "sentiment":
        pop = world.pop
        for c in targets:
            m = pop["country"] == c
            pop["sentiment"][m] = np.clip(pop["sentiment"][m] + pct / 100.0, -1.0, 1.0)
            world.macro["sentiment_shift"][c] += pct / 100.0
            log.add(world.day, EventKind.MACRO_UPDATE, -1, c, -1, pct / 100.0, c, "sentiment")


def _auto_news(
    world: World,
    key: tuple[str, int],
    countries: list[int],
    valence: float,
    pct: float,
    weight: float,
    headline: str | Callable[[float], str],
    log: EventLog,
) -> None:
    """Create a macro headline, or refresh the running story if the move continues in the same direction.

    A refreshed story's headline reports the cumulative move of the whole story (logged as ITEM_UPDATED).
    """
    make = headline if callable(headline) else (lambda _p: headline)
    k = zlib.crc32(f"{key[0]}:{key[1]}".encode()) & 0x7FFFFFFF
    last = world.macro.setdefault("last_auto_news", {}).get(str(k))
    direction = int(np.sign(pct))
    if last is not None:
        item = int(last["item"])
        same_story = int(last.get("direction", direction)) == direction  # a reversal is a new story
        if (
            same_story
            and world.day - int(last["day"]) <= AUTO_NEWS_REFRESH_DAYS
            and int(world.items["slot"][item]) >= 0
        ):
            total = 100.0 * ((1 + float(last.get("story_pct", 0.0)) / 100.0) * (1 + pct / 100.0) - 1.0)
            salience = float(np.clip(0.3 + abs(total) / 10.0, 0.3, 0.95))
            world.items["salience"][item] = max(float(world.items["salience"][item]), salience)
            world.items["valence"][item] = float(np.sign(valence)) * min(1.0, 0.3 + abs(total) / 10)
            world.items["active_until"][item] = max(int(world.items["active_until"][item]), world.day + 30)
            pu = world.pushes
            m = pu["item_id"] == item
            pu["end_day"][m] = np.maximum(pu["end_day"][m], world.day + AUTO_NEWS_REFRESH_DAYS)
            last["day"] = world.day
            last["story_pct"] = total
            log.add(world.day, EventKind.ITEM_UPDATED, -1, -1, item, total, -1, make(total))
            world.macro.setdefault("item_labels", {})[str(item)] = make(total)
            return
    salience = float(np.clip(0.3 + abs(pct) / 10.0, 0.3, 0.95))
    headline = make(pct)
    item = create_item(
        world,
        topic=int(Topic.ECONOMY),
        valence=valence,
        truth=0.5 + 0.5 * valence,
        salience=salience,
        origin=OriginKind.MACRO,
        origin_id=-1,
        entity_id=-1,
        active_days=30,
        release_utc_hour=10,
        sentiment_weight=weight,
        label=headline,
        log=log,
        macro_key=k,
    )
    if item < 0:
        return
    ch = world.channels
    for cid in range(ch.n):
        if int(ch["country"][cid]) in countries and int(ch["kind"][cid]) in AUTO_NEWS_KINDS:
            world.pushes.append(
                1,
                item_id=item,
                channel_id=cid,
                start_day=world.day,
                end_day=world.day + AUTO_NEWS_REFRESH_DAYS,
                intensity=1.0,
            )
    world.macro["last_auto_news"][str(k)] = {"item": item, "day": world.day, "direction": direction, "story_pct": pct}


def expire_items(world: World) -> int:
    it = world.items
    if it.n == 0:
        return 0
    done = np.flatnonzero((it["slot"] >= 0) & (it["active_until"] < world.day))
    for i in done:
        retire_item(world, int(i))
    return int(done.size)


def update_sentiment(world: World, params: SimParams) -> None:
    """Mood drifts toward how good the agent's known news is (weighted, squashed to [-1, 1])."""
    bs = world.beliefs
    live = np.flatnonzero(bs.slot_item >= 0)
    pop = world.pop
    n = pop.n
    if live.size == 0:
        news = np.zeros(n)
    else:
        w = world.items["sentiment_weight"][bs.slot_item[live]].astype(np.float64)
        if not np.any(w):
            news = np.zeros(n)
        else:
            aware = bs.m["aware"][:n][:, live].astype(np.float64)
            belief = bs.m["belief"][:n][:, live].astype(np.float64)
            news = np.tanh(((belief - 0.5) * 2.0 * aware) @ w)
    s = pop["sentiment"].astype(np.float64)
    pop["sentiment"] = ((1 - params.sentiment_rate) * s + params.sentiment_rate * news).astype(np.float32)


def propagate_entity_claims(world: World, params: SimParams) -> int:
    """Agents who learned a claim about a product today update their view of that product."""
    it = world.items
    bs = world.beliefs
    n = world.pop.n
    moved = 0
    for slot in np.flatnonzero(bs.slot_item >= 0):
        item = int(bs.slot_item[slot])
        p = int(it["entity_id"][item])
        if p < 0 or int(world.products["item_id"][p]) == item:
            continue  # not a claim, or the product's own launch item
        p_item = int(world.products["item_id"][p])
        p_slot = int(it["slot"][p_item])
        if p_slot < 0:
            continue
        new = np.flatnonzero(bs.m["first_day"][:n, slot] == world.day)
        if new.size == 0:
            continue
        first = bs.m["aware"][new, p_slot] == 0
        bs.m["aware"][new[first], p_slot] = 1
        bs.m["first_day"][new[first], p_slot] = world.day
        bs.m["belief"][new[first], p_slot] = 0.5
        cur = bs.m["belief"][new, p_slot].astype(np.float64)
        claim = bs.m["belief"][new, slot].astype(np.float64)
        bs.m["belief"][new, p_slot] = (cur + params.k_claim * (claim - cur)).astype(np.float32)
        moved += int(new.size)
    return moved
