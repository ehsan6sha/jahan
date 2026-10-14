"""Apply a Scenario to a World at the current day (equivalent to tick phase 0).

Events that start later are stored in ``world.scheduled`` (JSON) and executed by
``sim.world_events.apply_due`` at the start of their day, so they survive snapshots and resets.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from jahan.actors.actions import ACTOR_TYPES
from jahan.model.enums import CHANNEL_KIND_NAMES, TOPIC_NAMES, VALUE_NAMES, EventKind, OriginKind
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.world import MAX_PRODUCTS, World
from jahan.scenario.spec import (
    DEFAULT_SENTIMENT_WEIGHT,
    MacroPath,
    NewsEvent,
    ProductLaunch,
    Scenario,
    Statement,
)
from jahan.store.eventlog import EventLog

PERMANENT = 10_000


class InjectError(ValueError):
    pass


def _country_index(world: World, iso3: str) -> int:
    codes = [c.iso3 for c in world.countries]
    if iso3 not in codes:
        raise InjectError(f"country {iso3} is not in this world ({codes})")
    return codes.index(iso3)


def _push(
    world: World,
    item_id: int,
    countries: list[int],
    kinds: set[int],
    start: int,
    days: int,
    intensity: float,
    lang: int | None = None,
) -> int:
    ch = world.channels
    n = 0
    for cid in range(ch.n):
        if lang is not None and int(ch["lang"][cid]) != lang:
            continue
        if int(ch["country"][cid]) in countries and int(ch["kind"][cid]) in kinds:
            world.pushes.append(
                1, item_id=item_id, channel_id=cid, start_day=start, end_day=start + days, intensity=intensity
            )
            n += 1
    return n


def create_item(
    world: World,
    *,
    topic: int,
    valence: float,
    truth: float,
    salience: float,
    origin: OriginKind,
    origin_id: int,
    entity_id: int,
    active_days: int,
    release_utc_hour: int,
    sentiment_weight: float,
    label: str,
    log: EventLog,
    macro_key: int = -1,
) -> int:
    """Create a tracked info item (allocates a belief slot).  Returns the item id or -1 if no slot is free."""
    item_id = world.items.n
    slot = world.beliefs.alloc_slot(item_id)
    if slot < 0:
        log.add(world.day, EventKind.ITEM_CREATED, -1, -1, -1, -1.0, -1, f"no free slot: {label}")
        return -1
    world.items.append(
        1,
        id=item_id,
        created_day=world.day,
        release_utc_hour=release_utc_hour,
        topic=topic,
        valence=valence,
        truth=truth,
        salience=salience,
        credibility=0.5,
        origin_kind=int(origin),
        origin_id=origin_id,
        entity_id=entity_id,
        lang=-1,
        active_until=world.day + active_days,
        slot=slot,
        content_hash=hashlib.sha256(label.encode()).digest()[:16],
        sentiment_weight=sentiment_weight,
        macro_key=macro_key,
    )
    log.add(world.day, EventKind.ITEM_CREATED, -1, -1, item_id, float(slot), -1, label[:300])
    world.macro.setdefault("item_labels", {})[str(item_id)] = label[:300]
    return item_id


def _launch(world: World, ev: ProductLaunch, log: EventLog, prompt_id: int) -> int:
    day = world.day
    if world.products.n >= MAX_PRODUCTS:
        raise InjectError("product capacity reached")
    world.ensure_product_arrays()
    pid = world.products.n
    attrs = np.zeros(len(VALUE_NAMES), np.float32)
    for k, v in ev.attributes.items():
        attrs[VALUE_NAMES[k]] = v
    topic = int(TOPIC_NAMES[ev.topic])
    item_id = create_item(
        world,
        topic=topic,
        valence=ev.valence,
        truth=ev.true_quality,
        salience=ev.salience,
        origin=OriginKind.SCENARIO,
        origin_id=prompt_id,
        entity_id=pid,
        active_days=PERMANENT,
        release_utc_hour=ev.release_utc_hour,
        sentiment_weight=0.0,
        label=ev.name,
        log=log,
    )
    if item_id < 0:
        raise InjectError("no free tracked-item slot; raise max_slots in the run config")
    world.products.append(1, id=pid, topic=topic, attributes=attrs, true_quality=ev.true_quality, item_id=item_id)
    world.product_names.append(ev.name)
    if ev.owner_agent >= 0:
        world.macro.setdefault("product_owner", {})[ev.name] = ev.owner_agent
    countries = [_country_index(world, c) for c in ev.countries]
    fx = world.macro.get("fx") or [c.fx_local_per_usd for c in world.countries]
    for c in countries:
        spec = world.countries[c]
        if spec.iso3 in ev.price_local:
            world.product_price[pid, c] = ev.price_local[spec.iso3]
        elif ev.price_usd is not None:
            world.product_price_usd[pid, c] = ev.price_usd
            world.product_price[pid, c] = ev.price_usd * fx[c]
        else:
            raise InjectError(f"no price for {spec.iso3}: give price_usd or price_local")
        world.product_available[pid, c] = day + ev.available_from_day_offset
    n_push = 0
    for push in ev.pushes:
        target = [_country_index(world, c) for c in push.countries] if push.countries else countries
        kinds = {int(CHANNEL_KIND_NAMES[k]) for k in push.channel_kinds}
        n_push += _push(world, item_id, target, kinds, day + push.start_day_offset, push.duration_days, push.intensity)
    return n_push


def product_id_by_name(world: World, name: str) -> int:
    for p, n in enumerate(world.product_names):
        if n == name:
            return p
    raise InjectError(f"no product named {name!r} in this world (known: {world.product_names})")


def start_news(world: World, ev: NewsEvent, log: EventLog, prompt_id: int) -> int:
    """Create the news item now and push it through the requested channels."""
    countries = [_country_index(world, c) for c in ev.countries] if ev.countries else list(range(world.n_countries))
    entity = -1
    truth = 0.5 + 0.5 * ev.valence * ev.accuracy
    if ev.about_product:
        entity = product_id_by_name(world, ev.about_product)
        truth = float(world.products["true_quality"][entity])
    item_id = create_item(
        world,
        topic=int(TOPIC_NAMES[ev.topic]),
        valence=ev.valence,
        truth=truth,
        salience=ev.salience,
        origin={"scenario": OriginKind.SCENARIO, "user_feed": OriginKind.USER_FEED, "web": OriginKind.WEB}[ev.origin],
        origin_id=prompt_id,
        entity_id=entity,
        active_days=ev.duration_days + 20,
        release_utc_hour=ev.release_utc_hour,
        sentiment_weight=ev.weight(),
        label=ev.headline,
        log=log,
    )
    if item_id < 0:
        return 0
    kinds = {int(CHANNEL_KIND_NAMES[k]) for k in ev.channel_kinds}
    lang = world.lang_id(ev.language) if ev.language else None
    if lang is not None:
        world.items["lang"][item_id] = lang
    return _push(world, item_id, countries, kinds, world.day, ev.duration_days, ev.intensity, lang)


def notable_by_key(world: World, key: str) -> dict:
    for entry in world.notables:
        if entry["key"] == key:
            return entry
    raise InjectError(f"no notable {key!r} in this world (known: {[e['key'] for e in world.notables]})")


def statement_salience(visibility: float) -> float:
    """Engine-computed salience from the speaker's visibility (never from the wording)."""
    return float(np.clip(0.25 + 0.08 * visibility, 0.3, 0.95))


def start_statement(world: World, ev: Statement, log: EventLog, prompt_id: int) -> int:
    entry = notable_by_key(world, ev.speaker)
    roles = world.roles
    allowed = roles[entry["role"]].can_post if entry["role"] in roles else ["national"]
    audience = ev.audience if ev.audience in allowed else "national"
    home = int(entry["country"])
    entity = -1
    truth = 0.5 + 0.5 * ev.valence * ev.accuracy
    if ev.about_product:
        entity = product_id_by_name(world, ev.about_product)
        truth = float(world.products["true_quality"][entity])
    topic = ev.topic
    label = f"{SIMULATED_LABEL} {entry['holder_name']} ({entry['office']}): {ev.headline}"
    item_id = create_item(
        world,
        topic=int(TOPIC_NAMES[topic]),
        valence=ev.valence,
        truth=truth,
        salience=statement_salience(float(entry["visibility"])),
        origin=OriginKind.NOTABLE,
        origin_id=int(entry["agent_id"]),
        entity_id=entity,
        active_days=ev.duration_days + 60,
        release_utc_hour=ev.release_utc_hour,
        sentiment_weight=DEFAULT_SENTIMENT_WEIGHT.get(topic, 0.2),
        label=label,
        log=log,
    )
    if item_id < 0:
        return 0
    all_kinds = {int(k) for k in CHANNEL_KIND_NAMES.values()}
    n = _push(world, item_id, [home], all_kinds, world.day, ev.duration_days, 1.0)
    if audience == "global":
        others = [c for c in range(world.n_countries) if c != home]
        n += _push(world, item_id, others, all_kinds, world.day + 1, ev.duration_days, 0.7)  # pickup lag abroad
    return n


def _schedule_macro(world: World, ev: MacroPath) -> int:
    if ev.country is not None:
        _country_index(world, ev.country)  # validate
    n = 0
    for step in ev.steps:
        for d in range(step.days):
            world.scheduled.append(
                {
                    "day": world.day + step.start_day_offset + d,
                    "kind": "macro",
                    "variable": ev.variable,
                    "country": ev.country,
                    "pct": step.pct_per_day,
                    "auto_news": ev.auto_news,
                }
            )
            n += 1
    return n


def apply_scenario(world: World, scenario: Scenario, log: EventLog, prompt_id: int) -> dict[str, int]:
    log.add(world.day, EventKind.SCENARIO_INJECTED, -1, prompt_id, -1, 0.0, -1, scenario.prompt[:500])
    stats = {"events": len(scenario.events), "pushes": 0, "scheduled": 0}
    for ev in scenario.events:
        if isinstance(ev, ACTOR_TYPES):
            from jahan.actors import engine  # deferred: the actor engine builds on this module

            stats["scheduled"] += engine.schedule(world, ev, log, prompt_id)
        elif isinstance(ev, ProductLaunch):
            stats["pushes"] += _launch(world, ev, log, prompt_id)
        elif isinstance(ev, NewsEvent):
            if ev.start_day_offset == 0:
                stats["pushes"] += start_news(world, ev, log, prompt_id)
            else:
                spec: dict[str, Any] = ev.model_dump(mode="json")
                world.scheduled.append(
                    {"day": world.day + ev.start_day_offset, "kind": "news", "prompt_id": prompt_id, "spec": spec}
                )
                stats["scheduled"] += 1
        elif isinstance(ev, MacroPath):
            stats["scheduled"] += _schedule_macro(world, ev)
        elif isinstance(ev, Statement):
            notable_by_key(world, ev.speaker)  # validate now, even if it starts later
            if ev.start_day_offset == 0:
                stats["pushes"] += start_statement(world, ev, log, prompt_id)
            else:
                world.scheduled.append(
                    {
                        "day": world.day + ev.start_day_offset,
                        "kind": "statement",
                        "prompt_id": prompt_id,
                        "spec": ev.model_dump(mode="json"),
                    }
                )
                stats["scheduled"] += 1
        else:  # pragma: no cover - guarded by the discriminated union
            raise InjectError(f"unsupported event type {type(ev).__name__}")
    # stable order: by day, then insertion order
    world.scheduled.sort(key=lambda e: e["day"])
    return stats
