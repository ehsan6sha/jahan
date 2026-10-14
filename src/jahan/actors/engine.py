"""Applying actions inside the tick.  No LLM is ever called here.

Timing (part of the reproducibility contract):

* ``create_agent`` happens when its prompt is applied (before the day's availability masks exist, so the
  new person is part of that day).  ``control``, ``decision`` and ``price_change`` also happen then, or
  at the start of a later day (``start_day_offset``).
* Messages and posts are delivered right after the day's availability masks are computed.  A receiver
  awake at or after the send hour (UTC bins of this day) gets it today; anyone else gets it on the next
  day.  Acceptance uses the same bounded-confidence rule as word of mouth, with the receiver's trust
  in the sender: the tie's trust for people in the sender's circle, parasocial trust for public
  figures, and a stranger prior from reputation otherwise.
* Salience comes from the sender's visibility, never from the wording.  Each sender is limited to
  ``MAX_MESSAGES_PER_DAY`` messages and ``MAX_POSTS_PER_DAY`` posts (HEURISTIC limits on a person's day).

Edge direction: ``src -> dst`` means src listens to dst.  A sender's circle is every ``src`` with
``dst == sender``; a message to a target needs the edge ``target -> sender``.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, TypeAdapter

from jahan.actors.actions import (
    AUDIENCES,
    ActorEvent,
    Claim,
    Control,
    CreateAgent,
    Decision,
    Message,
    Post,
    PriceChange,
)
from jahan.gen.sampler import logit, sigmoid
from jahan.model.country import CAMP_TAU
from jahan.model.enums import CHANNEL_KIND_NAMES, TOPIC_NAMES, EdgeKind, EmpStatus, EventKind, Marital, OriginKind
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.schema import AGENT_COLUMNS
from jahan.model.world import World
from jahan.rng import MASK64, Stream, u01
from jahan.scenario.inject import InjectError, _push, create_item, product_id_by_name, statement_salience
from jahan.scenario.spec import DEFAULT_SENTIMENT_WEIGHT
from jahan.sim.availability import Masks
from jahan.sim.diffusion import acceptance
from jahan.sim.params import SimParams
from jahan.sim.ties import add_ties
from jahan.sim.trust import effective_trust, parasocial_trust
from jahan.store.eventlog import EventLog

ADAPTER: TypeAdapter[Any] = TypeAdapter(Annotated[ActorEvent, Field(discriminator="type")])

KIND_DEFERRED = "actor"  # messages and posts, delivered after the availability masks
KIND_LATER = "actor_now"  # control / decision / price change scheduled for a later day
KIND_DELIVER = "deliver"  # receivers who were asleep get it on the next day

MAX_MESSAGES_PER_DAY = 20  # HEURISTIC
MAX_POSTS_PER_DAY = 3  # HEURISTIC
COMMUNITY_REACH_SCALE = 1000.0  # HEURISTIC: 10**visibility people reach ~all of a local community this size
STRANGER_TRUST = 0.45  # prior trust in someone outside one's circle (same as an acquaintance tie)
MESSAGE_ACTIVE_DAYS = 7
LOG_KEEP_DAYS = 14
ALL_KINDS = {int(k) for k in CHANNEL_KIND_NAMES.values()}


def parse(spec: dict[str, Any]) -> ActorEvent:
    ev: ActorEvent = ADAPTER.validate_python(spec)
    return ev


# ------------------------------------------------------------------ who is who
def display_name(world: World, row: int) -> str:
    pop = world.pop
    custom = world.macro.get("custom_names", {}).get(str(int(pop["id"][row])))
    if custom:
        return str(custom)
    if pop["is_notable"][row] and 0 <= int(pop["role_id"][row]) < len(world.notables):
        return str(world.notables[int(pop["role_id"][row])]["holder_name"])
    nt = world.names[int(pop["country"][row])]
    return nt.display(int(pop["first_name_id"][row]), int(pop["last_name_id"][row]))


def notable_entry(world: World, row: int) -> dict[str, Any] | None:
    pop = world.pop
    col = int(pop["role_id"][row])
    if not pop["is_notable"][row] or not 0 <= col < len(world.notables):
        return None
    return world.notables[col]


def is_real_person(world: World, row: int) -> bool:
    """A roster figure (a real office holder): everything generated for them is labelled [simulated]."""
    entry = notable_entry(world, row)
    return entry is not None and not entry.get("created", False)


def allowed_audiences(world: World, row: int) -> list[str]:
    pop = world.pop
    out = ["circle"]
    entry = notable_entry(world, row)
    if entry is not None or float(pop["visibility"][row]) >= 2.0:
        out.append("community")
    if entry is not None:
        role = world.roles.get(entry["role"])
        can = list(role.can_post) if role is not None else ["national"]
        out += [a for a in ("national", "global") if a in can]
    return out


def effective_audience(world: World, row: int, requested: str) -> str:
    """The widest permitted audience that is not wider than the one requested."""
    cap = AUDIENCES.index(requested)
    best = 0
    for a in allowed_audiences(world, row):
        i = AUDIENCES.index(a)
        if best < i <= cap:
            best = i
    return AUDIENCES[best]


def circle_edges(world: World, agent_id: int) -> NDArray[np.int64]:
    """Edge rows of everyone who listens to ``agent_id`` (alive, active ties)."""
    e = world.edges
    rows = np.flatnonzero((e["dst"] == agent_id) & e["active"])
    return rows[world.pop["alive"][e["src"][rows]]]


def _alive(world: World, agent_id: int) -> bool:
    return 0 <= agent_id < world.pop.n and bool(world.pop["alive"][world.agent_row(agent_id)])


def _product(world: World, name: str) -> int:
    try:
        return product_id_by_name(world, name)
    except InjectError:
        return -1


# ------------------------------------------------------------------ validation
def check(world: World, ev: ActorEvent) -> str | None:
    """Why this action cannot happen now, or None.  Used before injecting (rejection) and when applying
    (a recorded action whose preconditions fail in another branch is skipped, not fatal)."""
    if isinstance(ev, CreateAgent):
        if ev.country not in [c.iso3 for c in world.countries]:
            return f"country {ev.country} is not in this world"
        if ev.household_of is not None and not _alive(world, ev.household_of):
            return f"agent {ev.household_of} does not exist or has died"
        if ev.notable_role is not None:
            if ev.notable_role not in world.roles:
                return f"unknown role {ev.notable_role}; known: {sorted(world.roles)}"
            if len(world.notables) >= world.ptrust.cols:
                return "all reserved public-figure slots are used"
        return None
    if isinstance(ev, PriceChange):
        p = _product(world, ev.product)
        if p < 0:
            return f"no product named {ev.product!r}"
        if ev.actor >= 0 and world.macro.get("product_owner", {}).get(ev.product) != ev.actor:
            return f"agent {ev.actor} does not make {ev.product!r} (only products it launched)"
        return None
    if isinstance(ev, Control):
        if not _alive(world, ev.agent):
            return f"agent {ev.agent} does not exist or has died"
        cur = int(world.pop["controlled_by"][world.agent_row(ev.agent)])
        if ev.on and cur not in (-1, ev.session):
            return f"agent {ev.agent} is controlled by session {cur}"
        if not ev.on and cur != ev.session:
            return f"agent {ev.agent} is not controlled by session {ev.session}"
        return None
    actor = ev.actor
    if not _alive(world, actor):
        return f"agent {actor} does not exist or has died"
    if isinstance(ev, Message | Post) and ev.claim.about_product and _product(world, ev.claim.about_product) < 0:
        return f"no product named {ev.claim.about_product!r}"
    if isinstance(ev, Message):
        if ev.target == actor:
            return "a message needs another person"
        if not _alive(world, ev.target):
            return f"agent {ev.target} does not exist or has died"
        edge = int(world.find_edges(np.array([ev.target]), np.array([actor]))[0])
        if edge < 0 or not world.edges["active"][edge]:
            return f"agent {ev.target} is not in agent {actor}'s circle"
    if isinstance(ev, Decision):
        p = _product(world, ev.product)
        if p < 0:
            return f"no product named {ev.product!r}"
        if ev.choice == "adopt":
            loc = int(world.pop["location_country"][world.agent_row(actor)])
            avail = int(world.product_available[p, loc])
            if avail < 0 or avail > world.day:
                return f"{ev.product} is not on sale where agent {actor} is"
            slot = int(world.items["slot"][int(world.products["item_id"][p])])
            if slot >= 0 and world.beliefs.m["adopted_day"][world.agent_row(actor), slot] >= 0:
                return f"agent {actor} already has {ev.product}"
    return None


# ------------------------------------------------------------------ logging
def _actor_of(ev: ActorEvent) -> int:
    if isinstance(ev, Control):
        return ev.agent
    if isinstance(ev, CreateAgent):
        return -1
    return ev.actor


def _log_action(world: World, ev: ActorEvent, log: EventLog, detail: str, other: int = -1, item: int = -1) -> None:
    actor = _actor_of(ev)
    country = int(world.pop["location_country"][world.agent_row(actor)]) if actor >= 0 else -1
    kind = EventKind.HUMAN_ACTION if ev.session >= 0 else EventKind.NOTABLE_ACTION
    log.add(world.day, kind, actor, other, item, float(ev.session), country, detail[:300])


def _skip(world: World, ev: ActorEvent, log: EventLog, reason: str) -> None:
    log.add(world.day, EventKind.ACTION_SKIPPED, _actor_of(ev), ev.session, -1, 0.0, -1, f"{ev.type}: {reason}"[:300])


# ------------------------------------------------------------------ entry points
def schedule(world: World, ev: ActorEvent, log: EventLog, prompt_id: int) -> int:
    """Called by apply_scenario.  Returns 1 if the action was scheduled for later in the tick or a later day."""
    if isinstance(ev, CreateAgent):
        reason = check(world, ev)
        if reason:
            _skip(world, ev, log, reason)
        else:
            create_agent(world, ev, log, prompt_id)
        return 0
    entry = {"day": world.day + ev.start_day_offset, "prompt_id": prompt_id, "spec": ev.model_dump(mode="json")}
    if isinstance(ev, Message | Post):
        world.scheduled.append({**entry, "kind": KIND_DEFERRED})
        return 1
    if ev.start_day_offset > 0:
        world.scheduled.append({**entry, "kind": KIND_LATER})
        return 1
    apply_now(world, ev, log, prompt_id)
    return 0


def apply_now(world: World, ev: ActorEvent, log: EventLog, prompt_id: int) -> None:
    reason = check(world, ev)
    if reason:
        _skip(world, ev, log, reason)
        return
    if isinstance(ev, Control):
        row = world.agent_row(ev.agent)
        world.pop["controlled_by"][row] = ev.session if ev.on else -1
        kind = EventKind.POSSESSED if ev.on else EventKind.RELEASED
        log.add(world.day, kind, ev.agent, ev.session, -1, 0.0, int(world.pop["location_country"][row]))
    elif isinstance(ev, Decision):
        _decide(world, ev, log)
    elif isinstance(ev, PriceChange):
        _price_change(world, ev, log)
    else:  # pragma: no cover - messages and posts are always deferred
        raise InjectError(f"{ev.type} cannot be applied at phase 0")


def deliver_actions(world: World, masks: Masks, params: SimParams, log: EventLog) -> dict[str, int]:
    """Tick phase 1b: messages, posts and next-day deliveries due today, in scheduling order."""
    day = world.day
    kinds = (KIND_DEFERRED, KIND_DELIVER)
    due = [e for e in world.scheduled if e["day"] <= day and e["kind"] in kinds]
    if not due:
        return {"actions_delivered": 0}
    world.scheduled = [e for e in world.scheduled if not (e["day"] <= day and e["kind"] in kinds)]
    used: dict[tuple[int, str], int] = {}
    delivered = 0
    for e in due:
        if e["kind"] == KIND_DELIVER:
            delivered += _deliver_pending(world, e, params, log)
            continue
        ev = parse(e["spec"])
        assert isinstance(ev, Message | Post)
        key = (ev.actor, ev.type)
        used[key] = used.get(key, 0) + 1
        limit = MAX_MESSAGES_PER_DAY if isinstance(ev, Message) else MAX_POSTS_PER_DAY
        if used[key] > limit:
            _skip(world, ev, log, f"daily limit of {limit} {ev.type}s reached")
            continue
        reason = check(world, ev)
        if reason:
            _skip(world, ev, log, reason)
            continue
        if isinstance(ev, Message):
            delivered += _message(world, ev, masks, params, log)
        else:
            delivered += _post(world, ev, masks, params, log)
    _prune_logs(world)
    return {"actions_delivered": delivered}


# ------------------------------------------------------------------ claims and delivery
def claim_item(
    world: World, row: int, claim: Claim, hour: int, active_days: int, salience: float, session: int, log: EventLog
) -> int:
    """Tracked item for a claim by agent ``row`` (reused while the same claim by the same speaker is live)."""
    pop = world.pop
    aid = int(pop["id"][row])
    name = display_name(world, row)
    label = (
        f"{SIMULATED_LABEL} {name}: {claim.headline}" if is_real_person(world, row) else f"{name}: {claim.headline}"
    )[:300]
    labels = world.macro.get("item_labels", {})
    items = world.items
    for i in range(items.n - 1, -1, -1):
        if labels.get(str(i)) == label and items["slot"][i] >= 0 and items["active_until"][i] >= world.day:
            items["active_until"][i] = max(int(items["active_until"][i]), world.day + active_days)
            return i
    entity, truth = -1, 0.5  # claims about the wider world are unverifiable in the simulation
    if claim.about_product:
        entity = product_id_by_name(world, claim.about_product)
        truth = float(world.products["true_quality"][entity])
    if pop["is_notable"][row]:
        origin = OriginKind.NOTABLE
    else:
        origin = OriginKind.HUMAN if session >= 0 else OriginKind.AGENT
    return create_item(
        world,
        topic=int(TOPIC_NAMES[claim.topic]),
        valence=claim.stance,
        truth=truth,
        salience=salience,
        origin=origin,
        origin_id=aid,
        entity_id=entity,
        active_days=active_days,
        release_utc_hour=hour,
        sentiment_weight=DEFAULT_SENTIMENT_WEIGHT.get(claim.topic, 0.2),
        label=label,
        log=log,
    )


def receive(
    world: World,
    item: int,
    sender: int,
    rows: NDArray[np.int64],
    trust: NDArray[np.float64],
    conveyed: float,
    params: SimParams,
    log: EventLog,
) -> dict[str, NDArray[Any]]:
    """Receivers ``rows`` hear ``item`` from ``sender`` now.  Same rule as word of mouth (bounded confidence,
    trust, openness, discernment); draws keyed by (day, receiver id, item) on the ACTION stream."""
    slot = int(world.items["slot"][item])
    empty = {"rows": rows[:0], "accepted": np.zeros(0, bool), "before": np.zeros(0), "after": np.zeros(0)}
    if slot < 0 or rows.size == 0:
        return empty
    bel = world.beliefs.m
    pop = world.pop
    day = world.day
    ids = pop["id"][rows]
    conv = np.full(rows.size, conveyed, dtype=np.float64)
    first = bel["aware"][rows, slot] == 0
    before = np.where(first, 0.5, bel["belief"][rows, slot].astype(np.float64))
    within, p_acc = acceptance(world, params, rows, conv, trust, before, first)
    acc = within & (u01(world.sim_seed, Stream.ACTION, day, ids, k=item) < p_acc)
    if first.any():
        f = rows[first]
        bel["aware"][f, slot] = 1
        bel["first_day"][f, slot] = day
        bel["belief"][f, slot] = 0.5
        log.add(day, EventKind.INFO_EXPOSED, ids[first], sender, item, conveyed, pop["location_country"][f])
    if acc.any():
        r = rows[acc]
        d = pop["discernment"][r].astype(np.float64)
        truth = float(world.items["truth"][item])
        target = conv[acc] * (1 - params.discernment_pull * d) + params.discernment_pull * d * truth
        cur = bel["belief"][r, slot].astype(np.float64)
        bel["belief"][r, slot] = (cur + params.k_wom * trust[acc] * (target - cur)).astype(np.float32)
        bel["heard_from"][r, slot] = sender
        bel["heard_val"][r, slot] = np.float32(conveyed)
    after = bel["belief"][rows, slot].astype(np.float64)
    return {"rows": rows, "accepted": acc, "before": before, "after": after}


def _deliver(
    world: World,
    item: int,
    sender: int,
    rows: NDArray[np.int64],
    trust: NDArray[np.float64],
    conveyed: float,
    hour: int,
    masks: Masks,
    params: SimParams,
    log: EventLog,
    message_id: int = -1,
) -> int:
    """Deliver now to receivers awake at or after ``hour`` (UTC); the rest get it on the next day."""
    if rows.size == 0:
        return 0
    awake_later = (masks.awake_utc[rows] >> np.uint32(hour)) != 0
    later = rows[~awake_later]
    if later.size:
        world.scheduled.append(
            {
                "day": world.day + 1,
                "kind": KIND_DELIVER,
                "item": item,
                "sender": sender,
                "receivers": [int(x) for x in world.pop["id"][later]],
                "trust": [float(x) for x in trust[~awake_later]],
                "value": conveyed,
                "message_id": message_id,
            }
        )
    res = receive(world, item, sender, rows[awake_later], trust[awake_later], conveyed, params, log)
    _record_message(world, message_id, res)
    return int(awake_later.sum())


def _deliver_pending(world: World, e: dict[str, Any], params: SimParams, log: EventLog) -> int:
    ids = np.asarray(e["receivers"], dtype=np.int64)
    rows = np.array([world.agent_row(int(i)) for i in ids], dtype=np.int64)
    alive = world.pop["alive"][rows]
    res = receive(
        world,
        int(e["item"]),
        int(e["sender"]),
        rows[alive],
        np.asarray(e["trust"], dtype=np.float64)[alive],
        float(e["value"]),
        params,
        log,
    )
    _record_message(world, int(e.get("message_id", -1)), res)
    return int(alive.sum())


def _record_message(world: World, message_id: int, res: dict[str, NDArray[Any]]) -> None:
    """Outcome of a one-to-one message (for briefings and replies that match the numbers)."""
    if message_id < 0 or res["rows"].size == 0:
        return
    rec = world.macro.get("messages", {}).get(str(message_id))
    if rec is None:
        return
    rec.update(
        {
            "delivered_day": world.day,
            "accepted": bool(res["accepted"][0]),
            "belief_before": round(float(res["before"][0]), 3),
            "belief_after": round(float(res["after"][0]), 3),
        }
    )


def _prune_logs(world: World) -> None:
    msgs = world.macro.get("messages", {})
    old = [k for k, v in msgs.items() if v["day"] < world.day - LOG_KEEP_DAYS]
    for k in old:
        del msgs[k]


def _message(world: World, ev: Message, masks: Masks, params: SimParams, log: EventLog) -> int:
    a = world.agent_row(ev.actor)
    t = world.agent_row(ev.target)
    sal = statement_salience(float(world.pop["visibility"][a]))
    item = claim_item(world, a, ev.claim, ev.utc_hour, MESSAGE_ACTIVE_DAYS, sal, ev.session, log)
    if item < 0:
        _skip(world, ev, log, "no free tracked-item slot")
        return 0
    edge = int(world.find_edges(np.array([t]), np.array([a]))[0])
    trust = effective_trust(world, np.array([edge]), params)
    msgs = world.macro.setdefault("messages", {})
    mid = int(world.macro.get("next_message_id", 0))
    world.macro["next_message_id"] = mid + 1
    msgs[str(mid)] = {
        "day": world.day,
        "actor": ev.actor,
        "target": ev.target,
        "item": item,
        "session": ev.session,
        "headline": ev.claim.headline,
        "text": ev.claim.text,
        "trust": round(float(trust[0]), 3),
        "delivered_day": None,
    }
    conveyed = 0.5 + 0.5 * ev.claim.stance
    n = _deliver(world, item, ev.actor, np.array([t]), trust, conveyed, ev.utc_hour, masks, params, log, mid)
    _log_action(world, ev, log, f"message: {ev.claim.headline}", other=ev.target, item=item)
    return n


def _post(world: World, ev: Post, masks: Masks, params: SimParams, log: EventLog) -> int:
    pop = world.pop
    a = world.agent_row(ev.actor)
    aid = int(pop["id"][a])
    aud = effective_audience(world, a, ev.audience)
    vis = float(pop["visibility"][a])
    sal = min(0.95, statement_salience(vis) + ev.salience_bonus)
    item = claim_item(world, a, ev.claim, ev.utc_hour, ev.duration_days + MESSAGE_ACTIVE_DAYS, sal, ev.session, log)
    if item < 0:
        _skip(world, ev, log, "no free tracked-item slot")
        return 0
    conveyed = 0.5 + 0.5 * ev.claim.stance
    e = world.edges
    erows = circle_edges(world, aid)
    circle = e["src"][erows].astype(np.int64)
    n = _deliver(
        world, item, aid, circle, effective_trust(world, erows, params), conveyed, ev.utc_hour, masks, params, log
    )
    entry = notable_entry(world, a)
    if AUDIENCES.index(aud) >= AUDIENCES.index("community"):
        # local community: friends of friends in the same country, reached with a visibility-based chance
        e2 = np.flatnonzero(np.isin(e["dst"], circle) & e["active"])
        cand = np.unique(e["src"][e2]).astype(np.int64)
        same = pop["location_country"][cand] == pop["location_country"][a]
        cand = cand[(cand != aid) & ~np.isin(cand, circle) & pop["alive"][cand] & same]
        p = min(1.0, 10.0**vis / COMMUNITY_REACH_SCALE)
        hit = cand[u01(world.sim_seed, Stream.ACTION, world.day, pop["id"][cand], k=item * 4 + 1) < p]
        if entry is not None:
            tr = parasocial_trust(world, hit, int(pop["role_id"][a]), params)
        else:
            rep = float(pop["global_rep"][a])
            tr = np.full(hit.size, float(sigmoid(logit(STRANGER_TRUST) + 0.8 * (rep - 0.5))))
        n += _deliver(world, item, aid, hit, tr, conveyed, ev.utc_hour, masks, params, log)
    if aud in ("national", "global") and entry is not None:
        home = int(entry["country"])
        n += _push(world, item, [home], ALL_KINDS, world.day, ev.duration_days, 1.0)
        if aud == "global":
            others = [c for c in range(world.n_countries) if c != home]
            n += _push(world, item, others, ALL_KINDS, world.day + 1, ev.duration_days, 0.7)
    note = "" if aud == ev.audience else f" (asked for {ev.audience}; allowed {aud})"
    _log_action(world, ev, log, f"post to {aud}: {ev.claim.headline}{note}", item=item)
    return n


# ------------------------------------------------------------------ decisions and prices
def _decide(world: World, ev: Decision, log: EventLog) -> None:
    p = product_id_by_name(world, ev.product)
    item = int(world.products["item_id"][p])
    if ev.choice == "adopt":
        from jahan.sim.decisions import income_basis

        row = world.agent_row(ev.actor)
        slot = int(world.items["slot"][item])
        loc = int(world.pop["location_country"][row])
        afford = float(world.product_price[p, loc]) / max(float(income_basis(world)[row]), 1.0)
        if slot >= 0:
            bel = world.beliefs.m
            if bel["aware"][row, slot] == 0:
                bel["aware"][row, slot] = 1
                bel["first_day"][row, slot] = world.day
            bel["adopted_day"][row, slot] = world.day
            adviser = int(bel["heard_from"][row, slot])
            log.add(world.day, EventKind.ADOPTED, ev.actor, adviser, item, afford, loc)
    _log_action(world, ev, log, f"decision: {ev.choice} {ev.product}", item=item)


def _price_change(world: World, ev: PriceChange, log: EventLog) -> None:
    p = product_id_by_name(world, ev.product)
    codes = [c.iso3 for c in world.countries]
    if ev.countries:
        cs = [codes.index(c) for c in ev.countries if c in codes]
    else:
        cs = [c for c in range(world.n_countries) if world.product_available[p, c] >= 0]
    factor = 1.0 + ev.pct / 100.0
    world.product_price[p, cs] *= np.float32(factor)
    world.product_price_usd[p, cs] *= factor
    _log_action(world, ev, log, f"price of {ev.product} {ev.pct:+.1f}% in {[codes[c] for c in cs]}")


# ------------------------------------------------------------------ creating people
def _rng(world: World, agent_id: int) -> np.random.Generator:
    return np.random.default_rng([world.sim_seed & MASK64, int(Stream.ACTION), world.day, int(agent_id)])


def _template(world: World, ev: CreateAgent, c: int, new_id: int) -> int:
    """A resident of the same country (and sex and age band when given) whose profile seeds the new person."""
    pop = world.pop
    base = pop["alive"] & ~pop["is_notable"] & (pop["pop_weight"] > 0) & (pop["country"] == c)
    if ev.sex is not None:
        base &= pop["sex"] == (0 if ev.sex == "F" else 1)
    if ev.region is not None:
        base &= pop["region"] == (1 if ev.region == "urban" else 0)
    cand = np.flatnonzero(base)
    if ev.age is not None:
        age = world.ages()
        for width in (3, 8, 15, 200):
            near = cand[np.abs(age[cand] - ev.age) <= width]
            if near.size:
                cand = near
                break
    if ev.age is None or ev.age >= 18:
        adults = cand[world.ages()[cand] >= 18]
        cand = adults if adults.size else cand
    if cand.size == 0:
        raise InjectError(f"no resident of {world.countries[c].iso3} to draw a profile from")
    draws = u01(world.sim_seed, Stream.ACTION, world.day, pop["id"][cand], k=new_id)
    return int(cand[int(np.argmin(draws))])


def create_agent(world: World, ev: CreateAgent, log: EventLog, prompt_id: int) -> int:
    pop = world.pop
    codes = [c.iso3 for c in world.countries]
    c = codes.index(ev.country)
    new_id = int(world.counters["agent"])
    tpl = _template(world, ev, c, new_id)
    world.alloc_ids("agent", 1)
    rng = _rng(world, new_id)
    cols: dict[str, Any] = {}
    for name, col in AGENT_COLUMNS.items():
        v = pop[name][tpl]
        cols[name] = np.array(v)[None, ...] if col.shape else v.item()
    age = float(ev.age) if ev.age is not None else float(world.ages()[tpl])
    cols.update(
        id=new_id,
        alive=True,
        country=c,
        birth_country=c,
        location_country=c,
        pop_weight=0.0,
        birth_day=world.day - int(round(age * 365.25)) - int(rng.integers(0, 365)),
        marital=int(Marital.SINGLE),
        spouse_id=-1,
        mother_id=-1,
        father_id=-1,
        n_children=0,
        last_birth_day=-100000,
        death_day=-1,
        is_notable=False,
        role_id=-1,
        controlled_by=-1,
        trip_end_day=-1,
        tz_offset_min=int(pop["home_tz_offset_min"][tpl]),
        home_tz_offset_min=int(pop["home_tz_offset_min"][tpl]),
        attractiveness_z=float(rng.standard_normal()),
        chronotype=float(rng.standard_normal()),
    )
    if ev.sex is not None:
        cols["sex"] = 0 if ev.sex == "F" else 1
    if ev.region is not None:
        cols["region"] = 1 if ev.region == "urban" else 0
    if ev.education is not None:
        cols["education"] = ev.education
    if ev.income_month is not None:
        cols["income_month"] = ev.income_month
    if ev.visibility is not None:
        cols["visibility"] = ev.visibility
    if ev.alignment is not None:
        cols["alignment"] = ev.alignment
    if ev.values:
        order = ["self", "money", "culture", "freedom", "family", "security"]
        v = np.array([max(ev.values.get(k, 0.0), 0.0) for k in order], np.float32)
        if v.sum() > 0:
            cols["values"] = (v / v.sum())[None, :]
    if ev.languages:
        lang = np.full(cols["lang"].shape[1], -1, np.int16)
        codes_l = [world.lang_id(x) for x in ev.languages][: lang.shape[0]]
        lang[: len(codes_l)] = codes_l
        cols["lang"] = lang[None, :]
        cols["lang_prof"] = np.where(lang >= 0, 1.0, 0.0).astype(np.float32)[None, :]
    if world.ages()[tpl] < 18 <= age:  # a child template cannot give an adult a job
        cols["emp_status"] = int(EmpStatus.UNEMPLOYED)
        cols["workplace_id"] = -1
    # household: alone (a new household) or joining someone's
    if ev.household_of is not None:
        hh = int(pop["household_id"][world.agent_row(ev.household_of)])
    else:
        hh = -1
    if hh < 0:
        hh = world.households.n
        world.households.append(1, id=hh, country=c, region=int(cols["region"]), n_members=0, income_pooled=0.0)
        world.counters["household"] = world.households.n
    cols["household_id"] = hh
    world.pop.append(1, **cols)
    world.beliefs.ensure_rows(world.pop.n)
    world.ptrust.ensure_rows(world.pop.n)
    pop = world.pop
    row = world.agent_row(new_id)
    world.households["n_members"][hh] += 1
    if pop["workplace_id"][row] >= 0:
        world.workplaces["size"][int(pop["workplace_id"][row])] += 1
    name = " ".join(x for x in (ev.first_name, ev.last_name) if x)
    if name:
        world.macro.setdefault("custom_names", {})[str(new_id)] = name
    _wire(world, ev, row)
    if ev.notable_role is not None:
        notable_col = len(world.notables)
        role = world.roles[ev.notable_role]
        pop["is_notable"][row] = True
        pop["role_id"][row] = notable_col
        if ev.visibility is None:
            pop["visibility"][row] = 4.0
        order = ["self", "money", "culture", "freedom", "family", "security"]
        holder = display_name(world, row)
        lean = float(pop["alignment"][row])
        world.notables.append(
            {
                "key": f"{ev.country}.created_{new_id}",
                "role": ev.notable_role,
                "office": ev.office or role.label,
                "holder_name": holder,
                "aliases": sorted({holder, *([ev.last_name] if ev.last_name else [])}),
                "alignment": lean,
                "camp_prior": {"aligned": 0.5, "neutral": 0.5, "opposed": 0.5},
                "camp_shift_logit": 0.0,
                "camp_status": "placeholder" if abs(lean) >= CAMP_TAU else "non_partisan",
                "sex": "F" if pop["sex"][row] == 0 else "M",
                "visibility": float(pop["visibility"][row]),
                "approval": {"value": 0.5, "source": "created in the simulation (no polling)"},
                "values": {k: float(pop["values"][row, i]) for i, k in enumerate(order)},
                "stances": [],
                "agent_id": new_id,
                "country": c,
                "as_of": world.date().isoformat(),
                "created": True,
            }
        )
    log.add(world.day, EventKind.AGENT_CREATED, new_id, tpl, -1, float(prompt_id), c, ev.description[:300] or None)
    return new_id


def _wire(world: World, ev: CreateAgent, row: int) -> None:
    """Household, workplace, friends and acquaintances (same country and region, shared language)."""
    pop = world.pop
    new_id = int(pop["id"][row])
    hh = int(pop["household_id"][row])
    members = np.flatnonzero((pop["household_id"] == hh) & pop["alive"])
    members = members[members != row]
    if members.size:
        add_ties(world, np.full(members.size, new_id), pop["id"][members], EdgeKind.HOUSEHOLD)
    w = int(pop["workplace_id"][row])
    if w >= 0:
        mates = np.flatnonzero((pop["workplace_id"] == w) & pop["alive"])
        mates = mates[mates != row][:5]
        if mates.size:
            add_ties(world, np.full(mates.size, new_id), pop["id"][mates], EdgeKind.COLLEAGUE)
    local = pop["alive"] & ~pop["is_notable"] & (pop["country"] == pop["country"][row])
    local &= pop["region"] == pop["region"][row]
    local[row] = False
    my_langs = pop["lang"][row][pop["lang"][row] >= 0]
    speaks = np.isin(pop["lang"], my_langs).any(axis=1)
    age = world.ages()
    near_age = np.abs(age - age[row]) <= 10
    draws = u01(world.sim_seed, Stream.ACTION, world.day, pop["id"], k=new_id + 1)
    for kind, k, mask in (
        (EdgeKind.FRIEND, ev.friends, local & speaks & near_age),
        (EdgeKind.ACQUAINTANCE, ev.acquaintances, local & speaks),
    ):
        cand = np.flatnonzero(mask)
        if k <= 0 or cand.size == 0:
            continue
        pick = cand[np.argsort(draws[cand], kind="stable")[:k]]
        add_ties(world, np.full(pick.size, new_id), pop["id"][pick], kind)
