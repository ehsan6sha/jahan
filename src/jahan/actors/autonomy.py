"""Public figures act on salient days (one LLM call per turn), within caps.

At the start of each simulated day the engine looks for triggers, from world state only:

* a live tracked item on a topic the figure's role cares about crosses ``AWARE_TRIGGER`` of the
  figure's country (once per item);
* a market-move headline in the figure's country appeared yesterday (economic roles);
* the figure is named in an item created yesterday by someone else: any roster alias (surname, title,
  handle, native-script name; model/mentions.py) or an @-mention of one of their handles in a social post;
* a news headline created yesterday on one of the role's topics, salience at least ``HEADLINE_TRIGGER``,
  pushed in the figure's country;
* the role's scheduled decision day (``meeting_every_days`` in roles.yaml, e.g. a central bank).

At most ``TURNS_PER_NOTABLE_PER_WEEK`` turns per figure and ``TURNS_PER_WEEK`` in the whole world per
rolling seven days; the most salient triggers win (ties by roster order).  Counters live in
``world.macro`` so resets reproduce them.  The model may return zero to two actions from the role's
permitted list; the engine validates and applies them (same path as a person's actions), so the model
never sets an effect size.  Figures controlled by a person never act on their own.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from jahan.actors.actions import Claim
from jahan.actors.engine import display_name
from jahan.actors.policy import PolicyError, compile_policy
from jahan.llm.provider import LLMClient, extract_json
from jahan.llm.recordings import MissingRecording
from jahan.model.enums import TOPIC_NAMES, EventKind, OriginKind, Topic
from jahan.model.mentions import mentions
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.world import World
from jahan.query.social_feed import recent_posts_lines, trending_lines
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.store.eventlog import EventLog

TURNS_PER_NOTABLE_PER_WEEK = 2
TURNS_PER_WEEK = 20
AWARE_TRIGGER = 0.2  # HEURISTIC share of the figure's country aware of an item
# HEURISTIC: a fresh headline this salient, on a role topic and pushed in the figure's country, is a trigger.
# Unlabelled feeds give 0.75 - 0.05 * rank, so roughly the top two stories of each feed qualify.
HEADLINE_TRIGGER = 0.7
HEADLINE_ORIGINS = {int(OriginKind.WEB), int(OriginKind.USER_FEED), int(OriginKind.SCENARIO)}
ECONOMIC_ROLES = {"head_of_state", "finance_minister", "central_bank_governor", "business_leader"}
ROLE_TOPICS: dict[str, set[str]] = {
    "head_of_state": {"politics", "economy", "world", "health"},
    "supreme_leader": {"politics", "religion", "world"},
    "finance_minister": {"economy"},
    "central_bank_governor": {"economy"},
    "foreign_minister": {"world", "politics"},
    "legislative_speaker": {"politics"},
    "business_leader": {"tech", "economy"},
    "opinion_leader": set(TOPIC_NAMES),
}

AUTONOMY_SYSTEM = f"""You decide what a real public figure does next inside a research society simulator.
Everything you produce is shown as {SIMULATED_LABEL}; it is not a real statement.
Choose zero, one or two actions from the permitted list.  Act only where the figure's documented public
positions (with sources) clearly imply a response to the situation; otherwise choose no action.  Never invent
private information, quotes or new positions.  Effect sizes are decided by the simulator, not by you.
Answer with JSON only: {{"actions": [{{"kind": "<permitted kind>", "params": {{...}},
"statement": "<one plain sentence, only for public_statement or directive>", "topic": "<topic>",
"stance": -1 | -0.5 | 0 | 0.5 | 1}}]}}  (an empty list means no action)."""


def _state(world: World) -> dict[str, Any]:
    st: dict[str, Any] = world.macro.setdefault("autonomy", {"turns": [], "seen": []})
    return st


def triggers(world: World) -> list[dict[str, Any]]:
    """Notables due a decision turn at the start of ``world.day`` (caps applied)."""
    day = world.day
    yesterday = day - 1
    st = _state(world)
    seen = set(st["seen"])
    pop = world.pop
    n = pop.n
    w = pop["pop_weight"].astype(np.float64) * pop["alive"]
    labels = world.macro.get("item_labels", {})
    items = world.items
    slots = np.flatnonzero(world.beliefs.slot_item >= 0)
    live = [int(world.beliefs.slot_item[s]) for s in slots]
    # population-weighted aware share per (country, live item), computed once
    loc = pop["location_country"].astype(np.int64)
    wc_all = np.zeros((world.n_countries, n))
    wc_all[loc, np.arange(n)] = w
    tot = np.maximum(wc_all.sum(axis=1), 1e-12)
    aware = world.beliefs.m["aware"][:n][:, slots].astype(np.float64)
    shares = (wc_all @ aware) / tot[:, None]
    pushed_to = {item: _countries_pushed(world, item) for item in live}
    product_items = {int(x) for x in world.products["item_id"]}
    cands: list[dict[str, Any]] = []
    for col, entry in enumerate(world.notables):
        aid = int(entry["agent_id"])
        row = world.agent_row(aid)
        if not pop["alive"][row] or pop["controlled_by"][row] >= 0:
            continue
        role = str(entry["role"])
        c = int(entry["country"])
        topics = {int(TOPIC_NAMES[t]) for t in ROLE_TOPICS.get(role, set())}
        reasons: list[tuple[float, str, str]] = []  # (salience, key, text)
        for j, item in enumerate(live):
            if int(items["origin_id"][item]) == aid and int(items["origin_kind"][item]) == OriginKind.NOTABLE:
                continue
            key = f"{col}:{item}"
            if key in seen:
                continue
            label = str(labels.get(str(item), f"item {item}"))
            share = float(shares[c, j])
            created = int(items["created_day"][item])
            in_country = c in pushed_to[item]
            origin = int(items["origin_kind"][item])
            salience = float(items["salience"][item])
            if created == yesterday and origin == OriginKind.MACRO and role in ECONOMIC_ROLES and in_country:
                reasons.append((0.8, key, f"market news in your country: {label}"))
            elif created == yesterday and _named(world, entry, item, label):
                reasons.append((0.9, key, f"you were named: {label}"))
            elif (
                created == yesterday
                and origin in HEADLINE_ORIGINS
                and item not in product_items
                and int(items["topic"][item]) in topics
                and salience >= HEADLINE_TRIGGER
                and in_country
            ):
                reasons.append((0.8 * salience, key, f"top headline in your country: {label}"))
            elif int(items["topic"][item]) in topics and share >= AWARE_TRIGGER:
                reasons.append(
                    (float(items["salience"][item]) * share, key, f"{share:.0%} of your country now knows: {label}")
                )
        every = world.roles[role].meeting_every_days if role in world.roles else None
        if every and day > 0 and day % every == 0:
            reasons.append((0.5, f"{col}:meeting:{day}", "your scheduled policy meeting is today"))
        if reasons:
            reasons.sort(key=lambda r: -r[0])
            cands.append({"col": col, "agent": aid, "salience": reasons[0][0], "reasons": reasons})
    recent = [t for t in st["turns"] if t["day"] > day - 7]
    per: dict[int, int] = {}
    for t in recent:
        per[t["col"]] = per.get(t["col"], 0) + 1
    budget = TURNS_PER_WEEK - len(recent)
    chosen = []
    for cand in sorted(cands, key=lambda x: (-x["salience"], x["col"])):
        if budget <= 0:
            break
        if per.get(cand["col"], 0) >= TURNS_PER_NOTABLE_PER_WEEK:
            continue
        chosen.append(cand)
        budget -= 1
    for cand in cands:  # a trigger is used once, whether or not the figure got a turn for it
        st["seen"].extend(r[1] for r in cand["reasons"])
    st["seen"] = st["seen"][-5000:]
    return chosen


def _named(world: World, entry: dict[str, Any], item: int, label: str) -> bool:
    """The item names the figure: any roster alias in its text, or an @-mention of one of their handles."""
    aliases = [str(entry["holder_name"]), *[str(a) for a in entry.get("aliases", [])]]
    if mentions(label, aliases):
        return True
    social = world.macro.get("social", {}).get("items", {}).get(str(item), {})
    tagged = {str(m).lstrip("@").casefold() for m in social.get("mentions", [])}
    handles = {a.lstrip("@").casefold() for a in aliases if a.startswith("@")}
    return bool(tagged & handles)


def _countries_pushed(world: World, item: int) -> set[int]:
    rows = np.flatnonzero(world.pushes["item_id"] == item)
    return {int(c) for c in world.channels["country"][world.pushes["channel_id"][rows]]}


def _brief(world: World, cand: dict[str, Any]) -> str:
    entry = world.notables[cand["col"]]
    role = world.roles.get(entry["role"])
    lines = [
        f"Figure: {entry['holder_name']}, {entry['office']} ({world.countries[int(entry['country'])].name}).",
        f"Simulated date: {world.date().isoformat()}.",
        "Documented public positions (with sources):",
    ]
    for s in entry.get("stances", []):
        lines.append(f"- {s['topic']}: {s['position']:+.1f} {s['summary']} ({s['source']})")
    if not entry.get("stances"):
        lines.append("- none documented")
    if entry.get("communication_style"):
        lines.append(f"Communication style (documented): {entry['communication_style']}")
    lines.extend(recent_posts_lines(world, str(entry["key"])))
    lines.extend(trending_lines(world, world.countries[int(entry["country"])].iso3))
    lines.append("Why you are deciding now:")
    lines.extend(f"- {r[2]}" for r in cand["reasons"][:4])
    lines.append("Permitted actions:")
    for kind, pol in role.policies.items() if role else []:
        if pol.param is not None:
            extra = " target=<ISO3>" if pol.scope == "target_country" else ""
            prod = " product=<name>" if pol.effect == "product_price" else ""
            lines.append(f"- {kind}: params {{{pol.param.name}: {pol.param.min:g}..{pol.param.max:g}{extra}{prod}}}")
        elif pol.effect == "product":
            lines.append(f"- {kind}: params {{name, price_usd, quality 0-1, countries}}")
        else:
            lines.append(f"- {kind}: needs statement; params {{audience: national|global}}")
    lines.append(f"Countries in the simulation: {[c.iso3 for c in world.countries]}.")
    lines.append(f"Topics: {sorted(TOPIC_NAMES)}.")
    return "\n".join(lines)


def decide(client: LLMClient, world: World, cand: dict[str, Any], log: EventLog) -> list[Any]:
    """One turn: the model proposes up to two actions; invalid ones are logged as ACTION_REJECTED."""
    raw = client.complete(
        "notable_autonomy",
        AUTONOMY_SYSTEM,
        [{"role": "user", "content": _brief(world, cand)}],
        max_tokens=600,
        purpose="notable_turn",
        sim_day=world.day,
    )
    aid = int(cand["agent"])
    try:
        data = extract_json(raw)
    except ValueError:
        data = None
    acts = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(acts, list):  # audit trail: an unreadable answer is not silently "no action"
        log.add(
            world.day,
            EventKind.ACTION_REJECTED,
            aid,
            -1,
            -1,
            0.0,
            -1,
            f"unreadable model answer (no action taken): {raw[:200]}",
        )
        acts = []
    events: list[Any] = []
    for a in acts[:2]:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind", ""))
        raw_params = a.get("params")
        params: dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}
        claim = None
        if a.get("statement"):
            topic = a.get("topic") if a.get("topic") in TOPIC_NAMES else Topic.POLITICS.name.lower()
            try:
                stance = float(a.get("stance", 0.0))
            except (TypeError, ValueError):
                stance = 0.0
            claim = Claim(headline=str(a["statement"])[:300], topic=str(topic), stance=max(-1.0, min(1.0, stance)))
        try:
            events.extend(compile_policy(world, aid, kind, dict(params), claim=claim, session=-1))
        except (PolicyError, ValueError) as e:
            log.add(world.day, EventKind.ACTION_REJECTED, aid, -1, -1, 0.0, -1, f"{kind}: {e}"[:300])
    return events


def run_turns(client: LLMClient, world: World, log: EventLog) -> int:
    """Start-of-day hook: give due figures a turn and apply their actions now.  Returns turns taken."""
    chosen = triggers(world)
    st = _state(world)
    taken = 0
    for cand in chosen:
        try:
            events = decide(client, world, cand, log)
        except MissingRecording:
            raise  # replay mode: the recorded timeline had this turn; diverging silently is not allowed
        except RuntimeError as e:  # budget share exhausted, provider failure: figures stop acting
            log.add(
                world.day,
                EventKind.ACTION_SKIPPED,
                int(cand["agent"]),
                -1,
                -1,
                0.0,
                -1,
                f"notable turn not taken: {e}"[:300],
            )
            st.setdefault("stopped", str(e))
            break
        st["turns"].append({"day": world.day, "col": cand["col"], "n_actions": len(events)})
        taken += 1
        if events:
            sc = Scenario(prompt=f"{display_name(world, world.agent_row(int(cand['agent'])))} acts", events=events)
            apply_scenario(world, sc, log, -1)
        log.add(
            world.day,
            EventKind.NOTABLE_ACTION,
            int(cand["agent"]),
            -1,
            -1,
            float(len(events)),
            -1,
            json.dumps({"turn": True, "reasons": [r[2] for r in cand["reasons"][:3]]})[:300],
        )
    st["turns"] = [t for t in st["turns"] if t["day"] > world.day - 30]
    return taken
