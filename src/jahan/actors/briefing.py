"""What a controlled agent learned yesterday, and in-character replies to their messages.

The briefing itself is deterministic text built from world state.  Replies (LLM) are written only for
one-to-one messages, from the recipient's closed-world mind snapshot, and are told how the recipient
actually took the message (accepted or not, trust in the sender, view before and after) so the words
match the numbers.  Replies are recorded like every LLM answer; they never change the world.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jahan.actors.engine import display_name, is_real_person, notable_entry
from jahan.llm.provider import LLMClient
from jahan.model.enums import EdgeKind, OriginKind
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.world import World
from jahan.query.persona import mind_snapshot
from jahan.scenario.posts import people, posts_of
from jahan.sim.params import SimParams
from jahan.sim.trust import parasocial_trust

REPLY_SYSTEM = """You are role-playing one simulated person in the Jahan society simulator.
Someone you know has just told you something.  Reply to them in the first person, in plain English, in at
most 60 words.  Use ONLY the facts in your mind snapshot and in the message.  Your reply must match how you
actually took the message, as stated (believed it or not, how much you trust them).  Do not invent events,
numbers, people or news.  Talk like a person, not a report: never mention a snapshot, a simulator, day numbers
or trust and belief scores."""


def _source(world: World, hf: int) -> str:
    if hf >= 0:
        return display_name(world, world.agent_row(hf))
    if hf <= -2 and -hf - 2 < len(world.channel_names):
        return f"the channel {world.channel_names[-hf - 2]}"
    return "conversations"


def briefing(world: World, agent_id: int) -> str:
    """Text for the start of day ``world.day``: what happened to this agent on the day before."""
    pop = world.pop
    row = world.agent_row(agent_id)
    yesterday = world.day - 1
    loc = int(pop["location_country"][row])
    off = int(pop["tz_offset_min"][row])
    lines = [
        f"=== {display_name(world, row)} (agent {agent_id}), {world.date().isoformat()} (day {world.day}), "
        f"in {world.countries[loc].name}, UTC{off / 60:+g} ===",
    ]
    if is_real_person(world, row):
        lines.append(f"You are playing a real public figure; everything you post is shown as {SIMULATED_LABEL}.")
    labels = world.macro.get("item_labels", {})
    bel = world.beliefs.m
    news = []
    for slot in np.flatnonzero(world.beliefs.slot_item >= 0):
        item = int(world.beliefs.slot_item[slot])
        if bel["aware"][row, slot] and int(bel["first_day"][row, slot]) == yesterday:
            label = labels.get(str(item)) or _product_label(world, item)
            src = _source(world, int(bel["heard_from"][row, slot]))
            news.append(f"- {label}  (from {src}; your view {float(bel['belief'][row, slot]):.2f})")
    lines.append("New to you yesterday:" if news else "Nothing new reached you yesterday.")
    lines.extend(news)
    mine = [m for m in world.macro.get("messages", {}).values() if m["actor"] == agent_id and m["day"] >= yesterday]
    if mine:
        lines.append("Your messages:")
        for m in mine:
            who = display_name(world, world.agent_row(int(m["target"])))
            if m.get("delivered_day") is None:
                lines.append(f'- to {who}: "{m["headline"]}" (they were asleep; it reaches them today)')
            else:
                took = "believed you" if m.get("accepted") else "was not convinced"
                lines.append(
                    f'- to {who}: "{m["headline"]}" ({who} {took}; trust in you {m["trust"]:.2f}; their view '
                    f"{m.get('belief_before', 0.5):.2f} -> {m.get('belief_after', 0.5):.2f})"
                )
    posted = posts_of(world, agent_id, world.day - 7)
    if posted:
        lines.append("Your posts (last 7 days):")
        for p in posted[-5:]:
            where = {"circle": "your circle", "community": "your community", "national": "the whole country"}.get(
                p["audience"], "the world"
            )
            reposts = f"; reposted {p['reposts']:,} times" if p.get("reposts") else ""
            lines.append(
                f'- day {p["day"]}, to {where}: "{p["headline"]}" reached {p["reached"]:,} simulated people '
                f"({people(p['reached_people'])} real people), {p['believed']:,} convinced by you{reposts}"
            )
    figures = []
    for i in range(world.items.n):
        if int(world.items["created_day"][i]) == yesterday and int(world.items["origin_kind"][i]) == OriginKind.NOTABLE:
            pushed = np.flatnonzero(world.pushes["item_id"] == i)
            if np.any(world.channels["country"][world.pushes["channel_id"][pushed]] == loc):
                figures.append(f"- {labels.get(str(i), f'item {i}')}")
    if figures:
        lines.append("Public figures in the news here:")
        lines.extend(figures[:5])
    e = world.edges
    rows = np.flatnonzero((e["dst"] == agent_id) & e["active"])
    lines.append(f"People who listen to you personally: {rows.size} (use `circle` to list them).")
    if notable_entry(world, row) is not None:
        home = int(pop["country"][row])
        public = np.flatnonzero(pop["alive"] & ~pop["is_notable"] & (pop["country"] == home))
        trust = parasocial_trust(world, public, int(pop["role_id"][row]), SimParams())
        lines.append(
            f"Your public in {world.countries[home].name}: {int((trust >= 0.5).sum()):,} of {public.size:,} simulated "
            "people trust you at least half-way; community, national and global posts reach them through the media."
        )
    return "\n".join(lines)


def _product_label(world: World, item: int) -> str:
    for p, name in enumerate(world.product_names):
        if int(world.products["item_id"][p]) == item:
            return name
    return f"item {item}"


def circle_members(world: World, agent_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """People who listen to this agent, most trusting first: id, name, tie kind, trust in the agent."""
    from jahan.sim.trust import effective_trust

    e = world.edges
    rows = np.flatnonzero((e["dst"] == agent_id) & e["active"])
    rows = rows[world.pop["alive"][e["src"][rows]]]
    if rows.size == 0:
        return []
    t = effective_trust(world, rows, world.config.params)
    order = np.argsort(-t, kind="stable")[:limit]
    return [
        {
            "id": int(e["src"][rows[k]]),
            "name": display_name(world, world.agent_row(int(e["src"][rows[k]]))),
            "tie": EdgeKind(int(e["kind"][rows[k]])).name.lower(),
            "trust": float(t[k]),
        }
        for k in order
    ]


def circle_listing(world: World, agent_id: int, limit: int = 25) -> str:
    members = circle_members(world, agent_id)
    if not members:
        return "Nobody listens to this agent."
    out = [f"{len(members)} people listen to agent {agent_id} (their trust in you):"]
    for m in members[:limit]:
        out.append(f"  {m['id']:>7}  {m['name']}  ({m['tie']}, trust {m['trust']:.2f})")
    return "\n".join(out)


def pending_replies(world: World, agent_id: int) -> list[dict[str, Any]]:
    """Messages from this agent delivered yesterday (each gets at most one reply)."""
    return [
        m
        for m in world.macro.get("messages", {}).values()
        if m["actor"] == agent_id and m.get("delivered_day") == world.day - 1
    ]


def reply(client: LLMClient, world: World, message: dict[str, Any]) -> str:
    target = world.agent_row(int(message["target"]))
    sender = display_name(world, world.agent_row(int(message["actor"])))
    e = world.edges
    edge = int(world.find_edges(np.array([target]), np.array([int(message["actor"])]))[0])
    relation = EdgeKind(int(e["kind"][edge])).name.lower() if edge >= 0 else "acquaintance"
    if message.get("accepted"):
        took = (
            f"you believed them; your view moved from {message['belief_before']:.2f} to "
            f"{message['belief_after']:.2f} (0 = very negative, 1 = very positive)"
        )
    else:
        took = f"you were not convinced; your view stayed at about {message.get('belief_after', 0.5):.2f}"
    words = message.get("text") or message["headline"]
    user = (
        f"Mind snapshot:\n{mind_snapshot(world, int(message['target']))}\n\n"
        f'{sender} (your {relation}; you trust them {message["trust"]:.2f} on a 0-1 scale) told you: "{words}"\n'
        f"How you took it: {took}.\nReply to {sender}."
    )
    answer = client.complete(
        "persona",
        REPLY_SYSTEM,
        [{"role": "user", "content": user}],
        max_tokens=200,
        purpose="message_reply",
        sim_day=world.day,
    )
    if is_real_person(world, target):
        answer = f"{SIMULATED_LABEL} {answer}"
    return answer.strip()
