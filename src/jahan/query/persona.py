"""Persona interviews: ask one agent, a group or the world how they see something.

Each interview is grounded in a closed-world *mind snapshot* built from the agent's state: profile,
values, the items they actually received (with source and day), what their closest ties think, and
their mood.  The model is told to answer only from that snapshot.  If the question names an item the
agent never encountered, a deterministic "not aware" answer is returned without calling the model.
Real public figures are answered as clearly labelled simulations of their sourced public positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jahan.llm.provider import LLMClient
from jahan.model.enums import Big5, EdgeKind, InfoTier, Marital, Topic, ValueDim
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.world import World
from jahan.query.scope import evaluate
from jahan.query.views import age_of, full_name
from jahan.rng import Stream, u01
from jahan.scenario.report import ISCO_LABELS
from jahan.sim.trust import effective_trust

PERSONA_SYSTEM = """You are role-playing one simulated person in the Jahan society simulator.
Answer the interviewer in the first person, in plain English, in at most 120 words.
Use ONLY the facts in the person's mind snapshot.  If the snapshot does not say something, the person
does not know it: say so in character.  Do not invent events, numbers, people or news.
Reflect the person's views (0 = very negative, 1 = very positive), mood, values, education and language.
Speak as this person would to a stranger: never mention a snapshot, a simulator, tracked items, day numbers
or scores.  Say "about two weeks ago" rather than "day 21", and describe feelings in words, not numbers.
"""

NOTABLE_SYSTEM = """You are producing a clearly labelled SIMULATION of how a real public figure might respond,
for a research simulator.  Base the answer ONLY on the documented public positions listed (with sources)
and the world facts given.  Do not invent quotes, private information or new positions; when the positions
do not cover the question, say the figure has not addressed it publicly.  At most 120 words.
"""


@dataclass
class Interview:
    agent_id: int
    name: str
    answer: str
    used_llm: bool
    snapshot: str


def _item_labels(world: World) -> dict[int, str]:
    """item id -> human label (product names, headlines) for items with a belief slot."""
    labels: dict[int, str] = {}
    for p, name in enumerate(world.product_names):
        labels[int(world.products["item_id"][p])] = name
    for e in world.macro.get("item_labels", {}).items():
        labels[int(e[0])] = str(e[1])
    return labels


def _ago(days: int) -> str:
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"about {round(days / 7)} weeks ago"
    return f"about {round(days / 30)} months ago"


def _feeling(view: float) -> str:
    for limit, words in ((0.2, "very negative"), (0.4, "negative"), (0.6, "mixed"), (0.8, "positive")):
        if view < limit:
            return words
    return "very positive"


def mind_snapshot(world: World, agent_id: int, labels: dict[int, str] | None = None, top_ties: int = 5) -> str:
    i = world.agent_row(agent_id)
    pop = world.pop
    spec = world.countries[int(pop["country"][i])]
    labels = labels if labels is not None else _item_labels(world)
    lines = [
        f"Name: {full_name(world, i)}. {('Woman' if pop['sex'][i] == 0 else 'Man')}, {age_of(world, i)}, lives in "
        f"{world.countries[int(pop['country'][i])].name} ({'city' if pop['region'][i] else 'countryside'}); "
        f"currently in {world.countries[int(pop['location_country'][i])].name}.",
        f"Marital status: {Marital(int(pop['marital'][i])).name.lower()}; children: {int(pop['n_children'][i])}.",
        f"Work: {ISCO_LABELS.get(int(pop['occupation'][i]), 'not employed')}; monthly income about "
        f"{float(pop['income_month'][i]):,.0f} {spec.currency}.",
        f"Languages: {', '.join(world.languages[int(x)] for x in pop['lang'][i] if x >= 0)}.",
        f"Follows news: {InfoTier(int(pop['info_tier'][i])).name.lower().replace('_', ' ')}; "
        f"{'online' if pop['internet'][i] else 'not online'}.",
    ]
    vals = pop["values"][i]
    top_v = np.argsort(-vals)[:3]
    lines.append("Cares most about: " + ", ".join(ValueDim(int(v)).name.lower() for v in top_v) + ".")
    b5 = pop["big5"][i]
    traits = [t.name.lower() for t in Big5 if abs(float(b5[t])) > 1.0]
    if traits:
        lines.append(
            "Notable traits: " + ", ".join(f"{'high' if b5[Big5[t.upper()]] > 0 else 'low'} {t}" for t in traits) + "."
        )
    lines.append(f"Economic mood: {float(pop['sentiment'][i]):+.2f} (-1 gloomy, +1 upbeat).")
    bel = world.beliefs.m
    known = []
    for slot in np.flatnonzero(world.beliefs.slot_item >= 0):
        item = int(world.beliefs.slot_item[slot])
        if not bel["aware"][i, slot]:
            continue
        hf = int(bel["heard_from"][i, slot])
        if hf >= 0:
            src = full_name(world, world.agent_row(hf))
        elif hf <= -2:
            src = f"the channel {world.channel_names[-hf - 2]}"
        else:
            src = "conversations"
        adopted = int(bel["adopted_day"][i, slot]) >= 0
        label = labels.get(item, f"item {item}")
        first = int(bel["first_day"][i, slot])
        view = float(bel["belief"][i, slot])
        known.append(
            f"- {label}: first heard {_ago(world.day - first)} (day {first}) from {src}; feels {_feeling(view)} "
            f"about it (view {view:.2f})" + ("; bought it" if adopted else "")
        )
    lines.append(
        "Things this person has heard about:"
        if known
        else "This person has not heard any of the news, products or rumours followed in this world."
    )
    lines.extend(known)
    e = world.edges
    rows = np.flatnonzero((e["src"] == agent_id) & e["active"])
    if rows.size:
        t = effective_trust(world, rows, world.config.params)
        order = np.argsort(-t, kind="stable")[:top_ties]
        lines.append("Closest people (trust 0-1):")
        for r_i in order:
            other = int(e["dst"][rows[r_i]])
            lines.append(
                f"- {full_name(world, world.agent_row(other))} ({EdgeKind(int(e['kind'][rows[r_i]])).name.lower()}), "
                f"trust {t[r_i]:.2f}"
            )
    return "\n".join(lines)


def _unknown_mentions(world: World, i: int, question: str, labels: dict[int, str]) -> list[str]:
    q = question.lower()
    bel = world.beliefs.m
    missing = []
    for item, label in labels.items():
        if label and label.lower() in q:
            slot = int(world.items["slot"][item])
            if slot < 0 or not bel["aware"][i, slot]:
                missing.append(label)
    return missing


def interview(client: LLMClient | None, world: World, agent_id: int, question: str) -> Interview:
    i = world.agent_row(agent_id)
    labels = _item_labels(world)
    name = full_name(world, i)
    if world.pop["is_notable"][i] and world.notables:
        entry = world.notables[int(world.pop["role_id"][i])]
        positions = "\n".join(
            f"- [{s['topic']}] {s['summary']} (source: {s['source']})" for s in entry.get("stances", [])
        )
        snap = (
            f"Figure: {entry['holder_name']}, {entry['office']} (verified {entry['as_of']}).\n"
            f"Documented public positions:\n{positions or '- none recorded'}"
        )
        if client is None:
            return Interview(agent_id, name, f"{SIMULATED_LABEL} (no LLM configured)", False, snap)
        ans = client.complete(
            "persona",
            NOTABLE_SYSTEM,
            [{"role": "user", "content": f"{snap}\n\nQuestion: {question}"}],
            max_tokens=300,
            purpose="notable_interview",
            sim_day=world.day,
        )
        return Interview(agent_id, name, f"{SIMULATED_LABEL} {ans.strip()}", True, snap)
    missing = _unknown_mentions(world, i, question, labels)
    snap = mind_snapshot(world, agent_id, labels)
    if missing:
        return Interview(agent_id, name, f"I haven't heard about {', '.join(missing)}.", False, snap)
    if client is None:
        return Interview(agent_id, name, "(no LLM configured; showing the snapshot only)", False, snap)
    ans = client.complete(
        "persona",
        PERSONA_SYSTEM,
        [{"role": "user", "content": f"Mind snapshot:\n{snap}\n\nInterviewer: {question}"}],
        max_tokens=300,
        purpose="persona_interview",
        sim_day=world.day,
    )
    return Interview(agent_id, name, ans.strip(), True, snap)


def sample_group(world: World, expr: str, k: int, salt: int = 0) -> list[int]:
    mask = evaluate(world, expr) & world.pop["alive"] & ~world.pop["is_notable"]
    ids = world.pop["id"][mask]
    if ids.size == 0:
        return []
    rank = u01(world.sim_seed, Stream.SAMPLE, world.day + salt, ids)
    return [int(x) for x in ids[np.argsort(rank, kind="stable")[:k]]]


def group_views(world: World, expr: str, item: int | None) -> str:
    """Numbers for a group (no LLM): awareness, mean view, adoption among the whole group."""
    mask = evaluate(world, expr) & world.pop["alive"] & ~world.pop["is_notable"]
    n = int(mask.sum())
    if n == 0:
        return "No agents match."
    w = world.pop["pop_weight"].astype(np.float64)[mask]
    out = [f"{n} agents match (representing about {w.sum():,.0f} people)."]
    if item is not None:
        slot = int(world.items["slot"][item])
        if slot >= 0:
            bel = world.beliefs.m
            aware = bel["aware"][: world.pop.n, slot][mask] == 1
            out.append(f"Aware: {100 * np.average(aware, weights=w):.1f}%.")
            if aware.any():
                out.append(
                    "Mean view among the aware: "
                    f"{np.average(bel['belief'][: world.pop.n, slot][mask][aware], weights=w[aware]):.2f}."
                )
            adopted = bel["adopted_day"][: world.pop.n, slot][mask] >= 0
            out.append(f"Adopted: {100 * np.average(adopted, weights=w):.1f}%.")
    out.append(f"Mean economic mood: {np.average(world.pop['sentiment'][mask], weights=w):+.2f}.")
    return " ".join(out)


__all__ = ["Interview", "interview", "mind_snapshot", "sample_group", "group_views", "Topic"]
