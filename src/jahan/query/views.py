"""Human-readable views of agents: list rows and a full profile card."""

from __future__ import annotations

import numpy as np

from jahan.model.enums import N_TOPICS, Big5, EdgeKind, EmpStatus, InfoTier, Marital, Sex, Topic, ValueDim
from jahan.model.notables import SIMULATED_LABEL
from jahan.model.world import World
from jahan.scenario.report import ISCO_LABELS
from jahan.sim.availability import compute_masks
from jahan.sim.trust import effective_trust

ISCED_LABELS = {
    0: "early childhood / none",
    1: "primary",
    2: "lower secondary",
    3: "upper secondary",
    4: "post-secondary non-tertiary",
    5: "short-cycle tertiary",
    6: "bachelor",
    7: "master",
    8: "doctorate",
}


def full_name(world: World, i: int) -> str:
    pop = world.pop
    custom = world.macro.get("custom_names", {}).get(str(int(pop["id"][i])))
    if custom:
        return str(custom)
    if pop["is_notable"][i] and 0 <= int(pop["role_id"][i]) < len(world.notables):
        return str(world.notables[int(pop["role_id"][i])]["holder_name"])
    nt = world.names[int(pop["country"][i])]
    return nt.display(int(pop["first_name_id"][i]), int(pop["last_name_id"][i]))


def notable_card(world: World, i: int) -> list[str]:
    """Header lines for a real public figure: office, verification, approval, sourced stances."""
    e = world.notables[int(world.pop["role_id"][i])]
    out = [
        f"  Office: {e['office']}  (real person; everything Jahan generates for them is {SIMULATED_LABEL})",
        f"  Verified {e['as_of']}: {e['verified']}",
        f"  Visibility: about 10^{e['visibility']:.1f} people; approval prior {e['approval']['value']:.2f} "
        f"({e['approval']['source']})",
    ]
    for s in e.get("stances", []):
        out.append(f"  Stance [{s['topic']}] {s['position']:+.1f}: {s['summary']} <{s['source']}>")
    return out


def age_of(world: World, i: int) -> int:
    return int((world.day - int(world.pop["birth_day"][i])) / 365.25)


def list_row(world: World, i: int) -> str:
    pop = world.pop
    iso = world.countries[int(pop["country"][i])].iso3
    sex = "F" if int(pop["sex"][i]) == Sex.F else "M"
    occ = (
        ISCO_LABELS.get(int(pop["occupation"][i]), "?")
        if int(pop["emp_status"][i]) == EmpStatus.EMPLOYED
        else EmpStatus(int(pop["emp_status"][i])).name.lower()
    )
    return f"{int(pop['id'][i]):>7}  {full_name(world, i):<24} {sex} {age_of(world, i):>3}  {iso}  {occ}"


def _hours(mask: int) -> str:
    hrs = [h for h in range(24) if (mask >> h) & 1]
    if not hrs:
        return "none"
    spans = []
    start = prev = hrs[0]
    for h in hrs[1:] + [None]:
        if h is None or h != prev + 1:
            spans.append(f"{start:02d}:00-{(prev + 1) % 24:02d}:00")
            if h is not None:
                start = h
        if h is not None:
            prev = h
    return ", ".join(spans)


def agent_card(world: World, agent_id: int, top_ties: int = 10) -> str:
    i = world.agent_row(agent_id)
    pop = world.pop
    c = int(pop["country"][i])
    spec = world.countries[c]
    lines: list[str] = []
    w = lines.append
    w(f"{full_name(world, i)}  (agent {agent_id}){'  [deceased]' if not pop['alive'][i] else ''}")
    if pop["is_notable"][i] and world.notables:
        lines.extend(notable_card(world, i))
        return "\n".join(lines)
    w(
        f"  {('Female' if int(pop['sex'][i]) == Sex.F else 'Male')}, {age_of(world, i)}, {spec.name}, "
        f"{'urban' if int(pop['region'][i]) else 'rural'}, UTC{int(pop['tz_offset_min'][i]) / 60:+.1f}"
    )
    eth = spec.ethnicity_codes()[int(pop["ethnicity"][i])] if pop["ethnicity"][i] >= 0 else "?"
    rel = spec.religion_codes()[int(pop["religion"][i])] if pop["religion"][i] >= 0 else "?"
    w(f"  Ethnicity: {eth}; religion: {rel}")
    langs = [world.languages[int(x)] for x in pop["lang"][i] if x >= 0]
    w(f"  Languages: {', '.join(langs)}")
    w(
        f"  Marital: {Marital(int(pop['marital'][i])).name.lower()}"
        + (
            f" to {full_name(world, int(pop['spouse_id'][i]))} ({int(pop['spouse_id'][i])})"
            if pop["spouse_id"][i] >= 0
            else ""
        )
    )
    w(
        f"  Education: {ISCED_LABELS[int(pop['education'][i])]}; "
        f"status: {EmpStatus(int(pop['emp_status'][i])).name.lower()}"
        + (f", {ISCO_LABELS[int(pop['occupation'][i])]}" if pop["occupation"][i] >= 0 else "")
    )
    w(f"  Income: {float(pop['income_month'][i]):,.0f} {spec.currency}/month")
    w(
        f"  Information: {InfoTier(int(pop['info_tier'][i])).name.lower()}"
        + (", online" if pop["internet"][i] else ", offline")
        + (f", focus on {Topic(int(pop['focus_topic'][i])).name.lower()}" if pop["focus_topic"][i] >= 0 else "")
    )
    b5 = pop["big5"][i]
    w("  Personality (z): " + ", ".join(f"{t.name.lower()} {b5[t]:+.1f}" for t in Big5))
    vals = pop["values"][i]
    top_v = np.argsort(-vals)[:3]
    w("  Cares most about: " + ", ".join(f"{ValueDim(int(v)).name.lower()} ({vals[v]:.2f})" for v in top_v))
    ints = pop["interests"][i]
    top_i = np.argsort(-ints)[:3]
    w("  Interests: " + ", ".join(f"{Topic(int(t)).name.lower()} ({ints[t] * N_TOPICS:.1f}x)" for t in top_i))
    w(
        f"  Risk tolerance {float(pop['risk_tolerance'][i]):.2f}; honesty {float(pop['honesty'][i]):.2f}; "
        f"aggression {float(pop['aggression'][i]):.2f}; discernment {float(pop['discernment'][i]):.2f}; "
        f"wants children {float(pop['wants_kids'][i]):.2f}"
    )
    w(f"  Visibility: about 10^{float(pop['visibility'][i]):.1f} people; reputation {float(pop['global_rep'][i]):.2f}")
    impair = []
    if pop["vision_impaired"][i]:
        impair.append("vision " + ("severe" if pop["vision_impaired"][i] == 2 else "moderate"))
    if pop["hearing_impaired"][i]:
        impair.append("hearing " + ("severe" if pop["hearing_impaired"][i] == 2 else "moderate"))
    w(
        f"  Health {float(pop['health'][i]):.2f}; attractiveness z {float(pop['attractiveness_z'][i]):+.1f}"
        + (f"; impairments: {', '.join(impair)}" if impair else "")
    )
    masks = compute_masks(world, world.config.params)
    w(
        f"  Today (local time): awake {_hours(int(masks.awake_local[i]))}; "
        f"work/school {_hours(int(masks.work_local[i]))}"
    )

    hh = int(pop["household_id"][i])
    members = [int(m) for m in np.flatnonzero(pop["household_id"] == hh) if m != i]
    if members:
        w("  Household: " + "; ".join(f"{full_name(world, m)} ({age_of(world, m)})" for m in members))
    e = world.edges
    rows = np.flatnonzero((e["src"] == agent_id) & e["active"])
    if rows.size:
        t = effective_trust(world, rows, world.config.params)
        order = np.argsort(-t, kind="stable")[:top_ties]
        w(f"  Circle ({rows.size} ties; top {min(top_ties, rows.size)} by trust):")
        for r_i in order:
            r = rows[r_i]
            other = int(e["dst"][r])
            w(
                f"    {EdgeKind(int(e['kind'][r])).name.lower():<12} {full_name(world, other):<24} "
                f"trust {t[r_i]:.2f}  (evidence +{float(e['alpha'][r]):.1f}/-{float(e['beta'][r]):.1f})"
            )
    bel = world.beliefs.m
    tracked = np.flatnonzero(world.beliefs.slot_item >= 0)
    for slot in tracked:
        item = int(world.beliefs.slot_item[slot])
        if not bel["aware"][i, slot]:
            w(f"  Item {item}: not aware")
            continue
        hf = int(bel["heard_from"][i, slot])
        src = full_name(world, hf) if hf >= 0 else (world.channel_names[-hf - 2] if hf <= -2 else "unknown")
        ad = int(bel["adopted_day"][i, slot])
        w(
            f"  Item {item}: belief {float(bel['belief'][i, slot]):.2f}, "
            f"first heard day {int(bel['first_day'][i, slot])}, "
            f"latest source {src}" + (f", adopted day {ad}" if ad >= 0 else "")
        )
    return "\n".join(lines)
