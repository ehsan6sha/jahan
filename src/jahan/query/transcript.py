"""Readable account of one agent's day, rebuilt by deterministic re-simulation.

Per-contact records are not logged (hundreds of thousands per day).  Instead we load the nearest
snapshot at or before the requested day, re-apply any prompts injected in between (from the ledger),
replay to that day, and run the day with a trace on the agent.  The trace lists who they met and what
was shared, believed or doubted.  Rendering is a template; an LLM may rewrite it (recorded).
"""

from __future__ import annotations

from pathlib import Path

from jahan.llm.provider import LLMClient
from jahan.model.enums import EdgeKind
from jahan.model.world import World
from jahan.query.views import full_name
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.tick import run_day
from jahan.store.eventlog import EventLog, MetricsLog
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot

TRANSCRIPT_SYSTEM = """Rewrite the simulator's trace of one person's day as a short, plain narrative
(at most 150 words).  Keep every fact exactly (who met whom, what was shared, whether it was believed);
add nothing else."""


def _nearest_snapshot(rd: RunDir, day: int) -> Path:
    best: tuple[int, Path] | None = None
    for p in rd.snapshots.glob("day_*.npz"):
        d = int(p.stem.split("_")[1])
        if d <= day and (best is None or d > best[0]):
            best = (d, p)
    if best is None:
        raise FileNotFoundError("no periodic snapshot at or before that day")
    return best[1]


def trace_day(rd: RunDir, agent_id: int, day: int) -> tuple[World, list[dict], EventLog]:
    """Re-simulate to ``day`` with a trace on ``agent_id``.  Returns the world after that day, the trace
    rows and that day's event log (which must equal the original timeline's events for the day).

    A periodic snapshot for day S is written when a run reaches S, before any prompt injected on day S,
    so every ledger prompt with day >= S is re-applied at the start of its day, in prompt order.
    """
    snap = _nearest_snapshot(rd, day)
    world = load_snapshot(snap)
    ledger = sorted(rd.ledger(), key=lambda e: (e["day"], e["prompt_id"]))
    rows: list[dict] = []
    day_log = EventLog()
    while world.day <= day:
        log = day_log if world.day == day else EventLog()
        for entry in ledger:
            if entry["day"] == world.day:
                apply_scenario(world, Scenario.load(rd.root / entry["scenario_file"]), log, int(entry["prompt_id"]))
        trace = {agent_id} if world.day == day else None
        run_day(world, log, MetricsLog(), trace, rows if trace else None)
    return world, rows, day_log


def render(world: World, agent_id: int, rows: list[dict], client: LLMClient | None = None) -> str:
    me = full_name(world, world.agent_row(agent_id))
    labels = {int(k): v for k, v in world.macro.get("item_labels", {}).items()}
    lines: list[str] = []
    met = [r for r in rows if r.get("type") == "contact" and r["agent"] == agent_id]
    if met:
        lines.append(
            f"{me} spent time with "
            + ", ".join(
                f"{full_name(world, world.agent_row(r['other']))} ({EdgeKind(r['kind']).name.lower()})" for r in met
            )
            + "."
        )
    else:
        lines.append(f"{me} had no conversations worth noting.")
    for r in (r for r in rows if r.get("type") == "share"):
        what = labels.get(int(r["item"]), f"item {r['item']}")
        if r["sharer"] == agent_id:
            other = full_name(world, world.agent_row(int(r["receiver"])))
            verb = "exaggerated" if r["distorted"] else "mentioned"
            reaction = "took it on board" if r["accepted"] else "was not convinced"
            lines.append(f"{me} {verb} {what} to {other} (view conveyed {r['conveyed']:.2f}); {other} {reaction}.")
        elif r["receiver"] == agent_id:
            other = full_name(world, world.agent_row(int(r["sharer"])))
            reaction = "believed it" if r["accepted"] else "doubted it"
            lines.append(
                f"{other} talked about {what} (view {r['conveyed']:.2f}); {me} {reaction} "
                f"(trust in {other} {r['trust']:.2f})."
            )
    text = "\n".join(lines)
    if client is not None:
        text = client.complete(
            "transcript",
            TRANSCRIPT_SYSTEM,
            [{"role": "user", "content": text}],
            max_tokens=400,
            purpose="transcript",
            sim_day=world.day,
        )
    return text
