"""Counterfactual experiments: K seeds x {baseline, treated} from the pre-prompt snapshot.

Baseline and treated share the simulation seed of each replicate, so every draw
not affected by the scenario is identical (common random numbers) and paired
differences isolate the scenario's effect.  Workers receive only paths and
seeds (Windows uses spawn; nothing large is pickled).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from jahan.model.enums import EventKind
from jahan.rng import derive_seed
from jahan.scenario.inject import InjectError, apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import Recorder, run_in_memory
from jahan.sim.tick import run_day
from jahan.sim.trust import effective_trust
from jahan.store.eventlog import EventLog
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot


def run_branch(
    snapshot: str,
    scenario_path: str | None,
    prompt_id: int,
    seed: int,
    horizon: int,
    out_dir: str,
    later: list[tuple[int, str, int]] | None = None,
) -> str:
    """One branch.  ``later``: (day, scenario path, prompt id) applied at the start of those days, in order."""
    world = load_snapshot(Path(snapshot))
    world.sim_seed = seed
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trust_before = effective_trust(world, slice(None), world.config.params)
    pre_log = EventLog()
    if scenario_path:
        apply_scenario(world, Scenario.load(Path(scenario_path)), pre_log, prompt_id)
    if later:
        rec = Recorder(world.day)
        queue = sorted(later, key=lambda x: (x[0], x[2]))
        for _ in range(horizon):
            while queue and queue[0][0] <= world.day:
                _, path, pid = queue.pop(0)
                try:
                    apply_scenario(world, Scenario.load(Path(path)), rec.events, pid)
                except InjectError as e:  # it depends on something this branch never created
                    rec.events.add(
                        world.day, EventKind.ACTION_SKIPPED, -1, pid, -1, 0.0, -1, f"prompt {pid}: {e}"[:300]
                    )
            run_day(world, rec.events, rec.metrics)
    else:
        rec = run_in_memory(world, horizon)
    events = EventLog()
    events.extend_table(pre_log.table())  # injection events of the treated branch come first on day 0
    events.extend_table(rec.events.table())
    pq.write_table(rec.metrics.table(), out / "metrics.parquet")
    pq.write_table(events.table(), out / "events.parquet")
    n = world.pop.n
    bel = world.beliefs.m
    arrays: dict[str, Any] = {"slot_item": world.beliefs.slot_item.copy()}
    for k in ("aware", "belief", "heard_from", "adopted_day", "satisfaction", "first_day"):
        arrays[k] = bel[k][:n].copy()
    trust_after = effective_trust(world, slice(None), world.config.params)
    delta = np.zeros(trust_after.shape[0], dtype=np.float32)  # ties created during the branch start at 0
    nb = trust_before.shape[0]
    delta[:nb] = trust_after[:nb] - trust_before
    arrays["trust_delta"] = delta
    arrays["edge_src"] = world.edges["src"].copy()
    arrays["edge_dst"] = world.edges["dst"].copy()
    np.savez(out / "final.npz", **arrays)
    (out / "branch.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "horizon": horizon,
                "treated": bool(scenario_path),
                "applied_prompts": [x[2] for x in later or []],
                "end_day": world.day,
            }
        ),
        encoding="utf-8",
    )
    return str(out)


def session_dir(rundir: RunDir, session_id: int) -> Path:
    return rundir.root / "experiments" / f"session_{session_id:04d}"


def run_session_experiment(rundir: RunDir, session_id: int, seeds: int, horizon: int | None, workers: int = 4) -> Path:
    """A person's session versus autopilot.

    Both branches start from the snapshot before the session's first turn and re-apply every later ledger
    prompt on its day (news, feeds, other users' prompts), except that the baseline skips this session's
    turns.  Public figures' autonomous decisions are not re-run inside branches (that would need new LLM
    calls); whatever was already scheduled in the starting snapshot still happens in both.
    """
    led = sorted(rundir.ledger(), key=lambda e: e["prompt_id"])
    mine = [e for e in led if e.get("session") == session_id]
    if not mine:
        raise KeyError(f"session {session_id} has no turns in the ledger")
    first = mine[0]
    start = int(first["day"])
    last = int(mine[-1]["day"])
    h = horizon or (last - start + 30)
    entries = [e for e in led if e["prompt_id"] >= first["prompt_id"] and int(e["day"]) < start + h]
    all_later = [(int(e["day"]), str(rundir.root / e["scenario_file"]), int(e["prompt_id"])) for e in entries]
    base_later = [x for x, e in zip(all_later, entries, strict=True) if e.get("session") != session_id]
    snap = rundir.root / first["snapshot_before"]
    base_seed = int(first["world_seed"])
    exp_dir = session_dir(rundir, session_id)
    tasks = []
    for s in range(seeds):
        seed = base_seed if s == 0 else derive_seed(base_seed, s)
        for branch, later in (("baseline", base_later), ("treated", all_later)):
            tasks.append((str(snap), None, -1, seed, h, str(exp_dir / f"seed_{s:02d}" / branch), later))
    if workers <= 1:
        for t in tasks:
            run_branch(*t)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run_branch, *zip(*tasks, strict=True)))
    (exp_dir / "experiment.json").write_text(
        json.dumps(
            {
                "session": session_id,
                "seeds": seeds,
                "horizon": h,
                "start_day": start,
                "session_prompts": [int(e["prompt_id"]) for e in mine],
                "other_prompts": [x[2] for x in base_later],
                "snapshot": first["snapshot_before"],
            }
        ),
        encoding="utf-8",
    )
    return exp_dir


def run_experiment(rundir: RunDir, prompt_id: int, seeds: int, horizon: int | None, workers: int = 4) -> Path:
    entry = rundir.prompt(prompt_id)
    snap = rundir.root / entry["snapshot_before"]
    scen_path = rundir.root / entry["scenario_file"]
    scenario = Scenario.load(scen_path)
    h = horizon or scenario.horizon_days
    base_seed = int(entry["world_seed"])
    exp_dir = rundir.experiment_dir(prompt_id)
    tasks = []
    for s in range(seeds):
        # replicate 0 replays the main timeline's seed, so seed-0 stories match what `jahan agent` shows
        seed = base_seed if s == 0 else derive_seed(base_seed, s)
        for branch in ("baseline", "treated"):
            tasks.append(
                (
                    str(snap),
                    str(scen_path) if branch == "treated" else None,
                    prompt_id,
                    seed,
                    h,
                    str(exp_dir / f"seed_{s:02d}" / branch),
                )
            )
    if workers <= 1:
        for t in tasks:
            run_branch(*t)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run_branch, *zip(*tasks, strict=True)))
    (exp_dir / "experiment.json").write_text(
        json.dumps({"prompt_id": prompt_id, "seeds": seeds, "horizon": h}), encoding="utf-8"
    )
    return exp_dir
