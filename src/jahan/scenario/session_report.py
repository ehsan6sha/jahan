"""Report for a session experiment: what a person's actions changed compared with autopilot."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from jahan.actors.actions import Control, CreateAgent, Decision, Message, Post, PriceChange
from jahan.model.enums import EventKind
from jahan.scenario.experiment import session_dir
from jahan.scenario.report import MetricIndex, _branch, _num, _pct
from jahan.scenario.spec import ProductLaunch, Scenario
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot


def _labels(events_path: Path, pre_labels: dict[str, int]) -> dict[str, int]:
    """label -> item id in one branch (items that existed before plus items created in the branch)."""
    out = dict(pre_labels)
    t = pq.read_table(events_path).to_pydict()
    for k, it, det in zip(t["kind"], t["item"], t["detail"], strict=True):
        if k in (EventKind.ITEM_CREATED, EventKind.ITEM_UPDATED) and it >= 0 and det:
            out.setdefault(str(det), int(it))
    return out


def _action_items(events_path: Path) -> list[int]:
    t = pq.read_table(events_path).to_pydict()
    seen: list[int] = []
    for k, it in zip(t["kind"], t["item"], strict=True):
        if k == EventKind.HUMAN_ACTION and it >= 0 and int(it) not in seen:
            seen.append(int(it))
    return seen


def _describe(ev: object) -> str:
    if isinstance(ev, Message):
        return f'message from {ev.actor} to {ev.target}: "{ev.claim.headline}" (stance {ev.claim.stance:+g})'
    if isinstance(ev, Post):
        return f'post by {ev.actor} to {ev.audience}: "{ev.claim.headline}" (stance {ev.claim.stance:+g})'
    if isinstance(ev, Control):
        return f"{'took control of' if ev.on else 'released'} agent {ev.agent}"
    if isinstance(ev, Decision):
        return f"agent {ev.actor}: {ev.choice} {ev.product}"
    if isinstance(ev, PriceChange):
        return f"price of {ev.product} {ev.pct:+g}%"
    if isinstance(ev, CreateAgent):
        return f"created a person in {ev.country}"
    if isinstance(ev, ProductLaunch):
        return f"launched {ev.name}"
    return type(ev).__name__


def build_session_report(rundir: RunDir, session_id: int) -> Path:
    exp_dir = session_dir(rundir, session_id)
    exp = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    seeds, horizon, start = int(exp["seeds"]), int(exp["horizon"]), int(exp["start_day"])
    pre = load_snapshot(rundir.root / exp["snapshot"])
    end = start + horizon - 1
    countries = [c.iso3 for c in pre.countries]
    pre_labels = {str(v): int(k) for k, v in pre.macro.get("item_labels", {}).items()}
    for p, name in enumerate(pre.product_names):
        pre_labels[name] = int(pre.products["item_id"][p])
    metrics = {}
    labels = {}
    for s in range(seeds):
        for b in ("baseline", "treated"):
            path = _branch(exp_dir, s, b)
            metrics[(s, b)] = MetricIndex(pq.read_table(path / "metrics.parquet"))
            labels[(s, b)] = _labels(path / "events.parquet", pre_labels)

    def diff(metric: str, label: str | None, country: int, day: int) -> list[float]:
        out = []
        for s in range(seeds):
            vals = []
            for b in ("treated", "baseline"):
                item = -1 if label is None else labels[(s, b)].get(label)
                vals.append(0.0 if item is None else metrics[(s, b)].get(metric, item, country, day))
            out.append(vals[0] - vals[1])
        return out

    lines: list[str] = []
    w = lines.append
    w(f"# Session report: session {session_id}")
    w("")
    w(
        f"From world day {start} ({pre.date(start).isoformat()}), {horizon} days, {seeds} replicate seeds. Each seed "
        "runs the session's turns (treated) and the same world without them (autopilot baseline) with identical "
        "random draws; other prompts in that period are applied in both. Seed 0 replays the main timeline. "
        "Differences are treated minus baseline; shares are population-weighted."
    )
    w("")
    w("## Turns")
    w("")
    by_id = {e["prompt_id"]: e for e in rundir.ledger()}
    for pid in exp["session_prompts"]:
        e = by_id.get(pid)
        if e is None:
            continue
        sc = Scenario.load(rundir.root / e["scenario_file"])
        for ev in sc.events:
            w(f"- day {e['day']}: {_describe(ev)}")
    w("")
    items = _action_items(_branch(exp_dir, 0, "treated") / "events.parquet")
    inv0 = {v: k for k, v in labels[(0, "treated")].items()}
    if items:
        w(f"## Reach of your messages and posts (aware share on day {end})")
        w("")
        w("| claim | country | aware, treated minus baseline |")
        w("|---|---|---|")
        for it in items:
            label = inv0.get(it)
            if label is None:
                continue
            for c, iso in enumerate(countries):
                d = diff("aware_share", label, c, end)
                if any(abs(x) > 0 for x in d):
                    w(f"| {label[:80]} | {iso} | {_pct(d, signed=True)} |")
        w("")
    products = sorted(
        {lab for (s, b), m in labels.items() for lab in m if lab in pre.product_names}
        | {lab for lab in labels[(0, "treated")] if lab in _launched(rundir, exp["session_prompts"])}
    )
    if products:
        w(f"## Products (adults who bought it, day {end})")
        w("")
        w("| product | country | treated minus baseline |")
        w("|---|---|---|")
        for name in products:
            for c, iso in enumerate(countries):
                d = diff("adopted_share_adults", name, c, end)
                if any(abs(x) > 1e-9 for x in d):
                    w(f"| {name} | {iso} | {_pct(d, signed=True)} |")
        w("")
    w(f"## Economic mood (day {end}, -1 gloomy .. +1 upbeat)")
    w("")
    w("| country | treated minus baseline |")
    w("|---|---|")
    for c, iso in enumerate(countries):
        w(f"| {iso} | {_num(diff('sentiment', None, c, end))} |")
    w("")
    w("## Caveats")
    w("")
    w(
        "- Public figures' autonomous decisions are not re-run inside the branches; only decisions already "
        "scheduled before the session started happen (in both branches)."
    )
    w("- People created during the session have population weight 0: they act on others but are not counted in shares.")
    w("- All behavioural coefficients are uncalibrated priors; policy effect sizes are placeholders (roles.yaml).")
    out_dir = rundir.root / "reports" / f"session_{session_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _launched(rundir: RunDir, prompt_ids: list[int]) -> set[str]:
    by_id = {e["prompt_id"]: e for e in rundir.ledger()}
    names: set[str] = set()
    for pid in prompt_ids:
        e = by_id.get(pid)
        if e is not None:
            names |= {
                ev.name
                for ev in Scenario.load(rundir.root / e["scenario_file"]).events
                if isinstance(ev, ProductLaunch)
            }
    return names
