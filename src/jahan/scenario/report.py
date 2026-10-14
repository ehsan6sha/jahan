"""Experiment report for any scenario mix.

Sections: products (adoption, who adopted, influence channels, trust, stories), news reach, economic
mood (treated minus baseline), macro paths, and data/model caveats.  All differences are paired
(treated minus baseline within the same seed), summarised across seeds.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from jahan import paths
from jahan.gen.registry import confidence_summary
from jahan.model.enums import EventKind, InfoTier, Sex
from jahan.model.world import World
from jahan.rng import Stream, u01
from jahan.scenario.spec import MacroPath, NewsEvent, ProductLaunch, Scenario
from jahan.sim.decisions import income_basis
from jahan.sim.trust import effective_trust
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot

ISCO_LABELS = {
    -1: "not employed",
    0: "armed forces",
    1: "manager",
    2: "professional",
    3: "technician",
    4: "clerical worker",
    5: "service or sales worker",
    6: "farm worker",
    7: "craft worker",
    8: "machine operator",
    9: "elementary occupation",
}
CHECKPOINTS = [7, 30, 90, 180, 365]
SIMULATED_LABEL = "[simulated]"


class MetricIndex:
    """metrics table -> {(metric, item, country): {day: value}}"""

    def __init__(self, table: Any) -> None:
        d = table.to_pydict()
        self.data: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
        for day, c, it, m, v in zip(d["day"], d["country"], d["item"], d["metric"], d["value"], strict=True):
            self.data[(m, int(it), int(c))][int(day)] = float(v)

    def get(self, metric: str, item: int, country: int, day: int, default: float = 0.0) -> float:
        return self.data.get((metric, item, country), {}).get(day, default)


def _branch(exp_dir: Path, s: int, b: str) -> Path:
    return exp_dir / f"seed_{s:02d}" / b


def _pct(x: list[float], signed: bool = False) -> str:
    a = 100 * np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return "n/a"
    f = "{:+.1f}" if signed else "{:.1f}"
    if a.size == 1:
        return f.format(a[0]) + "%"
    return (
        f.format(a.mean()) + f"% (sd {a.std(ddof=1):.1f}, range " + f.format(a.min()) + " to " + f.format(a.max()) + ")"
    )


def _num(x: list[float]) -> str:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return "n/a"
    if a.size == 1:
        return f"{a[0]:+.3f}"
    return f"{a.mean():+.3f} (sd {a.std(ddof=1):.3f})"


def _price(ev: ProductLaunch, world: World, country: int) -> float | None:
    spec = world.countries[country]
    if spec.iso3 in ev.price_local:
        return ev.price_local[spec.iso3]
    if ev.price_usd is not None:
        return ev.price_usd * spec.fx_local_per_usd
    return None


def _agent_label(world: World, i: int) -> str:
    pop = world.pop
    c = int(pop["country"][i])
    nt = world.names[c]
    age = int((world.day - int(pop["birth_day"][i])) / 365.25)
    sex = "F" if int(pop["sex"][i]) == Sex.F else "M"
    occ = ISCO_LABELS.get(int(pop["occupation"][i]), "?")
    area = "urban" if int(pop["region"][i]) == 1 else "rural"
    tier = InfoTier(int(pop["info_tier"][i])).name.lower().replace("_", " ")
    name = nt.display(int(pop["first_name_id"][i]), int(pop["last_name_id"][i]))
    return f"{name} ({sex}, {age}, {occ}, {world.countries[c].iso3} {area}, information: {tier})"


def _created_items(events_path: Path) -> list[tuple[int, str]]:
    """(item id, latest label) of items created in a branch, in creation order.

    Running stories (e.g. a currency slide) carry the headline of their last update.
    """
    t = pq.read_table(events_path).to_pydict()
    order: list[int] = []
    label: dict[int, str] = {}
    for k, it, det in zip(t["kind"], t["item"], t["detail"], strict=True):
        if it < 0:
            continue
        if k == EventKind.ITEM_CREATED:
            order.append(int(it))
            label[int(it)] = str(det)
        elif k == EventKind.ITEM_UPDATED:
            label[int(it)] = str(det)
    return [(i, label[i]) for i in order]


class Context:
    def __init__(self, rundir: RunDir, prompt_id: int) -> None:
        entry = rundir.prompt(prompt_id)
        self.prompt_id = prompt_id
        self.exp_dir = rundir.experiment_dir(prompt_id)
        exp = json.loads((self.exp_dir / "experiment.json").read_text(encoding="utf-8"))
        self.seeds, self.horizon = int(exp["seeds"]), int(exp["horizon"])
        self.scenario = Scenario.load(rundir.root / entry["scenario_file"])
        self.pre = load_snapshot(rundir.root / entry["snapshot_before"])
        self.d0 = self.pre.day
        self.n0 = self.pre.pop.n
        self.countries = [c.iso3 for c in self.pre.countries]
        self.metrics = {
            (s, b): MetricIndex(pq.read_table(_branch(self.exp_dir, s, b) / "metrics.parquet"))
            for s in range(self.seeds)
            for b in ("baseline", "treated")
        }
        self.created = _created_items(_branch(self.exp_dir, 0, "treated") / "events.parquet")
        self.out_dir = rundir.report_dir(prompt_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def checkpoints(self) -> list[int]:
        cps = [c for c in CHECKPOINTS if c <= self.horizon]
        return cps if cps and cps[-1] == self.horizon else [*cps, self.horizon]

    def item_for(self, label: str) -> int | None:
        for it, lab in self.created:
            if lab == label:
                return it
        return None

    def diff(self, metric: str, item: int, country: int, day: int) -> list[float]:
        return [
            self.metrics[(s, "treated")].get(metric, item, country, day)
            - self.metrics[(s, "baseline")].get(metric, item, country, day)
            for s in range(self.seeds)
        ]


def build_report(rundir: RunDir, prompt_id: int) -> Path:
    ctx = Context(rundir, prompt_id)
    lines: list[str] = []
    w = lines.append
    sc = ctx.scenario
    w(f"# Scenario report: prompt {prompt_id}")
    w("")
    w(f"**Prompt:** {sc.prompt or '(none)'}")
    w("")
    w(
        f"Injected on world day {ctx.d0} ({ctx.pre.date(ctx.d0).isoformat()}). {ctx.seeds} replicate seeds, "
        f"{ctx.horizon} simulated days each. Each seed runs a baseline (no scenario) and a treated branch with "
        "identical random draws, so differences isolate the scenario's effect. Seed 0 replays the main timeline. "
        "Shares are population-weighted."
    )
    w("")
    curve_rows: list[list[Any]] = []
    for ev in sc.events:
        if isinstance(ev, ProductLaunch):
            _product_section(ctx, ev, w, curve_rows)
    news = [ev for ev in sc.events if isinstance(ev, NewsEvent)]
    auto = [
        (it, lab)
        for it, lab in ctx.created
        if lab.startswith(("Gold price", "IRR", "USD")) or " against the US dollar" in lab
    ]
    if news or auto:
        _news_section(ctx, news, auto, w)
    _mood_section(ctx, w)
    if any(isinstance(ev, MacroPath) for ev in sc.events):
        _macro_section(ctx, w)
    with (ctx.out_dir / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(
            [
                "product",
                "country",
                "day",
                "seed",
                "aware_treated",
                "aware_baseline",
                "adopted_treated",
                "adopted_baseline",
            ]
        )
        cw.writerows(curve_rows)
    _caveats(ctx, w)
    path = ctx.out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _product_section(ctx: Context, ev: ProductLaunch, w: Any, curve_rows: list[list[Any]]) -> None:
    item = ctx.item_for(ev.name)
    if item is None:
        return
    pre = ctx.pre
    params = pre.config.params
    n0 = ctx.n0
    w(f"## {ev.name}: adoption and awareness (treated minus baseline)")
    w("")
    w("| Country | Day | Aware | Adopted (adults) |")
    w("|---|---|---|---|")
    launched = [ctx.countries.index(c) for c in ev.countries]
    for c in launched:
        for cp in ctx.checkpoints():
            day = ctx.d0 + cp - 1
            aw = ctx.diff("aware_share", item, c, day)
            ad = ctx.diff("adopted_share_adults", item, c, day)
            w(f"| {ctx.countries[c]} | {cp} | {_pct(aw)} | {_pct(ad)} |")
            for s in range(ctx.seeds):
                mt, mb = ctx.metrics[(s, "treated")], ctx.metrics[(s, "baseline")]
                curve_rows.append(
                    [
                        ev.name,
                        ctx.countries[c],
                        cp,
                        s,
                        mt.get("aware_share", item, c, day),
                        mb.get("aware_share", item, c, day),
                        mt.get("adopted_share_adults", item, c, day),
                        mb.get("adopted_share_adults", item, c, day),
                    ]
                )
    w("")
    finals = [np.load(_branch(ctx.exp_dir, s, "treated") / "final.npz") for s in range(ctx.seeds)]
    slots = [np.flatnonzero(f["slot_item"] == item) for f in finals]
    if any(s.size == 0 for s in slots):
        return
    slot = int(slots[0][0])
    pop = pre.pop
    weight = pop["pop_weight"].astype(np.float64)
    age = pre.ages()
    adult = age >= params.min_adopt_age
    in_market = adult & np.isin(pop["country"], launched)
    basis = income_basis(pre)
    quint = np.zeros(n0, dtype=np.int64)
    for c in launched:
        m = (pop["country"] == c) & adult
        if m.sum() >= 5:
            quint[m] = np.searchsorted(np.quantile(basis[m], [0.2, 0.4, 0.6, 0.8]), basis[m], side="right")
    groups = {
        "Income quintile (within country)": (quint, {i: f"Q{i + 1}" for i in range(5)}),
        "Information tier": (pop["info_tier"].astype(np.int64), {t.value: t.name.lower() for t in InfoTier}),
        "Age band": (
            np.select([age < 25, age < 40, age < 60], [0, 1, 2], 3),
            {0: "<25", 1: "25-39", 2: "40-59", 3: "60+"},
        ),
        "Sex": (pop["sex"].astype(np.int64), {0: "female", 1: "male"}),
        "Area": (pop["region"].astype(np.int64), {0: "rural", 1: "urban"}),
    }
    if len(launched) > 1:
        groups["Country"] = (pop["country"].astype(np.int64), {c: ctx.countries[c] for c in launched})
    adopted_by_seed = [f["adopted_day"][:n0, int(s[0])] >= 0 for f, s in zip(finals, slots, strict=True)]
    aware_by_seed = [f["aware"][:n0, int(s[0])] == 1 for f, s in zip(finals, slots, strict=True)]
    w(f"### Who adopted {ev.name} (treated, end of horizon, adults where it launched; mean over seeds)")
    w("")
    for title, (g, labels) in groups.items():
        w(f"**{title}**")
        w("")
        w("| Group | Aware | Adopted |")
        w("|---|---|---|")
        for k, label in labels.items():
            m = in_market & (g == k)
            if weight[m].sum() <= 0:
                continue
            aw = np.mean([np.average(a[m], weights=weight[m]) for a in aware_by_seed])
            ad = np.mean([np.average(a[m], weights=weight[m]) for a in adopted_by_seed])
            w(f"| {label} | {100 * aw:.1f}% | {100 * ad:.1f}% |")
        w("")
    # channels of influence
    w(f"### How people heard about {ev.name} and who influenced them")
    w("")
    wom, total = [], []
    infl: dict[int, int] = {}
    for s in range(ctx.seeds):
        t = pq.read_table(_branch(ctx.exp_dir, s, "treated") / "events.parquet").to_pydict()
        others = [
            o
            for k, o, it in zip(t["kind"], t["other"], t["item"], strict=True)
            if k == EventKind.ADOPTED and it == item
        ]
        total.append(len(others))
        wom.append(sum(1 for o in others if o >= 0))
        if s == 0:
            for o in others:
                if o >= 0:
                    infl[o] = infl.get(o, 0) + 1
    share = [a / b for a, b in zip(wom, total, strict=True) if b > 0]
    w(f"- Adopters whose last accepted advice came from a person rather than media: {_pct(share)}.")
    top = sorted(infl.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    if top:
        w("- Most influential agents in seed 0 (adoptions they advised):")
        for a, cnt in top:
            if a < n0:
                w(f"  - {_agent_label(pre, int(a))}: {cnt}")
    w("")
    # trust
    f0 = finals[0]
    td = f0["trust_delta"]
    heard = f0["heard_from"][:n0, slot]
    advised = np.zeros(td.shape[0], dtype=bool)
    adopters0 = np.flatnonzero(adopted_by_seed[0])
    adv = heard[adopters0]
    ok = (adv >= 0) & (adv < n0)
    if ok.any():
        rows = pre.find_edges(adopters0[ok].astype(np.int64), adv[ok].astype(np.int64))
        advised[rows[rows >= 0]] = True
    w(f"### Trust ({ev.name})")
    w("")
    if advised.any():
        w(
            f"- Ties that carried advice leading to an adoption (seed 0): mean trust change "
            f"{td[advised].mean():+.3f} over "
            f"{int(advised.sum())} ties; other ties {td[~advised].mean():+.3f}."
        )
    else:
        w("- No adoption was advised by a person in seed 0.")
    w("")
    _stories(
        ctx,
        ev,
        slot,
        f0,
        heard,
        aware_by_seed[0],
        adopted_by_seed[0],
        adult & np.isin(pop["country"], launched),
        basis,
        td,
        w,
    )


def _stories(
    ctx: Context,
    ev: ProductLaunch,
    slot: int,
    f0: Any,
    heard: Any,
    aw0: Any,
    ad0: Any,
    eligible: Any,
    basis: Any,
    td: Any,
    w: Any,
) -> None:
    pre = ctx.pre
    n0 = ctx.n0
    w(f"### Sampled stories ({ev.name}, seed 0, template narratives)")
    w("")
    rank = u01(pre.sim_seed, Stream.SAMPLE, ctx.prompt_id, np.arange(n0, dtype=np.int64))
    order = np.argsort(rank, kind="stable")
    picks = [i for i in order if ad0[i] and eligible[i]][:3] + [
        i for i in order if aw0[i] and not ad0[i] and eligible[i]
    ][:2]
    picks += [i for i in order if not aw0[i] and eligible[i]][:1]
    trust_pre = effective_trust(pre, slice(None), pre.config.params)
    for i in picks:
        fd = int(f0["first_day"][i, slot])
        hf = int(heard[i])
        if 0 <= hf < n0:
            src_txt = f"advice from {_agent_label(pre, hf)}"
        elif hf >= n0:
            src_txt = "advice from a person born during the run"
        elif hf <= -2:
            src_txt = f"the channel {pre.channel_names[-hf - 2]}"
        else:
            src_txt = "conversations it did not act on"
        story = f"- **{_agent_label(pre, int(i))}**: "
        if not aw0[i]:
            story += f"never heard of {ev.name} during the horizon."
        else:
            story += f"first heard of {ev.name} on day {fd - ctx.d0 + 1}; latest accepted source: {src_txt}. "
            story += f"View of the product at the end: {float(f0['belief'][i, slot]):.2f} (0 = bad, 1 = excellent). "
            if ad0[i]:
                adday = int(f0["adopted_day"][i, slot])
                price = _price(ev, pre, int(pre.pop["location_country"][i]))
                story += f"Adopted on day {adday - ctx.d0 + 1}"
                if price is not None:
                    share = 100 * price / max(float(basis[i]), 1.0)
                    story += f" (list price about {share:.0f}% of monthly income at launch)"
                sat = float(f0["satisfaction"][i, slot])
                story += f"; satisfaction {sat:.2f}." if sat >= 0 else "; outcome not yet realised."
                if 0 <= hf < n0:
                    r = pre.find_edges(np.array([i]), np.array([hf]))[0]
                    if r >= 0:
                        story += f" Trust in the adviser moved from {trust_pre[r]:.2f} to {trust_pre[r] + td[r]:.2f}."
            else:
                story += "Did not adopt."
        w(story)
    w("")


def _news_section(ctx: Context, news: list[NewsEvent], auto: list[tuple[int, str]], w: Any) -> None:
    w("## News reach (treated; share of each country aware)")
    w("")
    w("| Story | Country | Day | Aware |")
    w("|---|---|---|---|")
    stories = [(ctx.item_for(ev.headline), ev.headline) for ev in news] + auto[:6]
    for item, label in stories:
        if item is None:
            continue
        for c in range(len(ctx.countries)):
            vals = [
                ctx.metrics[(s, "treated")].get("aware_share", item, c, ctx.d0 + ctx.horizon - 1)
                for s in range(ctx.seeds)
            ]
            if max(vals) <= 0:
                continue
            w(f"| {label} | {ctx.countries[c]} | {ctx.horizon} | {_pct(vals)} |")
    w("")


def _mood_section(ctx: Context, w: Any) -> None:
    rows = []
    for c in range(len(ctx.countries)):
        cells = []
        for cp in ctx.checkpoints():
            cells.append(_num(ctx.diff("sentiment", -1, c, ctx.d0 + cp - 1)))
        rows.append((ctx.countries[c], cells))
    if all(all(cell.startswith("+0.000") for cell in cells) for _, cells in rows):
        return
    w("## Economic mood (treated minus baseline; mood ranges -1 to +1)")
    w("")
    w("| Country | " + " | ".join(f"Day {cp}" for cp in ctx.checkpoints()) + " |")
    w("|---|" + "---|" * len(ctx.checkpoints()))
    for iso, cells in rows:
        w(f"| {iso} | " + " | ".join(cells) + " |")
    w("")


def _macro_section(ctx: Context, w: Any) -> None:
    w("## Macro paths (treated, seed 0)")
    w("")
    m = ctx.metrics[(0, "treated")]
    last = ctx.d0 + ctx.horizon - 1
    for c, iso in enumerate(ctx.countries):
        start = float(ctx.pre.macro.get("fx", [float("nan")] * (c + 1))[c])  # before the scenario
        end = m.get("fx_local_per_usd", -1, c, last, float("nan"))
        if np.isfinite(start) and np.isfinite(end) and abs(end / start - 1) > 1e-9:
            w(
                f"- {iso} {ctx.pre.countries[c].currency} per USD: {start:,.4g} to {end:,.4g} "
                f"({100 * (end / start - 1):+.1f}%)."
            )
    g0 = float(ctx.pre.macro.get("gold_index", float("nan")))
    g1 = m.get("gold_index", -1, -1, last, float("nan"))
    if np.isfinite(g0) and np.isfinite(g1) and abs(g1 / g0 - 1) > 1e-9:
        w(f"- Gold index: {g0:.1f} to {g1:.1f} ({100 * (g1 / g0 - 1):+.1f}%).")
    w("")


def _caveats(ctx: Context, w: Any) -> None:
    w("## Caveats")
    w("")
    w(
        "- Behavioural coefficients are uncalibrated priors (see `jahan/sim/params.py`); "
        "treat magnitudes as illustrative."
    )
    w(
        "- Stories are template narratives generated from the simulation state, not quotes; any text attributed to a "
        f"real person elsewhere in Jahan is labelled {SIMULATED_LABEL}."
    )
    w("- Data confidence of the distributions used:")
    for iso, conf in confidence_summary(paths.data_dir(), ctx.countries).items():
        weak = sorted(k for k, v in conf.items() if v in ("provisional", "placeholder", "low"))
        w(
            f"  - {iso}: {len(weak)} of {len(conf)} sources are low, provisional or placeholder"
            + (f" ({', '.join(weak)})." if weak else ".")
        )
    w("")
