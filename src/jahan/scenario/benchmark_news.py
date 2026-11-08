"""B9 and B10 of `jahan benchmark` and the rule_v4 fit of s (data/backtests/benchmarks.yaml version 3).

B9   US stocks on unscheduled news days (data/backtests/news_days.yaml): each day's headlines are typed as one
     prompt into the What-if parser, the parsed news is injected into a world dated the day before, and the
     simulated release-day move (the people's part at the fitted kappa plus other investors' part) is scored
     against the real S&P 500.  The other investors' part is linear in s, so it is computed once at s = 1.
B10  ordinary-day movement against the real days it was fitted to (reported).

Claude's ratings are recorded under ``runs/_benchmark/recordings`` (git-ignored) and replay offline; a run that
needs a rating it does not have stops unless calls are allowed (`jahan benchmark --rate`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.provider import LLMClient
from jahan.model.investors import load_investors
from jahan.model.world import World
from jahan.scenario.inject import apply_scenario
from jahan.scenario.parser import parse_prompt
from jahan.scenario.spec import NewsEvent, Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

NEWS_DAYS = "backtests/news_days.yaml"
S_GRID = np.round(np.arange(0.0, 20.0 + 1e-9, 0.01), 2)
SD = 0.92
BOOT = 2000
BIG = 2.0  # percent: days whose real move is at least this big are checked for direction and size
LAST_RELEASE_HOUR = 20  # the headlines came out by the New York close
Progress = Callable[[str], None] | None


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def bench_root() -> Path:
    return paths.data_dir().parent / "runs" / "_benchmark"


@dataclass
class NewsDay:
    id: str
    day: dt.date
    split: str  # fit | seen | holdout | outcome
    chosen: list[str]
    headlines: list[str]
    real: list[float]  # S&P 500 percent moves: the trading day, then the next FRED dates
    tech: list[float] | None = None  # Nasdaq-100, when the git-ignored file exists


def load_news_days(data_dir: Path | None = None) -> list[NewsDay]:
    dd = data_dir or paths.data_dir()
    doc = yaml.safe_load((dd / NEWS_DAYS).read_text(encoding="utf-8"))
    tech_path = dd.parent / "runs" / "_benchmark" / "news_days_tech.yaml"
    tech = yaml.safe_load(tech_path.read_text(encoding="utf-8"))["moves"] if tech_path.exists() else {}
    out = []
    for d in doc["days"]:
        real = (d.get("actual") or {}).get("us_stocks")
        if not real:
            continue
        out.append(
            NewsDay(
                id=str(d["id"]),
                day=dt.date.fromisoformat(str(d["trading_day"])),
                split=str(d["split"]),
                chosen=list(d["chosen"]),
                headlines=[str(e["headline"]) for e in d["events"]],
                real=[float(x) for x in real],
                tech=tech.get(str(d["trading_day"])),
            )
        )
    return out


def news_prompt(d: NewsDay) -> str:
    when = f"{d.day:%A}, {d.day:%B} {d.day.day}, {d.day.year}"
    return f"News that came out by the New York stock market close on {when}:\n" + "\n".join(
        f"- {h}" for h in d.headlines
    )


def _parse_world(day: dt.date) -> World:
    """A small world dated the trading day, only to give the parser its date and countries (no public figures,
    so a 2023 headline is not read next to a 2026 roster)."""
    cfg = RunConfig(n_agents=300, countries=["USA", "DEU"], seed=1, start_date=day - dt.timedelta(days=1))
    cfg.notables = False
    world = generate_world(cfg)
    run_in_memory(world, 1)
    return world


def rate_day(client: LLMClient, d: NewsDay, purpose_tag: str = "") -> dict[str, Any]:
    world = _parse_world(d.day)
    res = parse_prompt(client, world, news_prompt(d), purpose_tag=purpose_tag)
    return {
        "scenario": None if res.scenario is None else res.scenario.model_dump(mode="json"),
        "questions": res.questions,
        "error": res.error,
    }


def news_events(rating: dict[str, Any]) -> tuple[list[NewsEvent], dict[str, int]]:
    """The parsed news, all on the trading day and out by the close; other event types are left out (counted)."""
    sc = rating.get("scenario")
    if not sc:
        return [], {"dropped": 0, "moved_hour": 0, "moved_day": 0}
    events, dropped, hour, day = [], 0, 0, 0
    for ev in Scenario.model_validate(sc).events:
        if not isinstance(ev, NewsEvent):
            dropped += 1
            continue
        upd: dict[str, Any] = {}
        if ev.start_day_offset:
            upd["start_day_offset"], day = 0, day + 1
        if ev.release_utc_hour > LAST_RELEASE_HOUR:
            upd["release_utc_hour"], hour = LAST_RELEASE_HOUR, hour + 1
        events.append(ev.model_copy(update=upd) if upd else ev)
    return events, {"dropped": dropped, "moved_hour": hour, "moved_day": day}


def ratings_summary(ratings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Test-retest agreement of market_valence (the strongest rated story of each day, first against second)."""

    def strongest(r: dict[str, Any]) -> float | None:
        evs, _ = news_events(r)
        vals = [float(e.market_valence) for e in evs if e.market_valence is not None]
        return max(vals, key=abs) if vals else None

    a, b, missing = [], [], 0
    for r in ratings.values():
        x, y = strongest(r["first"]), strongest(r["retest"])
        if x is None and y is None:
            continue
        if x is None or y is None:
            missing += 1
            continue
        a.append(x)
        b.append(y)
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return {
        "days_rated_both_times": len(a),
        "rated_only_once": missing,
        "market_valence_correlation": round(corr, 3),
        "mean_abs_difference": round(float(np.mean(np.abs(np.subtract(a, b)))), 3) if a else float("nan"),
        "same_sign_share": round(float(np.mean(np.sign(a) == np.sign(b))), 3) if a else float("nan"),
    }


@dataclass
class DayBasis:
    day: NewsDay
    people: float  # percent at the fitted kappa (seed average)
    others: float  # percent of the S&P 500 at s = 1
    tech_extra: float  # the tech line's own percent at s = 1
    notes: dict[str, int] = field(default_factory=dict)


def day_basis(
    d: NewsDay, events: list[NewsEvent], base: RunConfig, seeds: tuple[int, ...], kappa: float
) -> tuple[float, float, float]:
    people, others, tech = [], [], []
    for s in seeds:
        cfg = base.model_copy(deep=True)
        cfg.countries, cfg.n_agents = ["USA", "DEU"], base.n_agents
        cfg.start_date, cfg.seed, cfg.economy = d.day - dt.timedelta(days=1), base.seed + s, True
        p = cfg.params
        p.market_kappa, p.market_savings, p.market_quiet, p.market_shadow, p.market_others_s = (
            kappa,
            False,
            False,
            True,
            1.0,
        )
        world = generate_world(cfg)
        run_in_memory(world, 1)  # the day before
        if events:
            apply_scenario(world, Scenario(prompt=d.id, events=list(events)), EventLog(), 1)
        run_in_memory(world, 1)  # the trading day
        eco = world.macro["economy"]
        people.append(float(eco["markets"]["USA"]["last"]["people_pct"]))
        oth = eco["others"]["last"]
        others.append(float(oth["others_pct"]))
        tech.append(float(oth["tech_pct"]))
    return float(np.mean(people)), float(np.mean(others)), float(np.mean(tech))


def b9_scores(rows: list[DayBasis], s: float, tech_beta: float) -> dict[str, Any]:
    if not rows:
        return {"days": 0}
    sim = np.array([r.people + s * r.others for r in rows])
    real = np.array([r.day.real[0] for r in rows])
    zero = np.array([r.people for r in rows])  # s = 0: the engine without other investors
    big = np.abs(real) >= BIG
    out: dict[str, Any] = {
        "days": len(rows),
        "day0": round(float(np.mean(np.abs(sim - real)) / SD), 4),
        "day0_s0": round(float(np.mean(np.abs(zero - real)) / SD), 4),
        "day0_no_change": round(float(np.mean(np.abs(real)) / SD), 4),
        "big_days": int(big.sum()),
    }
    if big.any():
        ratio = sim[big] / real[big]
        out["big_direction_right"] = int(np.sum(np.sign(sim[big]) == np.sign(real[big])))
        out["big_within_factor_2"] = int(np.sum((ratio >= 0.5) & (ratio <= 2.0)))
        out["big_days_detail"] = [
            {"day": r.day.day.isoformat(), "real": round(float(x), 2), "sim": round(float(y), 2)}
            for r, x, y in zip([r for r, b in zip(rows, big, strict=True) if b], real[big], sim[big], strict=True)
        ]
    techs = [(r, r.day.tech[0]) for r in rows if r.day.tech]
    if techs:
        ts = np.array([tech_beta * (r.people + s * r.others) + s * r.tech_extra for r, _ in techs])
        tr = np.array([x for _, x in techs])
        out["tech_day0"] = round(float(np.mean(np.abs(ts - tr)) / SD), 4)
        out["tech_day0_no_change"] = round(float(np.mean(np.abs(tr)) / SD), 4)
    after = [(np.sign(r.day.real[0]), sum(r.day.real[1:])) for r in rows if abs(r.day.real[0]) >= 1.0]
    if after:  # other investors keep their move; do real moves partly reverse over the next 4 days?
        out["next_4_days_after_1pct_days"] = {
            "days": len(after),
            "mean_signed_move_pct": round(float(np.mean([sg * x for sg, x in after])), 3),
        }
    return out


def fit_s(rows: list[DayBasis]) -> tuple[float, list[float]]:
    p = np.array([r.people for r in rows])
    o = np.array([r.others for r in rows])
    real = np.array([r.day.real[0] for r in rows])
    losses = np.mean(np.abs(p[None, :] + S_GRID[:, None] * o[None, :] - real[None, :]), axis=1) / SD
    best = int(np.argmin(np.round(losses, 12)))  # ties and the edges resolve to the smallest s
    return float(S_GRID[best]), [float(x) for x in losses]


def bootstrap_s(rows: list[DayBasis], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p = np.array([r.people for r in rows])
    o = np.array([r.others for r in rows])
    real = np.array([r.day.real[0] for r in rows])
    picks = []
    for _ in range(BOOT):
        i = rng.integers(0, len(rows), len(rows))
        losses = np.mean(np.abs(p[i][None, :] + S_GRID[:, None] * o[i][None, :] - real[i][None, :]), axis=1)
        picks.append(float(S_GRID[int(np.argmin(np.round(losses, 12)))]))
    return float(np.percentile(picks, 2.5)), float(np.percentile(picks, 97.5))


# --------------------------------------------------------------------------------------- B2 with ratings


def rated_iran_example(client: LLMClient) -> tuple[Scenario, list[dict[str, Any]]]:
    """The Iran example with each news event's market rating and sector from the parser (rule_v4: rated before
    the run, recorded; the example's own salience kept)."""
    sc = Scenario.load(paths.data_dir().parent / "examples" / "scenarios" / "iran_us_tension.yaml")
    from jahan.scenario.backtest import STABILITY_DATE

    world = _parse_world(STABILITY_DATE)
    events, notes = [], []
    for ev in sc.events:
        if isinstance(ev, NewsEvent):
            res = parse_prompt(client, world, f"News today: {ev.headline}", purpose_tag="b2_rating")
            rated = [e for e in (res.scenario.events if res.scenario else []) if isinstance(e, NewsEvent)]
            mv = rated[0].market_valence if rated else None
            sector = rated[0].sectors if rated else "broad"
            ev = ev.model_copy(update={"market_valence": mv, "sectors": sector})
            notes.append({"headline": ev.headline, "market_valence": mv, "sectors": sector})
        events.append(ev)
    return Scenario(prompt=sc.prompt, events=events, horizon_days=sc.horizon_days), notes


def iran_example_v4(base: RunConfig, kappa: float, s: float, scenario: Scenario, days: int = 60) -> dict[str, Any]:
    """The rated Iran example (10,000 people, USA and IRN, saving and ordinary days off): the US index's largest
    one-day fall (people and other investors together) and its largest fall within 20 trading days."""
    from jahan.scenario.backtest import STABILITY_DATE

    cfg = base.model_copy(deep=True)
    cfg.countries, cfg.n_agents, cfg.start_date, cfg.economy, cfg.notables = (
        ["USA", "IRN"],
        10000,
        STABILITY_DATE,
        True,
        True,
    )
    p = cfg.params
    p.market_kappa, p.market_savings, p.market_quiet, p.market_others_s = kappa, False, False, s
    world = generate_world(cfg)
    apply_scenario(world, scenario, EventLog(), 1)
    levels = [float(world.macro["us_stocks_index"])]
    for _ in range(days):
        d = world.date()
        run_in_memory(world, 1)
        if d.weekday() < 5:
            levels.append(float(world.macro["us_stocks_index"]))
    daily = [100.0 * (b / a - 1.0) for a, b in zip(levels, levels[1:], strict=False)]
    worst = 0.0
    for a in range(len(levels)):
        for b in range(a + 1, min(len(levels), a + 21)):
            worst = min(worst, 100.0 * (levels[b] / levels[a] - 1.0))
    return {"largest_daily_fall_pct": min(daily), "largest_20_day_fall_pct": worst, "end_index": levels[-1]}


# ------------------------------------------------------------------------------------------------ the run


def _prompt_hash() -> str:
    import hashlib
    import json

    from jahan.scenario.parser import SYSTEM
    from jahan.scenario.spec import PromptScenario

    text = SYSTEM + json.dumps(PromptScenario.model_json_schema(), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _b2_bounded(base: RunConfig, kappa: float, s: float, scenario: Scenario, real: dict[str, Any]) -> dict[str, Any]:
    sim = iran_example_v4(base, kappa, s, scenario)
    ok = sim["largest_daily_fall_pct"] >= float(real["largest_daily_fall_pct"]) and sim[
        "largest_20_day_fall_pct"
    ] >= float(real["largest_20_day_fall_pct"])
    return {"s": s, "sim": sim, "passed": bool(ok)}


def run_b9(
    base: RunConfig,
    client: LLMClient,
    kappa: float,
    seeds: tuple[int, ...],
    tune: bool,
    actuals: dict[str, Any],
    progress: Progress = None,
) -> dict[str, Any]:
    """Rate every B9 day (twice), simulate each, and score; with ``tune``, choose s under rule_v4 and check its
    constraints (B2 with the rated Iran example; stability and B1 cannot react to s, which is stated)."""
    import json

    days = load_news_days()
    ratings: dict[str, dict[str, Any]] = {}
    for d in days:
        _say(progress, f"B9: rating {d.day}")
        ratings[d.id] = {"first": rate_day(client, d), "retest": rate_day(client, d, "retest")}
    rows = []
    for d in days:
        events, notes = news_events(ratings[d.id]["first"])
        _say(progress, f"B9: {d.day} ({len(events)} stories)")
        people, others, tech = day_basis(d, events, base, seeds, kappa)
        rows.append(DayBasis(d, people, others, tech, notes))
    root = bench_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "b9_basis.json").write_text(
        json.dumps(
            {
                "ratings": ratings,
                "rows": [
                    {"id": r.day.id, "people": r.people, "others_at_s1": r.others, "tech_at_s1": r.tech_extra}
                    for r in rows
                ],
            },
            default=str,
        ),
        encoding="utf-8",
    )
    inv = load_investors(paths.data_dir())
    assert inv is not None and inv.other_investors is not None
    tech_beta = float(inv.derived["us_tech_stocks"].beta or 1.0)
    fit_rows = [r for r in rows if r.day.split == "fit"]
    s = float(inv.other_investors.s or 0.0)
    record: dict[str, Any] | None = None
    if tune:
        s_opt, losses = fit_s(fit_rows)
        lo, hi = bootstrap_s(fit_rows)
        _say(progress, f"B9: optimum s = {s_opt}; checking B2 with the rated Iran example")
        scenario, rated = rated_iran_example(client)
        real = actuals["b2_iran_2026"]
        checks = {f"{s_opt:g}": _b2_bounded(base, kappa, s_opt, scenario, real)} if s_opt > 0 else {}
        s = s_opt
        substituted = False
        if s_opt > 0 and not checks[f"{s_opt:g}"]["passed"]:  # the largest grid s below it that meets B2
            lo_i, hi_i = 0, int(np.searchsorted(S_GRID, s_opt))
            while hi_i - lo_i > 1:
                mid = (lo_i + hi_i) // 2
                c = _b2_bounded(base, kappa, float(S_GRID[mid]), scenario, real)
                checks[f"{S_GRID[mid]:g}"] = c
                lo_i, hi_i = (mid, hi_i) if c["passed"] else (lo_i, mid)
            s, substituted = float(S_GRID[lo_i]), True
        fit_s_scores = b9_scores(fit_rows, s, tech_beta)
        if fit_s_scores["day0"] > fit_s_scores["day0_s0"]:
            s = 0.0  # registered: if no positive s beats s = 0 on the fit days, s is 0
        record = {
            "date": dt.date.today().isoformat(),
            "command": "jahan benchmark --only B9 --tune --rate",
            "rule": "data/global/investors.yaml other_investors.rule_v4 (registered 2026-11-08, before any B9 data)",
            "config": f"{base.n_agents} people, USA and DEU, everyday activity version 2, saving and ordinary days "
            f"off, kappa {kappa:g} fixed, seeds {list(seeds)}",
            "parser": {"model": client.model_for("scenario_parser"), "prompt_hash": _prompt_hash()},
            "fit_days": len(fit_rows),
            "grid": "0 to 20, step 0.01",
            "s_optimum": s_opt,
            "boundary_solution": bool(s_opt in (float(S_GRID[0]), float(S_GRID[-1]))),
            "s_ci95": [lo, hi],
            "substituted_for_b2": substituted,
            "s": s,
            "constraints": {
                "b2": {"iran_example_ratings": rated, "real": real, "runs": checks},
                "stability": "a scripted -3% day has no rated story, so other investors cannot react (unchanged "
                "from trading.kappa.fitted_v3)",
                "b1": "announcement headlines get no other-investor reaction, so B1 cannot change with s",
            },
            "loss_at": {f"{g:g}": round(losses[int(round(g / 0.01))], 4) for g in (0, 0.5, 1, 2, 3, 5, 10, 20)},
        }
    by_split: dict[str, Any] = {}
    for split in ("fit", "seen", "holdout", "outcome"):
        part = [r for r in rows if r.day.split == split]
        by_split[split] = b9_scores(part, s, tech_beta)
    hold = by_split["holdout"]
    out: dict[str, Any] = {
        "s": s,
        "kappa": kappa,
        "scores": by_split,
        "ratings": ratings_summary(ratings),
        "notes": {k: int(sum(r.notes.get(k, 0) for r in rows)) for k in ("dropped", "moved_hour", "moved_day")},
        "holdout_status": "provisional: {} news-chosen hold-out days (a verdict needs 20)".format(hold.get("days", 0)),
        "passed": bool(by_split["fit"].get("day0", 1e9) <= by_split["fit"].get("day0_s0", 0.0) + 1e-12),
        "record": record,
    }
    if record is not None:
        record["fit_scores"] = by_split["fit"]
        record["seen_scores"] = by_split["seen"]
        record["outcome_scores"] = by_split["outcome"]
        record["holdout_scores"] = hold
        record["ratings_retest"] = out["ratings"]
    return out


# ----------------------------------------------------------------------------------------------------- B10


def quiet_check(progress: Progress = None) -> dict[str, Any]:
    """B10: three quiet years at 3,000 people (USA and DEU) against the real days the coefficients came from."""
    inv = load_investors(paths.data_dir())
    q = None if inv is None or inv.other_investors is None else inv.other_investors.quiet
    if not q:
        return {"passed": None, "summary": "no ordinary-day coefficients yet"}
    series: dict[str, list[float]] = {"us_stocks": [], "gold": [], "euro": []}
    month: dict[str, list[float]] = {"us_stocks": [], "gold": [], "euro": []}
    metric = {"us_stocks": "us_stocks_quiet_pct", "gold": "gold_quiet_pct", "euro": "dollar_quiet_pct"}
    for seed in (1, 2, 3):
        _say(progress, f"B10: quiet year, seed {seed}")
        cfg = RunConfig(n_agents=3000, countries=["USA", "DEU"], seed=seed, start_date="2026-11-09")
        world = generate_world(cfg)
        rec = run_in_memory(world, 365)
        t = rec.metrics.table().to_pydict()
        rows: dict[str, dict[int, float]] = {}
        for d, m, v in zip(t["day"], t["metric"], t["value"], strict=True):
            if m in metric.values():
                rows.setdefault(m, {})[int(d)] = float(v)
        for key, name in metric.items():
            days = [d for d in sorted(rows[name]) if world.date(d).weekday() < 5]
            moves = [rows[name][d] for d in days]
            if key == "euro":  # the euro's dollar rate, from euros per dollar
                moves = [100.0 * (1.0 / (1.0 + x / 100.0) - 1.0) for x in moves]
            series[key] += moves
            lv = np.cumprod(1.0 + np.asarray(moves) / 100.0)
            month[key] += list(np.abs(lv[21:] / lv[:-21] - 1.0) * 100.0)
    out: dict[str, Any] = {"sim": {}, "real": q["real_b10"], "within_20pct": {}}
    ok = True
    for key in series:
        a = np.abs(np.asarray(series[key]))
        sim = {
            "mean_abs_daily_pct": round(float(a.mean()), 4),
            "p95_abs_daily_pct": round(float(np.percentile(a, 95)), 4),
            "median_abs_21_day_pct": round(float(np.median(month[key])), 4),
        }
        out["sim"][key] = sim
        within = {k: abs(sim[k] / float(q["real_b10"][key][k]) - 1.0) <= 0.2 for k in sim}
        out["within_20pct"][key] = within
        ok = ok and all(within.values())
    out["passed"] = bool(ok)
    return out
