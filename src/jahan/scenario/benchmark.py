"""Benchmarks of everyday activity against real data (`jahan benchmark`).

The checks, their pass limits, what may be tuned and the fit / hold-out split were registered in
``data/backtests/benchmarks.yaml`` before any run; the rule for kappa under late reactions is
``data/global/investors.yaml`` ``trading.kappa.rule_v2``.  Real data: ``data/backtests/benchmark_actuals.yaml``
(written by scripts/benchmark_actuals.py from FRED) and, for B5, Polymarket daily prices fetched at run time and
recorded under ``runs/_benchmark/recordings`` (git-ignored).  Deterministic given those inputs; no LLM.

B1  US stocks on announcement days, release day plus the next 4 FRED dates (score; kappa is chosen here)
B2  the size of a multi-week move (the Iran example against the real S&P 500 after the 2026 strikes)
B3  a quiet year (news part exactly flat, saving part without trend, moods steady)
B4  who invests and how often they look (simulated against the sources)
B5  event odds over quiet stretches (time passing against "the price stays where it started")
B6  mood and unemployment (exploratory, reported only)
B7, B8  statements (no benchmark outside the US; volume not modelled)
"""

from __future__ import annotations

import datetime as dt
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.economy import load_everyday
from jahan.model.investors import investor_masks
from jahan.model.questions import classify_kind
from jahan.scenario.backtest import STABILITY_DATE, BacktestEvent, BacktestFile, _combine, _config
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import MacroPath, MacroStep, QuestionEvent, Scenario
from jahan.sim import economy, questions
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

REGISTRATION = "backtests/benchmarks.yaml"
ACTUALS = "backtests/benchmark_actuals.yaml"
FIT_YEARS = (2023, 2024, 2025)
SEEN_YEARS = (2026,)
HOLDOUT_FROM = dt.date(2026, 10, 17)
GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.001), 3)
WINDOW = 4
SD = 0.92
BOOT = 2000
Progress = Callable[[str], None] | None


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def load_registration(data_dir: Path | None = None) -> dict[str, Any]:
    dd = data_dir or paths.data_dir()
    return dict(yaml.safe_load((dd / REGISTRATION).read_text(encoding="utf-8")))


def load_actuals(data_dir: Path | None = None) -> dict[str, Any]:
    dd = data_dir or paths.data_dir()
    return dict(yaml.safe_load((dd / ACTUALS).read_text(encoding="utf-8")))


def registration_record(data_dir: Path | None = None) -> dict[str, Any]:
    """The commit that registered the rules, and whether they changed since (then a run is refused)."""
    dd = data_dir or paths.data_dir()
    files = [str(dd / REGISTRATION), str(dd / "global" / "investors.yaml")]
    root = dd.parent

    def git(*args: str) -> str:
        res = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return res.stdout.strip() if res.returncode == 0 else ""

    commit = git("log", "-1", "--format=%H", "--", files[0])
    dirty = bool(git("status", "--porcelain", "--", files[0]))
    return {"registered_in": commit or None, "changed_since": dirty}


def _bootstrap(diffs: list[float], seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of ``diffs`` by resampling (fixed seed)."""
    if not diffs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs)
    means = a[rng.integers(0, a.size, (BOOT, a.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------------------------------------ B1 basis


@dataclass
class Basis:
    """One announcement in a world with everyday activity: the published estimate's day-0 move and the people's
    part (at the kappa it was run with) on the release day and each of the next FRED dates, against reality."""

    event: BacktestEvent
    est0: float
    people: list[float]  # percent, [release day, FRED date 1, ...]
    real: list[float]
    cut: bool  # the window was cut before the next event


def b1_events(file: BacktestFile) -> list[BacktestEvent]:
    return [e for e in file.events if e.kind == "forecast" and e.actual.us_stocks is not None and e.announcement]


def _windows(
    events: list[BacktestEvent], actuals: dict[str, Any]
) -> dict[str, tuple[list[dt.date], list[float], bool]]:
    """Each event's next FRED dates and real moves, cut before the next scored event's release date."""
    dates = sorted({e.date for e in events})
    out = {}
    for e in events:
        later = [d for d in dates if d > e.date]
        stop = later[0] if later else None
        rows = actuals["b1_windows"].get(e.id, [])
        keep = [r for r in rows if stop is None or dt.date.fromisoformat(str(r["date"])) < stop]
        out[e.id] = (
            [dt.date.fromisoformat(str(r["date"])) for r in keep],
            [float(r["us_stocks"]) for r in keep],
            len(keep) < len(rows),
        )
    return out


def _econ_config(base: RunConfig, date: dt.date, seed_offset: int, kappa: float, shadow: bool) -> RunConfig:
    cfg = _config(base, date, seed_offset, kappa=kappa, economy=True)
    cfg.params.market_savings = False  # rule v2: the realised cash-outs do not depend on kappa
    cfg.params.market_shadow = shadow
    return cfg


def event_basis(
    event: BacktestEvent,
    base: RunConfig,
    window: tuple[list[dt.date], list[float], bool],
    seed_offset: int,
    kappa: float,
    shadow: bool,
) -> Basis:
    """Run one announcement and read the people's part on each scored date (compounded over the calendar days
    since the previous scored date: weekends and holidays carry over)."""
    assert event.announcement is not None and event.actual.us_stocks is not None
    world = generate_world(_econ_config(base, event.date, seed_offset, kappa, shadow))
    run_in_memory(world, 1)  # the day before the release
    ann = event.announcement.model_copy(update={"start_day_offset": 0})
    apply_scenario(world, Scenario(prompt=event.title, events=[ann]), EventLog(), 1)
    dates, real_after, cut = window
    end = dates[-1] if dates else event.date
    daily: dict[dt.date, float] = {}
    est0 = 0.0
    while world.date() <= end:
        d = world.date()
        run_in_memory(world, 1)
        if d == event.date:
            est0 = float(world.macro["market"]["last"]["scripted_pct"])
        daily[d] = float(world.macro["economy"]["markets"]["USA"]["last"]["people_pct"])
    people = [daily[event.date]]
    prev = event.date
    for d in dates:
        level = 1.0
        day = prev + dt.timedelta(days=1)
        while day <= d:
            level *= 1.0 + daily.get(day, 0.0) / 100.0
            day += dt.timedelta(days=1)
        people.append(100.0 * (level - 1.0))
        prev = d
    return Basis(event, est0, people, [float(event.actual.us_stocks), *real_after], cut)


def b1_score(rows: list[Basis], kappa: float, people_scale: float = 1.0) -> dict[str, Any]:
    """The B1 scores at ``kappa`` (the basis was run at ``people_scale``; shadow runs are linear in kappa)."""
    k = kappa / people_scale
    five, day0, five0, day00, nc5, nc0 = [], [], [], [], [], []
    for r in rows:
        sim = [_combine(r.est0, k * r.people[0])] + [k * p for p in r.people[1:]]
        without = [r.est0] + [0.0] * (len(r.people) - 1)
        five.append(float(np.mean([abs(s - x) for s, x in zip(sim, r.real, strict=True)])) / SD)
        five0.append(float(np.mean([abs(s - x) for s, x in zip(without, r.real, strict=True)])) / SD)
        nc5.append(float(np.mean([abs(x) for x in r.real])) / SD)
        day0.append(abs(sim[0] - r.real[0]) / SD)
        day00.append(abs(without[0] - r.real[0]) / SD)
        nc0.append(abs(r.real[0]) / SD)
    lo, hi = _bootstrap([a - b for a, b in zip(five, five0, strict=True)])
    return {
        "events": float(len(rows)),
        "five_day": float(np.mean(five)) if five else float("nan"),
        "five_day_without_people": float(np.mean(five0)) if five0 else float("nan"),
        "five_day_no_change": float(np.mean(nc5)) if nc5 else float("nan"),
        "day0": float(np.mean(day0)) if day0 else float("nan"),
        "day0_without_people": float(np.mean(day00)) if day00 else float("nan"),
        "day0_no_change": float(np.mean(nc0)) if nc0 else float("nan"),
        "five_day_diff_ci95": [lo, hi],
    }


def average_basis(per_seed: list[list[Basis]]) -> list[Basis]:
    out = []
    for rows in zip(*per_seed, strict=True):
        people = list(np.mean([r.people for r in rows], axis=0))
        out.append(
            Basis(
                rows[0].event,
                float(np.mean([r.est0 for r in rows])),
                [float(p) for p in people],
                rows[0].real,
                rows[0].cut,
            )
        )
    return out


def optimum(rows: list[Basis], people_scale: float = 1.0) -> tuple[float, list[float]]:
    losses = [b1_score(rows, float(k), people_scale)["five_day"] for k in GRID]
    best = int(np.argmin(np.round(losses, 12)))  # ties and the edges resolve to the smallest kappa
    return float(GRID[best]), losses


# ------------------------------------------------------------------------------------------ B2, stability


def iran_example(base: RunConfig, kappa: float, days: int = 60) -> dict[str, Any]:
    """The Iran example in a 10,000-person USA and Iran world with everyday activity (saving off)."""
    cfg = base.model_copy(deep=True)
    cfg.countries, cfg.n_agents, cfg.start_date, cfg.economy, cfg.notables = (
        ["USA", "IRN"],
        10000,
        STABILITY_DATE,
        True,
        True,
    )
    cfg.params.market_kappa = kappa
    cfg.params.market_savings = False
    world = generate_world(cfg)
    apply_scenario(
        world, Scenario.load(paths.data_dir().parent / "examples" / "scenarios" / "iran_us_tension.yaml"), EventLog(), 1
    )
    falls, levels = [], []
    for _ in range(days):
        d = world.date()
        run_in_memory(world, 1)
        falls.append(float(world.macro["economy"]["markets"]["USA"]["last"]["people_pct"]))
        if d.weekday() < 5:
            levels.append(float(world.macro["us_stocks_index"]))
    worst = 0.0
    for a in range(len(levels)):
        for b in range(a + 1, min(len(levels), a + 21)):
            worst = min(worst, 100.0 * (levels[b] / levels[a] - 1.0))
    return {"largest_daily_fall_pct": min(falls), "largest_20_day_fall_pct": worst, "end_index": levels[-1]}


def b2_check(base: RunConfig, kappa: float, actuals: dict[str, Any]) -> dict[str, Any]:
    real = actuals["b2_iran_2026"]
    sim = iran_example(base, kappa)
    ok = sim["largest_daily_fall_pct"] >= float(real["largest_daily_fall_pct"]) and sim[
        "largest_20_day_fall_pct"
    ] >= float(real["largest_20_day_fall_pct"])
    return {
        "kappa": kappa,
        "sim": sim,
        "real": {k: real[k] for k in ("from", "to", "largest_daily_fall_pct", "largest_20_day_fall_pct")},
        "passed": bool(ok),
    }


def stability_v2(base: RunConfig, kappa: float, days: int = 60) -> dict[str, Any]:
    """A scripted -3% day in a 10,000-person world with everyday activity: the people's part stays bounded and
    reverts (rule v2)."""
    cfg = base.model_copy(deep=True)
    cfg.n_agents, cfg.start_date, cfg.economy = 10000, STABILITY_DATE, True
    cfg.params.market_kappa = kappa
    cfg.params.market_savings = False
    world = generate_world(cfg)
    shock = MacroPath(variable="us_stocks", steps=[MacroStep(days=1, pct_per_day=-3.0)])
    apply_scenario(world, Scenario(prompt="crash", events=[shock]), EventLog(), 1)
    path = []
    for _ in range(days):
        run_in_memory(world, 1)
        path.append(float(world.macro["economy"]["markets"]["USA"]["people_index"]))
    dev = [abs(p - 100.0) for p in path]
    peak = max(dev)
    bounded = min(path) > 50.0 and max(path) < 200.0
    reverted = peak <= 0.1 or dev[-1] <= 0.5 * peak
    return {
        "kappa": kappa,
        "people_index_min": round(min(path), 3),
        "people_index_max": round(max(path), 3),
        "people_index_day_60": round(path[-1], 3),
        "bounded": bool(bounded),
        "reverted": bool(reverted),
        "passed": bool(bounded and reverted),
    }


# ---------------------------------------------------------------------------------------------- the fit


@dataclass
class FitResult:
    record: dict[str, Any]
    fit_rows: list[Basis] = field(default_factory=list)
    seen_rows: list[Basis] = field(default_factory=list)


def fit_kappa(
    file: BacktestFile,
    base: RunConfig,
    actuals: dict[str, Any],
    seeds: tuple[int, ...] = (0, 1, 2),
    progress: Progress = None,
) -> FitResult:
    """Choose kappa for everyday activity under rule v2 (data/global/investors.yaml)."""
    events = b1_events(file)
    win = _windows(events, actuals)
    fit_ev = [e for e in events if e.date.year in FIT_YEARS]
    seen_ev = [e for e in events if e.date.year in SEEN_YEARS and e.date < HOLDOUT_FROM]
    hold_ev = [e for e in events if e.date >= HOLDOUT_FROM]
    per_seed: list[list[Basis]] = []
    per_seed_seen: list[list[Basis]] = []
    for s in seeds:
        _say(
            progress, f"basis pass, seed offset {s}: {len(fit_ev)} fit and {len(seen_ev)} seen events (shadow, kappa 1)"
        )
        per_seed.append([event_basis(e, base, win[e.id], s, 1.0, True) for e in fit_ev])
        per_seed_seen.append([event_basis(e, base, win[e.id], s, 1.0, True) for e in seen_ev])
    rows = average_basis(per_seed)
    seen_rows = average_basis(per_seed_seen) if seen_ev else []
    k_opt, losses = optimum(rows)
    seed_opts = {str(s): optimum(r)[0] for s, r in zip(seeds, per_seed, strict=True)}

    def day0_ok(k: float) -> bool:
        sc = b1_score(rows, k)
        return sc["day0"] <= sc["day0_without_people"] + 1e-12

    checks: dict[str, Any] = {}

    def passes(k: float) -> bool:
        if k == 0.0:
            return True
        _say(progress, f"constraints at kappa {k:g}")
        b2, stab, d0 = b2_check(base, k, actuals), stability_v2(base, k), day0_ok(k)
        checks[f"{k:g}"] = {"day0": d0, "b2": b2, "stability": stab}
        return bool(d0 and b2["passed"] and stab["passed"])

    used = k_opt
    if not passes(k_opt):  # the largest grid kappa below the optimum that meets every constraint (bisection)
        lo, hi = 0.0, k_opt
        for _ in range(11):
            mid = round(0.5 * (lo + hi), 3)
            if mid in (lo, hi):
                break
            if passes(mid):
                lo = mid
            else:
                hi = mid
        used = lo
    _say(progress, f"full-feedback check at kappa {used:g}")
    full = [event_basis(e, base, win[e.id], seeds[0], used, False) for e in fit_ev] if used > 0 else []
    shadow_loss = b1_score(per_seed[0], used)["five_day"]
    full_loss = b1_score(full, used, people_scale=used)["five_day"] if full else shadow_loss
    refined = None
    if used > 0 and abs(full_loss - shadow_loss) > 0.005:
        tries = {}
        for m in np.round(np.arange(0.5, 1.5 + 1e-9, 0.1), 1):
            k = round(float(used * m), 3)
            if not passes(k):
                continue
            rows_k = [event_basis(e, base, win[e.id], seeds[0], k, False) for e in fit_ev]
            tries[f"{k:g}"] = b1_score(rows_k, k, people_scale=k)["five_day"]
        if tries:
            best = min(tries, key=lambda x: (tries[x], float(x)))
            refined, used = tries, float(best)
    fit_score = b1_score(rows, used)
    seen_score = b1_score(seen_rows, used) if seen_rows else None
    reg = registration_record()
    record = {
        "date": dt.date.today().isoformat(),
        "command": "jahan benchmark --tune",
        "rule": "data/global/investors.yaml trading.kappa.rule_v2 (pre-registered 2026-10-17)",
        "registered_in": reg["registered_in"],
        "config": f"{base.name}, {base.n_agents} people, everyday activity, saving off",
        "fit_events": len(rows),
        "windows_cut": sum(r.cut for r in rows),
        "grid": "0 to 2, step 0.001",
        "kappa_optimum": k_opt,
        "boundary_solution": bool(k_opt in (float(GRID[0]), float(GRID[-1]))),
        "per_seed_optimum": seed_opts,
        "kappa": used,
        "constraints": checks,
        "shadow_vs_full_five_day": {"shadow": round(shadow_loss, 4), "full": round(full_loss, 4)},
        "refinement": refined,
        "fit_scores": {k: (v if isinstance(v, list) else round(v, 4)) for k, v in fit_score.items()},
        "seen_2026_scores": None
        if seen_score is None
        else {k: (v if isinstance(v, list) else round(v, 4)) for k, v in seen_score.items()},
        "holdout": f"{len(hold_ev)} forecast events from {HOLDOUT_FROM}; scored once 12 exist",
        "loss_at": {
            f"{k:g}": round(losses[int(round(k * 1000))], 4) for k in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
        },
    }
    return FitResult(record, rows, seen_rows)


# ------------------------------------------------------------------------------------------------ B3, B4


def quiet_year(
    sizes: tuple[int, ...] = (3000, 10000),
    seeds: tuple[int, ...] = (1, 2, 3),
    days: int = 365,
    progress: Progress = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"sizes": {}, "passed": True}
    for n in sizes:
        steps, ends, mood_ok, news_flat = [], [], True, True
        for s in seeds:
            _say(progress, f"quiet year: {n} people, seed {s}")
            cfg = RunConfig(n_agents=n, countries=["USA", "DEU", "IRN"], seed=s, start_date=STABILITY_DATE)
            world = generate_world(cfg)
            rec = run_in_memory(world, days)
            t = rec.metrics.table().to_pydict()
            sav = [
                v
                for m, c, v in zip(t["metric"], t["country"], t["value"], strict=True)
                if m == "stocks_savings_pct" and c == 0
            ]
            news = [v for m, v in zip(t["metric"], t["value"], strict=True) if m == "stocks_news_pct"]
            news_flat &= all(v == 0.0 for v in news)
            steps += sav
            ends.append(100.0 * (float(np.prod([1 + v / 100 for v in sav])) - 1.0))
            for c in range(world.n_countries):
                mood = {
                    d: v
                    for d, cc, m, v in zip(t["day"], t["country"], t["metric"], t["value"], strict=True)
                    if m == "sentiment" and cc == c
                }
                ref = mood.get(30)
                if ref is not None and any(abs(v - ref) > 0.05 for d, v in mood.items() if d >= 30):
                    mood_ok = False
        a = np.asarray(steps)
        se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
        no_trend = abs(float(a.mean())) <= 2 * se + 1e-15
        size = {
            "news_part_flat": bool(news_flat),
            "saving_mean_step_pct": float(a.mean()),
            "saving_mean_step_se": se,
            "saving_no_trend": bool(no_trend),
            "saving_largest_step_pct": float(np.abs(a).max()) if a.size else 0.0,
            "saving_year_end_pct": [round(x, 4) for x in ends],
            "saving_year_wander_sd_pct": float(np.std(ends, ddof=1)) if len(ends) > 1 else 0.0,
            "moods_steady": bool(mood_ok),
        }
        out["sizes"][str(n)] = size
        out["passed"] &= bool(news_flat and no_trend and mood_ok)
    return out


def who_invests(n: int = 10000, progress: Progress = None) -> dict[str, Any]:
    _say(progress, "who invests: the m1 countries at 10,000 people")
    cfg = RunConfig(
        n_agents=n,
        countries=["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"],
        seed=20261013,
        start_date=STABILITY_DATE,
    )
    world = generate_world(cfg)
    w = world.pop["pop_weight"].astype(np.float64)
    rows: dict[str, Any] = {}
    ok = True
    for iso, mk in world.macro["economy"]["markets"].items():
        inv = investor_masks(world, mk["cfg"])
        share = 100.0 * float(w[inv.holds].sum() / w[inv.adults].sum()) if w[inv.adults].sum() > 0 else float("nan")
        source = mk["cfg"]["adult_share"] if mk["cfg"]["adult_share"] is not None else 57.7  # SCF bands' average
        good = abs(share - source) <= 5.0
        ok &= good
        rows[iso] = {
            "simulated": round(share, 2),
            "source": round(float(source), 2),
            "holders": int(inv.holds.sum()),
            "passed": bool(good),
        }
    rows["USA"]["gallup_2025"] = 62.0
    att = load_everyday(paths.data_dir())
    assert att is not None
    rates = economy.login_rates(world, world.macro["economy"]["cfg"]["attention"]) * 730.0
    usa = investor_masks(world, world.macro["economy"]["markets"]["USA"]["cfg"])
    days = rates[usa.holds]
    med, mean = float(np.median(days)), float(np.mean(days))
    implied = att.attention.mean_days()  # the interpolated spread's own mean (sampling aside)
    log_ok = abs(med - 36.0) <= 3.6 and abs(mean - 85.2) <= 8.52
    return {
        "ownership": rows,
        "logins": {
            "median_days": round(med, 1),
            "mean_days": round(mean, 1),
            "spread_mean_days": round(implied, 1),
            "holders": int(usa.holds.sum()),
            "published_median": 36,
            "published_mean": 85.2,
            "passed": bool(log_ok),
        },
        "passed": bool(ok and log_ok),
    }


# ----------------------------------------------------------------------------------------------------- B5

B5_QUERIES = ("Iran", "Israel", "Russia Ukraine ceasefire", "China Taiwan", "recession", "government shutdown", "Fed")
B5_COUNTRIES = {
    "iran": "IRN",
    "china": "CHN",
    "taiwan": "CHN",
    "india": "IND",
    "brazil": "BRA",
    "germany": "DEU",
    "nigeria": "NGA",
    "indonesia": "IDN",
}


@dataclass
class OddsCase:
    slug: str
    question: str
    kind: str
    end: str
    prices: list[tuple[str, float]]
    closed: bool


def odds_cases(rec: Any, fetch_day: str, progress: Progress = None, fetcher: Any = None) -> list[OddsCase]:
    from jahan.ingest.polymarket import price_history, search

    seen: dict[str, OddsCase] = {}
    for q in B5_QUERIES:
        _say(progress, f"Polymarket search: {q}")
        for row in search(q, rec, fetch_day, limit=50, fetcher=fetcher, include_closed=True):
            if row.slug in seen or not row.clob_token_id or not row.resolves_by:
                continue
            if not "2025-01-01" <= row.resolves_by <= "2026-12-31":
                continue
            kind, _ = classify_kind(row.question)
            if kind == "other":
                continue
            prices = price_history(row.clob_token_id, rec, fetch_day, fetcher)
            if len(prices) < 30:
                continue
            seen[row.slug] = OddsCase(row.slug, row.question, kind, row.resolves_by, prices, row.closed)
    return list(seen.values())


def odds_case_run(case: OddsCase, talk: bool, days: int = 60) -> dict[str, Any] | None:
    """Simulated odds from the first price with no news, against the real daily prices (until the price first
    passes 0.97 or 0.03, the end date, or 60 days)."""
    p0 = case.prices[0][1]
    if not 0.03 < p0 < 0.97:
        return None
    start = dt.date.fromisoformat(case.prices[0][0])
    end = dt.date.fromisoformat(case.end)
    real = {}
    for d, p in case.prices[1:]:
        day = dt.date.fromisoformat(d)
        if day >= end or (day - start).days > days or not 0.03 < p < 0.97:
            break
        real[day] = p
    if not real:
        return None
    text = case.question.lower()
    countries = ["USA"] + sorted({iso for k, iso in B5_COUNTRIES.items() if k in text} - {"USA"})
    cfg = RunConfig(n_agents=3000, countries=countries, seed=7, start_date=start, notables=False)
    if not talk:
        cfg.params.odds_talk_p = 0.0
    world = generate_world(cfg)
    qid = "b5-" + "".join(ch if ch.isalnum() else "-" for ch in case.slug.lower())[:60].strip("-")
    q = QuestionEvent(
        id=qid if len(qid) >= 3 else "b5-question",
        question=case.question[:300],
        price=p0,
        fetch_date=start.isoformat(),
        resolves_by=end,
        countries=countries,
        kind="by_date" if case.kind == "by_date" else "holds_through",
        kind_by="rule",
    )
    apply_scenario(world, Scenario(prompt="b5", events=[q]), EventLog(), 1)
    sim_err, flat_err = [], []
    last = max(real)
    while world.date() < last:
        run_in_memory(world, 1)
        today = world.date()
        if today in real:
            s = questions.current_odds(world)[q.id]["odds"]
            sim_err.append(abs(s - real[today]))
            flat_err.append(abs(p0 - real[today]))
    if not sim_err:
        return None
    return {
        "slug": case.slug,
        "kind": case.kind,
        "end": case.end,
        "days": len(sim_err),
        "sim": float(np.mean(sim_err)),
        "flat": float(np.mean(flat_err)),
    }


def odds_benchmark(cases: list[OddsCase], progress: Progress = None) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {"no_talk": [], "talk": []}
    for i, c in enumerate(cases):
        _say(progress, f"odds {i + 1}/{len(cases)}: {c.question[:60]}")
        for key, talk in (("no_talk", False), ("talk", True)):
            r = odds_case_run(c, talk)
            if r is not None:
                rows[key].append(r)

    def summ(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for split, test in (
            ("fit", lambda r: r["end"] < "2026-07-01"),
            ("holdout", lambda r: r["end"] >= "2026-07-01"),
        ):
            part = [r for r in rs if test(r)]
            by_kind = {}
            for k in ("by_date", "holds_through", "all"):
                sel = [r for r in part if k == "all" or r["kind"] == k]
                if not sel:
                    continue
                diffs = [r["sim"] - r["flat"] for r in sel]
                by_kind[k] = {
                    "questions": len(sel),
                    "sim_points": round(100 * float(np.mean([r["sim"] for r in sel])), 2),
                    "flat_points": round(100 * float(np.mean([r["flat"] for r in sel])), 2),
                    "diff_ci95_points": [round(100 * x, 2) for x in _bootstrap(diffs)],
                    "no_worse": bool(np.mean([r["sim"] for r in sel]) <= np.mean([r["flat"] for r in sel]) + 1e-12),
                }
            out[split] = by_kind
        return out

    time_passing = summ(rows["no_talk"])
    res: dict[str, Any] = {"time_passing": time_passing, "with_talk": summ(rows["talk"]), "questions": len(cases)}
    fit = time_passing.get("fit", {})
    res["passed"] = bool(fit) and all(v["no_worse"] for v in fit.values())
    res["rows"] = rows["no_talk"]
    return res


# ----------------------------------------------------------------------------------------------------- B6


def mood_unemployment(actuals: dict[str, Any], days: int = 365, progress: Progress = None) -> dict[str, Any]:
    """Exploratory: US mood after layoffs raise unemployment about 1 point (no news), against the real
    sentiment-unemployment slope.  Reported only."""
    out = {}
    for name, mult in (("baseline", 0.0), ("layoffs", 23.1)):  # +23.1% job losses: about +1 point at the steady state
        _say(progress, f"mood and unemployment: {name}")
        cfg = RunConfig(n_agents=10000, countries=["USA"], seed=20261013, start_date=STABILITY_DATE)
        world = generate_world(cfg)
        if mult:
            lever = MacroPath(variable="job_loss_rate", steps=[MacroStep(days=1, pct_per_day=mult)])
            apply_scenario(world, Scenario(prompt="layoffs", events=[lever]), EventLog(), 1)
        rec = run_in_memory(world, days)
        t = rec.metrics.table().to_pydict()
        mood = [v for m, v in zip(t["metric"], t["value"], strict=True) if m == "sentiment"]
        unemp = [v for m, v in zip(t["metric"], t["value"], strict=True) if m == "unemployed_share"]
        out[name] = {
            "mood_end": mood[-1],
            "unemployment_end_pct": 100 * unemp[-1],
            "daily_change_sd": float(np.std(np.diff(mood[30:]))),
        }
    du = out["layoffs"]["unemployment_end_pct"] - out["baseline"]["unemployment_end_pct"]
    dm = out["layoffs"]["mood_end"] - out["baseline"]["mood_end"]
    sd = out["baseline"]["daily_change_sd"]
    sim_sd = (dm / du) / sd if du and sd else float("nan")
    real = actuals["b6_sentiment_unemployment"]
    ratio = sim_sd / float(real["slope_in_sd"]) if real["slope_in_sd"] else float("nan")
    return {
        "unemployment_rise_points": round(du, 3),
        "mood_change": dm,
        "sim_slope_per_point": dm / du if du else float("nan"),
        "sim_slope_in_daily_change_sd": sim_sd,
        "real": real,
        "ratio_sim_to_real": ratio,
        "flag": bool(not (1 / 3 <= abs(ratio) <= 3)) if ratio == ratio else True,
        "note": "reported only: real sentiment also moves with news about jobs; the two scales differ",
        "runs": out,
    }


STATEMENTS = {
    "B7": "No benchmark exists outside the US; every non-US size is an untested illustration using borrowed "
    "kappa and multiplier.",
    "B8": "Trading volume is not modelled: Jahan models each holder's whole stock position, not stock picking.",
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_markdown(report: dict[str, Any]) -> str:
    """The benchmark report as Markdown (docs/benchmarks.md)."""
    out = [
        "# Everyday activity benchmarks",
        "",
        f"Run {report['date']} (`{report['command']}`); checks, limits and splits registered in "
        f"`data/backtests/benchmarks.yaml` (commit {report.get('registered_in') or 'unknown'}) before any run. "
        "Simulated, not a forecast.",
        "",
        "| Check | Result | Passed |",
        "|---|---|---|",
    ]
    for key in ("B1", "B2", "B3", "B4", "B5", "B6"):
        c = report["checks"].get(key)
        if c is None:
            continue
        verdict = "yes" if c.get("passed") else "reported only" if c.get("passed") is None else "NO"
        out.append(f"| {key} | {c['summary']} | {verdict} |")
    for key, text in STATEMENTS.items():
        out.append(f"| {key} | {text} | stated |")
    out += [
        "",
        "## Details",
        "",
        "```yaml",
        yaml.safe_dump(report["checks"], sort_keys=False, allow_unicode=True).rstrip(),
        "```",
        "",
    ]
    return "\n".join(out)
