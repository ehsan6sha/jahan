"""Backtests: replay real announcements in a world dated the day before, and compare with what happened.

On an announcement, US stocks, gold and the euro's dollar rate move by published event-study estimates times
the surprise (data/global/markets.yaml).  US stocks also move by the simulated holders' trading on their change
in mood (sim/market.py), reported as a separate column: "estimates only" (the published estimates as Jahan
applies them), "people only" and "hybrid" (both), next to a naive "no change" forecast.  ``fit_people``
chooses kappa, the one fitted number of the people's part, under the rule in data/global/investors.yaml.
Separately the backtest measures how far the news spreads among the simulated people.  The backtest world
holds people, political camps and public figures, but no news from before the event.  Deterministic and
offline: no LLM, no network.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.investors import load_investors
from jahan.model.markets import MarketsConfig, combined, load_markets, market_moves
from jahan.model.notables import is_later_roster
from jahan.model.world import World
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Announcement, Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

TARGETS: tuple[str, ...] = ("us_stocks", "gold", "dollar")
LABELS = {"us_stocks": "US stocks", "gold": "Gold", "dollar": "Dollar"}
ADULT_AGE = 18.0
MAJOR_SD = 2.0  # a miss larger than this many daily s.d. is a major discrepancy
SIGN_ACTUAL_SD = 1.0  # a wrong sign counts when the real move exceeds this many s.d. ...
SIGN_FORECAST_SD = 0.25  # ... and the forecast is at least this large (smaller forecasts have no meaningful sign)


class Moves(BaseModel):
    """Close-to-close percent changes on the release day.  Stocks: S&P 500 close; gold: spot price after the
    New York close (futures settle before a 14:00 Fed release); dollar: the ICE dollar index (DXY)."""

    model_config = ConfigDict(extra="forbid")
    us_stocks: float | None = None
    gold: float | None = None
    dollar: float | None = None

    def get(self, target: str) -> float | None:
        return getattr(self, target)


class BacktestEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    date: dt.date  # release day (New York)
    title: str
    # forecast: known inputs only; hindsight_only: nothing to forecast, only the after-the-fact column;
    # not_modelled: no published estimate exists, Jahan moves nothing unless a prompt scripts it
    kind: Literal["forecast", "hindsight_only", "not_modelled"] = "forecast"
    announcement: Announcement | None = None
    expected_note: str = ""  # where the market expectation comes from ("assumed" when no source was found)
    two_year_bp: float | None = None  # the day's change in the 2-year Treasury yield (hindsight column only)
    actual: Moves
    sources: list[str] = Field(default_factory=list)
    note: str = ""


class SocietyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: str
    days_after: int = 2
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    source: str  # registry id
    note: str = ""


class Noise(BaseModel):
    """Typical daily move (s.d. of daily percent changes) used to judge errors."""

    model_config = ConfigDict(extra="forbid")
    us_stocks: float = Field(gt=0)
    gold: float = Field(gt=0)
    dollar: float = Field(gt=0)
    note: str = ""

    def get(self, target: str) -> float:
        return float(getattr(self, target))


class BacktestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieved: dt.date
    noise_sd: Noise
    events: list[BacktestEvent]
    society: list[SocietyCheck] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> BacktestFile:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @classmethod
    def load_all(cls, files: list[Path]) -> BacktestFile:
        """Several event files as one, in date order.  They must judge errors by the same typical moves."""
        parts = [cls.load(p) for p in files]
        if not parts:
            raise ValueError("no backtest event files")
        noise = parts[0].noise_sd
        for p, part in zip(files[1:], parts[1:], strict=True):
            if part.noise_sd.model_dump(exclude={"note"}) != noise.model_dump(exclude={"note"}):
                raise ValueError(f"{p.name}: noise_sd differs from {files[0].name}")
        events = sorted((e for part in parts for e in part.events), key=lambda e: (e.date, e.id))
        ids = [e.id for e in events]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate event ids across backtest files")
        return cls(
            retrieved=max(p.retrieved for p in parts),
            noise_sd=noise,
            events=events,
            society=[s for part in parts for s in part.society],
        )


def default_events_files() -> list[Path]:
    return sorted((paths.data_dir() / "backtests").glob("events_*.yaml"))


def default_events_file() -> Path:
    return paths.data_dir() / "backtests" / "events_2026.yaml"


@dataclass
class EventResult:
    event: BacktestEvent
    surprise: float | None
    jahan: dict[str, float] = field(default_factory=dict)  # release-day % from the published estimates only
    people: dict[str, float] = field(default_factory=dict)  # US stocks: the simulated holders' trading only
    hybrid: dict[str, float] = field(default_factory=dict)  # both (stocks); gold and dollar as in ``jahan``
    hindsight: dict[str, float] = field(default_factory=dict)  # with the day's 2-year move as guidance
    side: dict[str, float] = field(default_factory=dict)  # inactive estimates, shown for comparison
    aware: list[list[float]] = field(default_factory=list)  # per seed: share of US adults aware, by day after


def _config(base: RunConfig, date: dt.date, seed_offset: int, kappa: float | None = None) -> RunConfig:
    cfg = base.model_copy(deep=True)
    cfg.start_date = date - dt.timedelta(days=1)
    cfg.seed = base.seed + seed_offset
    # the published benchmark and kappa were made with people trading on the day they learn news; everyday
    # activity's backtest (late reactions, rule v2) is a separate, pre-registered run
    cfg.economy = False
    if kappa is not None:
        cfg.params.market_kappa = kappa
    return cfg


def check_data(cfg: RunConfig, data_dir: Path | None = None) -> list[str]:
    """Data problems for a past-dated world, except that the roster is dated after the world's start."""
    dd = data_dir or paths.data_dir()
    found = validate_data(dd, cfg.countries, start_date=cfg.start_date if cfg.notables else None)
    return [p for p in found if not is_later_roster(p)]


def _pct(after: float, before: float) -> float:
    return 100.0 * (after / before - 1.0)


def adults_aware(world: World, headline: str, iso3: str = "USA") -> float:
    """Population-weighted share of adults living in ``iso3`` who know the story with this headline."""
    labels = world.macro.get("item_labels", {})
    items = [int(k) for k, v in labels.items() if v == headline[:300]]
    codes = [c.iso3 for c in world.countries]
    if not items or iso3 not in codes:
        return float("nan")
    slot = int(world.items["slot"][items[-1]])
    if slot < 0:
        return float("nan")
    pop = world.pop
    n = pop.n
    who = pop["alive"] & (pop["country"] == codes.index(iso3)) & (world.ages() >= ADULT_AGE)
    w = pop["pop_weight"].astype(np.float64) * who
    aware = world.beliefs.m["aware"][:n, slot] == 1
    return float((w * aware).sum() / w.sum()) if w.sum() > 0 else float("nan")


def simulate(
    event: BacktestEvent, base: RunConfig, seed_offset: int = 0, days_after: int = 2, kappa: float | None = None
) -> EventResult:
    """Run one event in a fresh world dated the day before: market moves on the day, and how many US adults
    know the headline at the end of each day from the release day to ``days_after`` days later.

    ``kappa`` overrides the people's-trading coefficient the world copies from data/global/investors.yaml."""
    assert event.announcement is not None
    cfg = _config(base, event.date, seed_offset, kappa)
    problems = check_data(cfg)
    if problems:
        raise ValueError("data problems:\n" + "\n".join(problems))
    world = generate_world(cfg)
    run_in_memory(world, 1)  # the day before the release
    codes = [c.iso3 for c in world.countries]
    stocks0 = float(world.macro["us_stocks_index"])
    gold0 = float(world.macro["gold_index"])
    fx0 = list(world.macro["fx"])
    ann = event.announcement.model_copy(update={"start_day_offset": 0})
    apply_scenario(world, Scenario(prompt=event.title, events=[ann]), EventLog(), 1)
    res = EventResult(event, ann.actual - ann.expected)
    shares: list[float] = []
    for d in range(days_after + 1):
        run_in_memory(world, 1)
        if d == 0:
            market = world.macro.get("market")
            # the estimates' part of the day: the index before people traded (the same number as before they did)
            scripted = (
                float(market["last"]["scripted_pct"])
                if market
                else _pct(float(world.macro["us_stocks_index"]), stocks0)
            )
            res.jahan = {
                "us_stocks": scripted,
                "gold": _pct(float(world.macro["gold_index"]), gold0),
                "dollar": _pct(world.macro["fx"][codes.index("DEU")], fx0[codes.index("DEU")])
                if "DEU" in codes
                else float("nan"),
            }
            if market:
                res.people = {"us_stocks": float(market["last"]["people_pct"])}
                res.hybrid = {**res.jahan, "us_stocks": float(market["last"]["total_pct"])}
        shares.append(adults_aware(world, ann.headline))
    res.aware.append(shares)
    return res


def hindsight(event: BacktestEvent, cfg_markets: MarketsConfig) -> tuple[dict[str, float], dict[str, float]]:
    """After the fact: the day's 2-year yield move beyond what the decision explains, used as the guidance
    surprise (not a forecast: nobody knows it before the release).  Also the inactive estimates."""
    ann = event.announcement
    if ann is None:
        return {}, {}
    side = {m.target: m.pct for m in market_moves(cfg_markets, ann.indicator, ann.actual, ann.expected) if not m.active}
    ind = cfg_markets.indicators[ann.indicator]
    if event.two_year_bp is None or not ind.guidance or ind.target_share_of_2y is None:
        return {}, side
    guidance = event.two_year_bp - ind.target_share_of_2y * (ann.actual - ann.expected)
    moves = market_moves(cfg_markets, ann.indicator, ann.actual, ann.expected, guidance_bp=guidance)
    return combined(moves), side


def score(sim: float, actual: float, sd: float) -> dict[str, Any]:
    err = sim - actual
    major = abs(err) > MAJOR_SD * sd or (
        np.sign(sim) != np.sign(actual) and abs(actual) > SIGN_ACTUAL_SD * sd and abs(sim) >= SIGN_FORECAST_SD * sd
    )
    return {"err_sd": err / sd, "naive_err_sd": abs(actual) / sd, "major": bool(major)}


@dataclass
class Report:
    file: BacktestFile
    results: list[EventResult]
    agents: int
    seeds: int
    config_name: str
    kappa: float | None = None  # people's trading coefficient used (None: worlds without people's trading)
    fit: dict[str, Any] | None = None  # the record of data/global/investors.yaml, when kappa was fitted


def effective_kappa(base: RunConfig, kappa: float | None = None) -> float | None:
    """The kappa a backtest world uses: an explicit override, the config's, or data/global/investors.yaml's."""
    if kappa is not None:
        return kappa
    if base.params.market_kappa is not None:
        return float(base.params.market_kappa)
    inv = load_investors(paths.data_dir())
    return None if inv is None else float(inv.trading.kappa.value)


def run_backtests(
    file: BacktestFile,
    base: RunConfig,
    seeds: int = 3,
    progress: Callable[[str], None] | None = None,
    kappa: float | None = None,
    only: Callable[[BacktestEvent], bool] | None = None,
) -> Report:
    """Every event in the file (or those ``only`` keeps).  Forecast events run in a fresh world (``seeds``
    worlds for events with a news-reach check); after-the-fact and unmodelled events need no world."""
    markets = load_markets(paths.data_dir())
    if markets is None:
        raise ValueError("data/global/markets.yaml is missing")
    results = []
    society_events = {s.event for s in file.society}
    for ev in file.events:
        if only is not None and not only(ev):
            continue
        if ev.kind == "not_modelled" or ev.announcement is None:
            results.append(EventResult(ev, None))
            continue
        if progress:
            progress(f"{ev.date.isoformat()} {ev.title}")
        if ev.kind == "hindsight_only":
            res = EventResult(ev, ev.announcement.actual - ev.announcement.expected)
        else:
            res = simulate(ev, base, 0, kappa=kappa)
            for s in range(1, seeds if ev.id in society_events else 1):
                res.aware += simulate(ev, base, s, kappa=kappa).aware
        res.hindsight, res.side = hindsight(ev, markets)
        results.append(res)
    inv = load_investors(paths.data_dir())
    fitted = None if inv is None or kappa is not None else inv.trading.kappa.fitted
    return Report(file, results, base.n_agents, seeds, base.name, effective_kappa(base, kappa), fitted)


# ---------------------------------------------------------------------------------------------------- fitting

FIT_YEARS = (2023, 2024, 2025)
HOLDOUT_YEARS = (2026,)
KAPPA_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.001), 3)
STABILITY_DAYS = 40
STABILITY_SHOCK_PCT = -3.0
STABILITY_DATE = dt.date(2026, 10, 13)  # a Tuesday


def _combine(scripted: float, people: float) -> float:
    """Total percent move of the day when people trade after the scripted part (compounding)."""
    return 100.0 * ((1.0 + scripted / 100.0) * (1.0 + people / 100.0) - 1.0)


def _scored(results: list[EventResult]) -> list[EventResult]:
    return [r for r in _forecasts(results) if r.event.actual.us_stocks is not None and "us_stocks" in r.people]


def _mean_misses(rows: list[EventResult], kappa: float, sd: float) -> dict[str, float]:
    """Mean miss in s.d. of each column at ``kappa``, from a run at kappa = 1 (the people's part is linear)."""
    est, ppl, hyb, naive = [], [], [], []
    for r in rows:
        real = float(r.event.actual.us_stocks)  # type: ignore[arg-type]
        s, a = r.jahan["us_stocks"], kappa * r.people["us_stocks"]
        est.append(abs(s - real) / sd)
        ppl.append(abs(a - real) / sd)
        hyb.append(abs(_combine(s, a) - real) / sd)
        naive.append(abs(real) / sd)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {
        "events": float(len(rows)),
        "estimates_only": mean(est),
        "people_only": mean(ppl),
        "hybrid": mean(hyb),
        "no_change": mean(naive),
    }


def choose_kappa(rows: list[EventResult], sd: float) -> tuple[float, list[float]]:
    """The grid kappa with the smallest mean hybrid miss (ties and edges go to the smallest)."""
    losses = [_mean_misses(rows, float(k), sd)["hybrid"] for k in KAPPA_GRID]
    return float(KAPPA_GRID[int(np.argmin(losses))]), losses


def stability(base: RunConfig, kappa: float) -> dict[str, Any]:
    """A scripted -3% day in a fresh world: does the people's part stay bounded and revert within 40 days?

    Bounded: the people's part of the index never halves or doubles it.  Reverted: by day 40 it has recovered
    at least half of its largest deviation (or never moved by more than 0.1%)."""
    from jahan.scenario.spec import MacroPath, MacroStep  # local: keep the module's import surface small

    cfg = base.model_copy(deep=True)
    cfg.start_date = STABILITY_DATE
    cfg.params.market_kappa = kappa
    world = generate_world(cfg)
    shock = MacroPath(variable="us_stocks", steps=[MacroStep(days=1, pct_per_day=STABILITY_SHOCK_PCT)])
    apply_scenario(world, Scenario(prompt="stability check", events=[shock]), EventLog(), 1)
    path = []
    for _ in range(STABILITY_DAYS + 1):
        run_in_memory(world, 1)
        path.append(float(world.macro["market"]["people_index"]))
    dev = [abs(np.log(p / 100.0)) for p in path]
    worst = max(dev)
    bounded = min(path) > 50.0 and max(path) < 200.0
    reverted = worst < np.log(1.001) or dev[-1] <= 0.5 * worst
    return {
        "kappa": kappa,
        "shock_pct": STABILITY_SHOCK_PCT,
        "days": STABILITY_DAYS,
        "people_index_min": round(min(path), 3),
        "people_index_max": round(max(path), 3),
        "people_index_day_40": round(path[-1], 3),
        "bounded": bool(bounded),
        "reverted": bool(reverted),
        "passed": bool(bounded and reverted),
    }


def scenario_sizes(base: RunConfig, kappa: float, days: int = 60) -> dict[str, dict[str, float]]:
    """Reported, not judged: how far the people's part of the index moves in two what-ifs at ``kappa``.

    Compare with real days in data/backtests/events_2026.yaml (US-Iran escalations moved the S&P 500 by
    less than 1% on the day)."""
    from jahan.scenario.spec import NewsEvent  # local: keep the module's import surface small

    cases = {
        "iran_us_tension example": Scenario.load(
            paths.data_dir().parent / "examples" / "scenarios" / "iran_us_tension.yaml"
        ),
        "recession headline (valence -0.8, salience 0.9, 7 days)": Scenario(
            prompt="recession",
            events=[
                NewsEvent(
                    headline="US economy enters recession as layoffs surge",
                    topic="economy",
                    valence=-0.8,
                    salience=0.9,
                    countries=["USA"],
                    duration_days=7,
                )
            ],
        ),
    }
    out = {}
    for name, sc in cases.items():
        cfg = base.model_copy(deep=True)
        cfg.start_date = STABILITY_DATE
        cfg.params.market_kappa = kappa
        world = generate_world(cfg)
        apply_scenario(world, sc, EventLog(), 1)
        path = []
        for _ in range(days):
            run_in_memory(world, 1)
            path.append(float(world.macro["market"]["people_index"]))
        low = min(path)
        out[name] = {
            "min": round(low, 2),
            "min_day": path.index(low),
            "max": round(max(path), 2),
            "day_60": round(path[-1], 2),
        }
    return out


def fit_people(
    file: BacktestFile, base: RunConfig, progress: Callable[[str], None] | None = None
) -> tuple[dict[str, Any], Report]:
    """Choose kappa under the pre-registered rule (data/global/investors.yaml) and return the record to paste.

    One run at kappa = 1 gives each event's release-day people's move A; at any kappa it is kappa x A, because
    kappa reaches people's mood only through headlines about the people's own moves, which spread from the
    next day.  That is checked by re-running every event at the chosen kappa."""
    sd = file.noise_sd.us_stocks
    fit_or_hold = lambda e: e.kind == "forecast" and e.date.year in FIT_YEARS + HOLDOUT_YEARS  # noqa: E731
    at_one = run_backtests(file, base, seeds=1, progress=progress, kappa=1.0, only=fit_or_hold)
    scored = _scored(at_one.results)
    fit_rows = [r for r in scored if r.event.date.year in FIT_YEARS]
    hold_rows = [r for r in scored if r.event.date.year in HOLDOUT_YEARS]
    kappa, losses = choose_kappa(fit_rows, sd)
    edge = kappa in (float(KAPPA_GRID[0]), float(KAPPA_GRID[-1]))
    if progress:
        progress(f"chosen kappa {kappa:g}; checking linearity and stability")
    check = run_backtests(file, base, seeds=1, kappa=kappa, only=fit_or_hold)
    by_id = {r.event.id: r for r in _scored(check.results)}
    worst_lin = max(
        (
            abs(by_id[r.event.id].people["us_stocks"] - kappa * r.people["us_stocks"])
            for r in scored
            if r.event.id in by_id
        ),
        default=0.0,
    )
    stab = stability(base, kappa)
    used = kappa
    if not stab["passed"]:  # the largest grid kappa below the optimum that passes (bisection)
        lo, hi = 0.0, kappa
        for _ in range(11):
            mid = round(0.5 * (lo + hi), 3)
            if stability(base, mid)["passed"]:
                lo = mid
            else:
                hi = mid
        used = lo
        stab = {**stability(base, used), "optimum_failed_at": kappa}
    opposite = [
        r.event.id
        for r in fit_rows + hold_rows
        if abs(used * r.people["us_stocks"]) > abs(r.jahan["us_stocks"]) > 0
        and np.sign(r.people["us_stocks"]) != np.sign(r.jahan["us_stocks"])
    ]
    events_n = len(fit_rows)
    record = {
        "date": dt.date.today().isoformat(),
        "command": "jahan backtest --fit-people",
        "config": f"{base.name}, {base.n_agents} people, seed offset 0",
        "rule": "data/global/investors.yaml trading.kappa.rule (pre-registered 2026-10-15)",
        "fit_events": f"{events_n} forecast events {FIT_YEARS[0]}-{FIT_YEARS[-1]} with a real US stock move",
        "grid": f"{KAPPA_GRID[0]:g} to {KAPPA_GRID[-1]:g}, step 0.001",
        "kappa_optimum": kappa,
        "boundary_solution": bool(edge),
        "kappa": used,
        "linearity_check_max_abs_diff_pct": float(f"{worst_lin:.2e}"),
        "fit_mean_miss_sd": {k: round(v, 4) for k, v in _mean_misses(fit_rows, used, sd).items()},
        "holdout_mean_miss_sd": {k: round(v, 4) for k, v in _mean_misses(hold_rows, used, sd).items()},
        "loss_at": {f"{k:g}": round(losses[int(round(k * 1000))], 4) for k in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0)},
        "opposite_sign_days": opposite,
        "stability": stab,
        "scenario_people_index": scenario_sizes(base, used),
    }
    return record, at_one


def _f(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) < 0.5 * 10**-digits:
        return f"{0:.{digits}f}%"
    return f"{x:+.{digits}f}%"


def _surprise_text(ev: BacktestEvent, surprise: float | None) -> str:
    ann = ev.announcement
    if ann is None or surprise is None:
        return ""
    if ann.indicator == "us_policy_rate":
        return f"{surprise:+.1f} bp"
    return f"{surprise:+.0f}k jobs"


def _forecasts(results: list[EventResult]) -> list[EventResult]:
    return [r for r in results if r.event.kind == "forecast"]


Column = Literal["jahan", "people", "hybrid"]


def summary(
    report: Report, results: list[EventResult] | None = None, column: Column = "jahan"
) -> dict[str, dict[str, float]]:
    """Per market: mean miss in daily s.d. of Jahan and of "no change", major misses, and direction calls
    and hits, over ``results`` (default: every forecast event).  ``column``: the published estimates alone
    ("jahan"), the simulated holders' trading alone ("people", US stocks only) or both ("hybrid")."""
    noise = report.file.noise_sd
    out: dict[str, dict[str, float]] = {}
    for t in TARGETS:
        errs, naive, calls, hits, majors = [], [], 0, 0, 0
        for r in _forecasts(report.results if results is None else results):
            real = r.event.actual.get(t)
            sim = getattr(r, column).get(t)
            if real is None or sim is None or math.isnan(sim):
                continue
            sd = noise.get(t)
            errs.append(abs(sim - real) / sd)
            naive.append(abs(real) / sd)
            majors += int(score(sim, real, sd)["major"])
            if abs(sim) >= SIGN_FORECAST_SD * sd:
                calls += 1
                hits += int(np.sign(sim) == np.sign(real))
        out[t] = {
            "events": float(len(errs)),
            "jahan_mean_miss_sd": float(np.mean(errs)) if errs else float("nan"),
            "no_change_mean_miss_sd": float(np.mean(naive)) if naive else float("nan"),
            "major_misses": float(majors),
            "direction_calls": float(calls),
            "direction_hits": float(hits),
        }
    return out


GROUP_LABELS = {"us_policy_rate": "Fed decisions", "us_payrolls": "Jobs reports"}


def groups(report: Report) -> list[tuple[str, list[EventResult]]]:
    """Forecast events by kind of announcement and year, then per kind, then all."""
    fc = _forecasts(report.results)
    out: list[tuple[str, list[EventResult]]] = []
    for ind, label in GROUP_LABELS.items():
        mine = [r for r in fc if r.event.announcement is not None and r.event.announcement.indicator == ind]
        for year in sorted({r.event.date.year for r in mine}):
            out.append((f"{label} {year}", [r for r in mine if r.event.date.year == year]))
        if len({r.event.date.year for r in mine}) > 1:
            years = sorted({r.event.date.year for r in mine})
            out.append((f"**{label} {years[0]}-{years[-1]}**", mine))
    out.append(("**All**", fc))
    return out


def to_json(report: Report) -> dict[str, Any]:
    noise = report.file.noise_sd
    rows = []
    for r in report.results:
        ev = r.event
        row: dict[str, Any] = {
            "id": ev.id,
            "date": ev.date.isoformat(),
            "title": ev.title,
            "kind": ev.kind,
            "surprise": r.surprise,
            "actual": ev.actual.model_dump(),
            "jahan": r.jahan,
            "people": r.people,
            "hybrid": r.hybrid,
            "hindsight": r.hindsight,
            "side": r.side,
            "aware": r.aware,
        }
        if ev.kind == "forecast":
            row["scores"] = {
                t: score(r.jahan[t], real, noise.get(t))
                for t in TARGETS
                if (real := ev.actual.get(t)) is not None and t in r.jahan and not math.isnan(r.jahan[t])
            }
        rows.append(row)
    return {
        "config": report.config_name,
        "agents": report.agents,
        "seeds": report.seeds,
        "noise_sd": noise.model_dump(),
        "events": rows,
        "summary": summary(report),
        "summary_people": summary(report, column="people"),
        "summary_hybrid": summary(report, column="hybrid"),
        "kappa": report.kappa,
        "fit": report.fit,
        "society": [s.model_dump() for s in report.file.society],
    }


def render_summary(report: Report) -> list[str]:
    """One row per kind of announcement and year: average misses next to "no change", major misses and
    direction calls."""
    lines = [
        '**Summary.** Average miss in typical daily moves, Jahan\'s published estimates / "no change"; major misses '
        "out of forecasts; "
        "direction right out of the forecasts large enough to have one (a quarter of a typical move or more).",
        "",
        "| Events | Count | US stocks | Gold | Dollar | Major misses | Direction right |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, rows in groups(report):
        sm = summary(report, rows)
        cells = [
            f"{sm[t]['jahan_mean_miss_sd']:.2f} / {sm[t]['no_change_mean_miss_sd']:.2f}" if sm[t]["events"] else "n/a"
            for t in TARGETS
        ]
        majors = sum(int(sm[t]["major_misses"]) for t in TARGETS)
        cells_n = sum(int(sm[t]["events"]) for t in TARGETS)
        calls = sum(int(sm[t]["direction_calls"]) for t in TARGETS)
        hits = sum(int(sm[t]["direction_hits"]) for t in TARGETS)
        lines.append(
            f"| {label} | {len(rows)} | " + " | ".join(cells) + f" | {majors} of {cells_n} | {hits} of {calls} |"
        )
    lines.append("")
    lines.extend(render_people(report))
    return lines


def _people_on(report: Report) -> bool:
    return any("us_stocks" in r.people for r in report.results)


def render_people(report: Report) -> list[str]:
    """US stocks with the simulated holders' trading: the estimates alone, the people alone, both."""
    if not _people_on(report):
        return []
    k = report.kappa
    fit = report.fit or {}
    holdout = f"; {', '.join(str(y) for y in HOLDOUT_YEARS)} is held out and was never used to choose it"
    lines = [
        f"**US stocks with people's trading.** kappa = {k:g}"
        + (", a boundary solution" if fit.get("boundary_solution") else "")
        + (f", fitted on {FIT_YEARS[0]}-{FIT_YEARS[-1]}{holdout}" if fit else ", not fitted")
        + '. Average miss in typical daily moves; "people" is the simulated holders\' trading alone, "hybrid" '
        "adds it to the published estimates; major misses and directions are for the hybrid.",
        "",
        "| Events | Count | Estimates only | People only | Hybrid | No change | Major misses | Direction right |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, rows in groups(report):
        est = summary(report, rows)["us_stocks"]
        ppl = summary(report, rows, "people")["us_stocks"]
        hyb = summary(report, rows, "hybrid")["us_stocks"]
        if not hyb["events"]:
            continue
        lines.append(
            f"| {label} | {int(hyb['events'])} | {est['jahan_mean_miss_sd']:.2f} | {ppl['jahan_mean_miss_sd']:.2f} | "
            f"{hyb['jahan_mean_miss_sd']:.2f} | {hyb['no_change_mean_miss_sd']:.2f} | "
            f"{int(hyb['major_misses'])} of {int(hyb['events'])} | "
            f"{int(hyb['direction_hits'])} of {int(hyb['direction_calls'])} |"
        )
    lines.append("")
    return lines


def render_markdown(report: Report, years: set[int] | None = None) -> str:
    """The benchmark: a summary over every event, then per-event tables (only ``years`` when given)."""
    noise = report.file.noise_sd
    by_id = {r.event.id: r for r in report.results}
    shown = [r for r in report.results if years is None or r.event.date.year in years]
    lines: list[str] = []
    w = lines.append
    w(
        f"<!-- generated by `jahan backtest` ({report.config_name} config, {report.agents:,} people, "
        f"{report.seeds} seeds for news reach); edit data/backtests, not this table -->"
    )
    w("")
    lines.extend(render_summary(report))
    which = "" if years is None else f", {', '.join(str(y) for y in sorted(years))}"
    w(f"**Market moves on the day{which}** (percent, close to close; Jahan / real). Misses are in typical daily moves")
    w(
        f"(one daily s.d.: US stocks {noise.us_stocks:.2f}%, gold {noise.gold:.2f}%, dollar {noise.dollar:.2f}%). "
        "A major miss is more than 2 s.d., or the wrong direction on a move over 1 s.d."
    )
    w("")
    people = _people_on(report)
    w(
        "| Date | Event | Surprise | US stocks | Gold | Dollar | Miss (s.d.): stocks, gold, dollar | Major miss |"
        + (" US stocks with people / real |" if people else "")
    )
    w("|---|---|---|---|---|---|---|---|" + ("---|" if people else ""))
    for r in shown:
        ev = r.event
        if ev.kind != "forecast":
            continue
        cells, misses, majors = [], [], []
        for t in TARGETS:
            real = ev.actual.get(t)
            sim = r.jahan.get(t)
            cells.append(f"{_f(sim)} / {_f(real)}")
            if real is None or sim is None or math.isnan(sim):
                misses.append("n/a")
                continue
            s = score(sim, real, noise.get(t))
            misses.append(f"{abs(s['err_sd']):.1f}")
            if s["major"]:
                majors.append(LABELS[t])
        extra = f" {_f(r.hybrid.get('us_stocks'))} / {_f(ev.actual.us_stocks)} |" if people else ""
        w(
            f"| {ev.date.isoformat()} | {ev.title} | {_surprise_text(ev, r.surprise)} | "
            + " | ".join(cells)
            + f" | {', '.join(misses)} | {', '.join(majors) or 'none'} |"
            + extra
        )
    w("")
    sm = summary(report, shown)
    parts = []
    for t in TARGETS:
        s = sm[t]
        parts.append(
            f'{LABELS[t]}: Jahan misses by {s["jahan_mean_miss_sd"]:.2f} s.d. on average, "no change" by '
            f"{s['no_change_mean_miss_sd']:.2f}; direction called {int(s['direction_calls'])} times, right "
            f"{int(s['direction_hits'])}"
        )
    w("Summary over these events. " + ". ".join(parts) + ".")
    w("")
    hind = [r for r in shown if r.hindsight or r.side]
    if hind:
        w(
            "**After the fact, not a forecast.** The day's 2-year Treasury yield move, minus the part the decision "
            "itself explains, used as the forward-guidance surprise; and the gold estimate Jahan does not apply "
            "(its sign is disputed)."
        )
        w("")
        w(
            "| Date | Event | 2-year yield | US stocks with hindsight / real | Dollar with hindsight / real "
            "| Gold, unused estimate / real |"
        )
        w("|---|---|---|---|---|---|")
        for r in hind:
            ev = r.event
            two = f"{ev.two_year_bp:+.0f} bp" if ev.two_year_bp is not None else "n/a"
            w(
                f"| {ev.date.isoformat()} | {ev.title} | {two} | {_f(r.hindsight.get('us_stocks'))} / "
                f"{_f(ev.actual.us_stocks)} | {_f(r.hindsight.get('dollar'))} / {_f(ev.actual.dollar)} | "
                f"{_f(r.side.get('gold'))} / {_f(ev.actual.gold)} |"
            )
        w("")
    other = [r.event for r in shown if r.event.kind == "not_modelled"]
    if other:
        w("**Not modelled.** No published estimate exists, so Jahan moves nothing unless a prompt scripts the move.")
        w("")
        w("| Date | Event | US stocks | Gold | Dollar |")
        w("|---|---|---|---|---|")
        for ev in other:
            w(
                f"| {ev.date.isoformat()} | {ev.title} | {_f(ev.actual.us_stocks)} | {_f(ev.actual.gold)} | "
                f"{_f(ev.actual.dollar)} |"
            )
        w("")
    for chk in report.file.society:
        hit = by_id.get(chk.event)
        if hit is None or not hit.aware or (years is not None and hit.event.date.year not in years):
            continue
        arr = np.array(hit.aware, dtype=np.float64)  # seeds x days
        mean = arr.mean(axis=0)
        last = arr[:, min(chk.days_after, arr.shape[1] - 1)]
        inside = chk.low <= float(last.mean()) <= chk.high
        headline = hit.event.announcement.headline if hit.event.announcement else ""
        w(
            f'**News reach.** Share of US adults who know the headline "{headline}": '
            + ", ".join(f"{100 * v:.1f}% by the end of day {d}" for d, v in enumerate(mean))
            + f" (day 0 is the release day; mean of {arr.shape[0]} seeds, day {chk.days_after} range "
            f"{100 * last.min():.1f}-{100 * last.max():.1f}%). Survey band {100 * chk.low:.0f}-{100 * chk.high:.0f}%: "
            + ("inside." if inside else "outside, a major discrepancy.")
        )
        if chk.note:
            w("")
            w(chk.note)
        w("")
    return "\n".join(lines).rstrip() + "\n"
