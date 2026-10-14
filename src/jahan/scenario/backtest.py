"""Backtests: replay real announcements in a world dated the day before, and compare with what happened.

Jahan has no market model.  On an announcement, US stocks, gold and the euro's dollar rate move by published
event-study estimates times the surprise (data/global/markets.yaml), so a backtest tests those estimates as
Jahan applies them, next to a naive "no change" forecast.  Separately it measures how far the news spreads
among the simulated people.  The backtest world holds people, political camps and public figures, but no
news from before the event.  Deterministic and offline: no LLM, no network.
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


def default_events_file() -> Path:
    return paths.data_dir() / "backtests" / "events_2026.yaml"


@dataclass
class EventResult:
    event: BacktestEvent
    surprise: float | None
    jahan: dict[str, float] = field(default_factory=dict)  # simulated release-day % (engine run)
    hindsight: dict[str, float] = field(default_factory=dict)  # with the day's 2-year move as guidance
    side: dict[str, float] = field(default_factory=dict)  # inactive estimates, shown for comparison
    aware: list[list[float]] = field(default_factory=list)  # per seed: share of US adults aware, by day after


def _config(base: RunConfig, date: dt.date, seed_offset: int) -> RunConfig:
    cfg = base.model_copy(deep=True)
    cfg.start_date = date - dt.timedelta(days=1)
    cfg.seed = base.seed + seed_offset
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


def simulate(event: BacktestEvent, base: RunConfig, seed_offset: int = 0, days_after: int = 2) -> EventResult:
    """Run one event in a fresh world dated the day before: market moves on the day, and how many US adults
    know the headline at the end of each day from the release day to ``days_after`` days later."""
    assert event.announcement is not None
    cfg = _config(base, event.date, seed_offset)
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
            res.jahan = {
                "us_stocks": _pct(float(world.macro["us_stocks_index"]), stocks0),
                "gold": _pct(float(world.macro["gold_index"]), gold0),
                "dollar": _pct(world.macro["fx"][codes.index("DEU")], fx0[codes.index("DEU")])
                if "DEU" in codes
                else float("nan"),
            }
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


def run_backtests(
    file: BacktestFile, base: RunConfig, seeds: int = 3, progress: Callable[[str], None] | None = None
) -> Report:
    """Every event in the file.  Forecast events run in a fresh world (``seeds`` worlds for events with a
    news-reach check); after-the-fact and unmodelled events need no world."""
    markets = load_markets(paths.data_dir())
    if markets is None:
        raise ValueError("data/global/markets.yaml is missing")
    results = []
    society_events = {s.event for s in file.society}
    for ev in file.events:
        if ev.kind == "not_modelled" or ev.announcement is None:
            results.append(EventResult(ev, None))
            continue
        if progress:
            progress(f"{ev.date.isoformat()} {ev.title}")
        if ev.kind == "hindsight_only":
            res = EventResult(ev, ev.announcement.actual - ev.announcement.expected)
        else:
            res = simulate(ev, base, 0)
            for s in range(1, seeds if ev.id in society_events else 1):
                res.aware += simulate(ev, base, s).aware
        res.hindsight, res.side = hindsight(ev, markets)
        results.append(res)
    return Report(file, results, base.n_agents, seeds, base.name)


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


def summary(report: Report) -> dict[str, dict[str, float]]:
    """Per market: mean miss in daily s.d. of Jahan and of "no change", and direction calls and hits."""
    noise = report.file.noise_sd
    out: dict[str, dict[str, float]] = {}
    for t in TARGETS:
        errs, naive, calls, hits = [], [], 0, 0
        for r in report.results:
            if r.event.kind != "forecast":
                continue
            real = r.event.actual.get(t)
            sim = r.jahan.get(t)
            if real is None or sim is None or math.isnan(sim):
                continue
            sd = noise.get(t)
            errs.append(abs(sim - real) / sd)
            naive.append(abs(real) / sd)
            if abs(sim) >= SIGN_FORECAST_SD * sd:
                calls += 1
                hits += int(np.sign(sim) == np.sign(real))
        out[t] = {
            "events": float(len(errs)),
            "jahan_mean_miss_sd": float(np.mean(errs)) if errs else float("nan"),
            "no_change_mean_miss_sd": float(np.mean(naive)) if naive else float("nan"),
            "direction_calls": float(calls),
            "direction_hits": float(hits),
        }
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
        "society": [s.model_dump() for s in report.file.society],
    }


def render_markdown(report: Report) -> str:
    """The benchmark tables (used for the README)."""
    noise = report.file.noise_sd
    by_id = {r.event.id: r for r in report.results}
    lines: list[str] = []
    w = lines.append
    w(
        f"<!-- generated by `jahan backtest` ({report.config_name} config, {report.agents:,} people, "
        f"{report.seeds} seeds for news reach); edit data/backtests, not this table -->"
    )
    w("")
    w("**Market moves on the day** (percent, close to close; Jahan / real). Misses are in typical daily moves")
    w(
        f"(one daily s.d.: US stocks {noise.us_stocks:.2f}%, gold {noise.gold:.2f}%, dollar {noise.dollar:.2f}%). "
        "A major miss is more than 2 s.d., or the wrong direction on a move over 1 s.d."
    )
    w("")
    w("| Date | Event | Surprise | US stocks | Gold | Dollar | Miss (s.d.): stocks, gold, dollar | Major miss |")
    w("|---|---|---|---|---|---|---|---|")
    for r in report.results:
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
        w(
            f"| {ev.date.isoformat()} | {ev.title} | {_surprise_text(ev, r.surprise)} | "
            + " | ".join(cells)
            + f" | {', '.join(misses)} | {', '.join(majors) or 'none'} |"
        )
    w("")
    sm = summary(report)
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
    hind = [r for r in report.results if r.hindsight or r.side]
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
    other = [r.event for r in report.results if r.event.kind == "not_modelled"]
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
        if hit is None or not hit.aware:
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
