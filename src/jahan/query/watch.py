"""The watch list: series a person follows while time moves (`jahan watch`, the Watch page).

Read-only over the world: the list itself lives in ``runs/<name>/watch.json`` (outside the world state, so
watching never changes a world's fingerprint), and each series' history is read back from the timeline's daily
metrics (``runs/<name>/metrics/metrics_*.parquet``).  A reset deletes the metric chunks after the reset day,
so history always matches the world on disk.

Series ids:
  market:us_stocks        US stock index (S&P 500), 100 = the world's start; two parts: announcements and
                          scripted paths, and the simulated holders' trading
  market:us_tech_stocks   US tech stocks (Nasdaq-100), derived: beta x each day's US stock move
  market:gold             gold index, 100 = the world's start
  market:dollar           euros per US dollar (worlds with Germany)
  market:fx:<ISO3>        a country's currency per US dollar
  mood:<ISO3>             average economic mood of the people in a country, -1..1
  odds:<id>               simulated odds of a watched question (see query/odds.py)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from jahan import paths
from jahan.model.investors import load_investors
from jahan.model.world import World
from jahan.sim.market import tech_index
from jahan.store.eventlog import chunk_range, list_chunks
from jahan.store.rundir import RunDir

DISCLAIMER = (
    "Simulated, not a forecast: educational and testing use only; not financial advice; no warranty. "
    "US stocks move by published estimates on announcements, scripted what-ifs, and the simulated holders' "
    "trading; event odds are the average belief of simulated traders."
)
WATCH_FILE = "watch.json"


@dataclass
class SeriesSpec:
    id: str
    kind: str  # market | mood | odds
    label: str
    metric: str
    country: int = -1
    item: int = -1
    unit: str = "index"  # index | rate | mood | probability
    components: bool = False  # US stocks: people's part and the announcements' part
    derived_beta: float | None = None  # derived from US stocks
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- watch file


def load_list(rd: RunDir) -> dict[str, dict[str, Any]]:
    data = rd._read(WATCH_FILE, {"version": 1, "series": {}})
    return dict(data.get("series", {}))


def save_list(rd: RunDir, series: dict[str, dict[str, Any]]) -> None:
    rd._write(WATCH_FILE, {"version": 1, "series": series})


# ------------------------------------------------------------------------------------------------- catalog


def _tech_beta(world: World) -> float | None:
    market = world.macro.get("market")
    if market is not None:
        beta = market["cfg"].get("tech_beta")
        return None if beta is None else float(beta)
    inv = load_investors(paths.data_dir())  # worlds made before people traded: the current coefficient
    tech = None if inv is None else inv.derived.get("us_tech_stocks")
    return None if tech is None or tech.beta is None else float(tech.beta)


def odds_specs(world: World) -> list[SeriesSpec]:
    """Watched questions as series (filled in by query/odds.py once questions exist)."""
    try:
        from jahan.query.odds import question_specs
    except ImportError:
        return []
    return question_specs(world)


def catalog(world: World) -> list[SeriesSpec]:
    codes = [c.iso3 for c in world.countries]
    out = [
        SeriesSpec(
            "market:us_stocks",
            "market",
            "US stocks (S&P 500)",
            "us_stocks_index",
            components="market" in world.macro,
            note="announcements and scripted moves, plus the simulated holders' trading"
            if "market" in world.macro
            else "announcements and scripted moves (this world was made before people traded)",
        )
    ]
    beta = _tech_beta(world)
    if beta is not None:
        out.append(
            SeriesSpec(
                "market:us_tech_stocks",
                "market",
                "US tech stocks (Nasdaq-100, derived)",
                "us_stocks_index",
                derived_beta=beta,
                components="market" in world.macro,
                note=f"derived: {beta:g} x each day's US stock move, not modelled on its own",
            )
        )
    out.append(SeriesSpec("market:gold", "market", "Gold", "gold_index"))
    if "DEU" in codes:
        out.append(
            SeriesSpec(
                "market:dollar",
                "market",
                "US dollar (euros per dollar)",
                "fx_local_per_usd",
                country=codes.index("DEU"),
                unit="rate",
                note="up: the dollar strengthens",
            )
        )
    for k, c in enumerate(world.countries):
        if c.currency != "USD" and c.iso3 != "DEU":
            out.append(
                SeriesSpec(
                    f"market:fx:{c.iso3}",
                    "market",
                    f"{c.currency} per US dollar ({c.name})",
                    "fx_local_per_usd",
                    country=k,
                    unit="rate",
                    note=f"up: the {c.currency} weakens",
                )
            )
    for k, c in enumerate(world.countries):
        out.append(
            SeriesSpec(
                f"mood:{c.iso3}",
                "mood",
                f"Economic mood in {c.name}",
                "sentiment",
                country=k,
                unit="mood",
                note="average of people in the country, -1 (gloomy) to 1 (upbeat); drives people's trading",
            )
        )
    return out + odds_specs(world)


def resolve(world: World, series_id: str) -> SeriesSpec:
    cat = catalog(world)
    for s in cat:
        if s.id == series_id:
            return s
    raise ValueError(f"unknown series {series_id!r}; this world has: " + ", ".join(s.id for s in cat))


# --------------------------------------------------------------------------------------------- read values


def current_value(world: World, spec: SeriesSpec) -> float:
    m = world.macro
    if spec.kind == "odds":
        return float(spec.extra.get("now", float("nan")))
    if spec.metric == "us_stocks_index":
        level = float(m.get("us_stocks_index", 100.0))
        if spec.derived_beta is None:
            return level
        return float("nan")  # derived from the history (see show)
    if spec.metric == "gold_index":
        return float(m.get("gold_index", 100.0))
    if spec.metric == "fx_local_per_usd":
        return float(m["fx"][spec.country])
    if spec.metric == "sentiment":
        pop = world.pop
        w = pop["pop_weight"].astype(np.float64) * pop["alive"] * (pop["location_country"] == spec.country)
        return float(np.average(pop["sentiment"], weights=w)) if w.sum() > 0 else float("nan")
    raise ValueError(f"no current value for {spec.id}")


def read_metric_series(
    rd: RunDir, keys: set[tuple[str, int, int]], first_day: int = 0
) -> dict[tuple[str, int, int], dict[int, float]]:
    """{(metric, country, item): {day: value}} from the timeline's metric chunks, days >= ``first_day``."""
    out: dict[tuple[str, int, int], dict[int, float]] = {k: {} for k in keys}
    names = sorted({k[0] for k in keys})
    if not names:
        return out
    for path in list_chunks(rd.metrics, "metrics"):
        if chunk_range(path)[1] < first_day:
            continue
        t = pq.read_table(path, filters=[("metric", "in", names)]).to_pydict()
        for d, c, it, name, v in zip(t["day"], t["country"], t["item"], t["metric"], t["value"], strict=True):
            key = (name, int(c), int(it))
            if key in out and d >= first_day:
                out[key][int(d)] = float(v)
    return out


def _keys(spec: SeriesSpec) -> set[tuple[str, int, int]]:
    keys = {(spec.metric, spec.country, spec.item)}
    if spec.components:
        keys |= {("us_stocks_people_pct", -1, -1), ("us_stocks_scripted_pct", -1, -1)}
    return keys


def _history(world: World, spec: SeriesSpec, data: dict[tuple[str, int, int], dict[int, float]]) -> list[dict]:
    """Every recorded day of a series (day, date, value and, for US stocks, the day's two parts)."""
    vals = data[(spec.metric, spec.country, spec.item)]
    days = sorted(vals)
    people = data.get(("us_stocks_people_pct", -1, -1), {})
    scripted = data.get(("us_stocks_scripted_pct", -1, -1), {})
    if spec.derived_beta is not None:
        prev, total = 100.0, []
        for d in days:
            total.append(100.0 * (vals[d] / prev - 1.0))
            prev = vals[d]
        levels = tech_index(total, spec.derived_beta)
        series = dict(zip(days, levels, strict=True))
    else:
        series = vals
    rows = []
    for d in days:
        row: dict[str, Any] = {"day": d, "date": world.date(d).isoformat(), "value": series[d]}
        if spec.components and d in people:
            b = spec.derived_beta or 1.0
            row["people_pct"] = b * people[d]
            row["scripted_pct"] = b * scripted.get(d, 0.0)
        rows.append(row)
    return rows


def _value_before(
    world: World, spec: SeriesSpec, hist: list[dict], since_day: int, now: float, stored: float | None = None
) -> float:
    """The series' value when it was watched: the end of the day before ``since_day`` (or the value stored
    when it was added, e.g. a question's starting odds)."""
    before = [r for r in hist if r["day"] < since_day]
    if before:
        return float(before[-1]["value"])
    if stored is not None:
        return float(stored)
    if since_day == 0:  # watched from the world's start
        if spec.unit == "index":
            return 100.0
        if spec.unit == "rate":
            return float(world.macro["fx0"][spec.country])
        if spec.unit == "mood":
            return 0.0
    return float(hist[0]["value"]) if hist else now


def _compound(pcts: list[float]) -> float:
    level = 1.0
    for p in pcts:
        level *= 1.0 + p / 100.0
    return 100.0 * (level - 1.0)


def show(rd: RunDir, world: World, days: int = 60, only: list[str] | None = None) -> dict[str, Any]:
    """The watch list with each series' recent history, its change since it was watched, and the parts."""
    watched = load_list(rd)
    cat = catalog(world)
    by_id = {s.id: s for s in cat}
    ids = [i for i in (only or list(watched)) if i in by_id]
    if only is None:  # questions are watched from the day they were opened, by `watch add` or a what-if
        for s in cat:
            if s.kind == "odds" and s.id not in watched:
                ids.append(s.id)
                watched[s.id] = {"since_day": int(s.extra.get("added_day", 0))}
    specs = [by_id[i] for i in ids]
    keys: set[tuple[str, int, int]] = set()
    for s in specs:
        keys |= _keys(s)
    data = read_metric_series(rd, keys, 0)
    first_shown = max(0, world.day - days)
    qrows: dict[str, Any] = {}
    if any(s.kind == "odds" for s in specs):
        from jahan.query.odds import question_rows

        qrows = {r["series"]: r for r in question_rows(rd, world)}
    series = []
    for s in specs:
        hist = _history(world, s, data)
        entry = watched.get(s.id, {"since_day": world.day})
        since_day = int(entry.get("since_day", world.day))
        rebased = since_day > world.day
        if rebased:
            since_day = world.day
        now = hist[-1]["value"] if hist else (100.0 if s.derived_beta is not None else current_value(world, s))
        stored = entry.get("since_value") if not rebased else None
        since_value = _value_before(world, s, hist, since_day, now, None if stored is None else float(stored))
        after = [r for r in hist if r["day"] >= since_day]
        comp = None
        if s.components:
            comp = {
                "people_pct": _compound([r.get("people_pct", 0.0) for r in after]),
                "scripted_pct": _compound([r.get("scripted_pct", 0.0) for r in after]),
            }
        pct = None if s.unit in ("mood", "probability") or not since_value else 100.0 * (now / since_value - 1.0)
        series.append(
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "unit": s.unit,
                "note": s.note,
                "since_day": since_day,
                "since_date": world.date(since_day).isoformat(),
                "since_value": since_value,
                "rebased": rebased,
                "now": now,
                "since_watched_pct": pct,
                "since_watched_abs": now - since_value,
                "components": comp,
                "question": qrows.get(s.id),
                "history": [r for r in hist if r["day"] >= first_shown],
            }
        )
    return {
        "day": world.day,
        "date": world.date().isoformat(),
        "days": days,
        "disclaimer": DISCLAIMER,
        "catalog": [
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "unit": s.unit,
                "watched": s.id in watched,
                "components": s.components,
                "note": s.note,
            }
            for s in cat
        ],
        "series": series,
    }


def add(rd: RunDir, world: World, ids: list[str], since_values: dict[str, float] | None = None) -> list[str]:
    """Watch these series from today (a question must be opened first, as a prompt: `jahan watch add odds:...`)."""
    watched = load_list(rd)
    added = []
    for i in ids:
        resolve(world, i)
        if i not in watched or watched[i].get("since_day", -1) > world.day:
            entry: dict[str, Any] = {"since_day": world.day, "added": dt.date.today().isoformat()}
            if since_values and i in since_values:
                entry["since_value"] = since_values[i]
            watched[i] = entry
            added.append(i)
    save_list(rd, watched)
    return added


def remove(rd: RunDir, ids: list[str]) -> list[str]:
    watched = load_list(rd)
    gone = [i for i in ids if watched.pop(i, None) is not None]
    save_list(rd, watched)
    return gone


def market_levels(world: World) -> dict[str, Any]:
    """Current market levels for `status --json`."""
    m = world.macro
    market = m.get("market")
    codes = [c.iso3 for c in world.countries]
    out: dict[str, Any] = {
        "us_stocks": {"index": float(m.get("us_stocks_index", 100.0))},
        "us_tech_stocks": None,
        "gold": {"index": float(m.get("gold_index", 100.0))},
        "dollar": {"eur_per_usd": float(m["fx"][codes.index("DEU")])} if "DEU" in codes and "fx" in m else None,
        "people_trading": market is not None,
        "people_on": market is not None and bool(world.config.params.market_people),
        "kappa": None if market is None else float(market["cfg"]["kappa"]),
    }
    if market is not None:
        out["us_stocks"].update(
            {
                "scripted_index": float(market["scripted_index"]),
                "people_index": float(market["people_index"]),
                "last": dict(market["last"]),
            }
        )
    beta = _tech_beta(world)
    if beta is not None:
        out["us_tech_stocks"] = {"beta": beta, "derived": True}
    return out
