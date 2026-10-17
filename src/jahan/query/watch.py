"""The watch list: series a person follows while time moves (`jahan watch`, the Watch page).

Read-only over the world: the list itself lives in ``runs/<name>/watch.json`` (outside the world state, so
watching never changes a world's fingerprint), and each series' history is read back from the timeline's daily
metrics (``runs/<name>/metrics/metrics_*.parquet``).  A reset deletes the metric chunks after the reset day,
so history always matches the world on disk.

Series ids:
  market:us_stocks        US stock index (S&P 500), 100 = the world's start; parts: announcements and
                          scripted paths, and the simulated holders' trading (with everyday activity: their
                          news reactions, and saving and cashing out); also market:stocks:USA
  market:stocks:<ISO3>    another country's stock index moved by its residents (worlds with everyday activity)
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
ALIASES = {"market:stocks:USA": "market:us_stocks"}
COMPONENTS = {  # part -> what it means (shown on the Watch page)
    "people": "people's trading",
    "news": "news reactions",
    "savings": "saving and cashing out",
    "scripted": "announcements and scripted",
    "time": "time passing (heuristic)",
    "talk": "traders talking (heuristic)",
}


@dataclass
class SeriesSpec:
    id: str
    kind: str  # market | mood | odds
    label: str
    metric: str
    country: int = -1
    item: int = -1
    unit: str = "index"  # index | rate | mood | probability
    components: tuple[str, ...] = ()  # the parts of a market's daily move (see COMPONENTS)
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
    economy = world.macro.get("economy")
    us_parts = _us_parts(world)
    out = [
        SeriesSpec(
            "market:us_stocks",
            "market",
            "US stocks (S&P 500)",
            "us_stocks_index",
            components=us_parts,
            note=(
                "announcements and scripted moves, plus the simulated holders' news reactions (as they look at "
                "their accounts) and cash-outs that differ from normal"
                if economy is not None
                else "announcements and scripted moves, plus the simulated holders' trading"
                if "market" in world.macro
                else "announcements and scripted moves (this world was made before people traded)"
            ),
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
                components=us_parts,
                note=f"derived: {beta:g} x each day's US stock move, not modelled on its own",
            )
        )
    for iso, mk in (economy or {}).get("markets", {}).items():
        if iso == "USA" or iso not in codes:
            continue
        mc = mk["cfg"]
        out.append(
            SeriesSpec(
                f"market:stocks:{iso}",
                "market",
                f"{mc['label']} ({mc['name']})",
                "stocks_index",
                country=codes.index(iso),
                components=("news",),
                note=_local_note(mc, mk["last"]),
                extra={"illustrative": bool(mc.get("illustrative")) or bool(mk["last"].get("few_holders"))},
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


def _us_parts(world: World) -> tuple[str, ...]:
    if "economy" in world.macro and "USA" in world.macro["economy"]["markets"]:
        return ("news", "savings", "scripted")
    return ("people", "scripted") if "market" in world.macro else ()


def _local_note(mc: dict[str, Any], last: dict[str, Any]) -> str:
    parts = [f"moved by its residents' trading, 100 = the day everyday activity started; {mc['currency']}, nominal"]
    if mc.get("borrowed"):
        parts.append("borrowed from the US: " + ", ".join(mc["borrowed"]))
    reasons = list(mc.get("reasons", []))
    if last.get("few_holders"):
        reasons.append(f"only {last['holders']} simulated holders: a few people carry it (a bigger world helps)")
    if reasons:
        parts.append("illustrative: " + "; ".join(reasons))
    return "; ".join(parts)


def resolve(world: World, series_id: str) -> SeriesSpec:
    series_id = ALIASES.get(series_id, series_id)
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
    if spec.metric == "stocks_index":
        iso = world.countries[spec.country].iso3
        return float(m["economy"]["markets"][iso]["index"])
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


def _part_keys(world: World, spec: SeriesSpec) -> dict[str, list[tuple[str, int, int]]]:
    """Metric rows behind each part, in order of preference (the first one recorded on a day is used: days
    before everyday activity started have only the older engine's people's part)."""
    if spec.metric == "stocks_index":
        return {"news": [("stocks_news_pct", spec.country, -1)], "savings": [("stocks_savings_pct", spec.country, -1)]}
    if spec.kind == "odds":  # changes in probability points, one row per part and day
        qid = spec.metric[len("odds.") :]
        return {p: [(f"odds_{p}.{qid}", -1, spec.item)] for p in ("time", "news", "talk")}
    codes = [c.iso3 for c in world.countries]
    us = codes.index("USA") if "USA" in codes else -1
    return {
        "people": [("us_stocks_people_pct", -1, -1)],
        "news": [("stocks_news_pct", us, -1), ("us_stocks_people_pct", -1, -1)],
        "savings": [("stocks_savings_pct", us, -1)],
        "scripted": [("us_stocks_scripted_pct", -1, -1)],
    }


def _keys(world: World, spec: SeriesSpec) -> set[tuple[str, int, int]]:
    keys = {(spec.metric, spec.country, spec.item)}
    parts = _part_keys(world, spec)
    for name in spec.components:
        keys |= set(parts[name])
    return keys


def _history(world: World, spec: SeriesSpec, data: dict[tuple[str, int, int], dict[int, float]]) -> list[dict]:
    """Every recorded day of a series (day, date, value and, for US stocks, the day's two parts)."""
    vals = data[(spec.metric, spec.country, spec.item)]
    days = sorted(vals)
    parts = _part_keys(world, spec)
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
        b = spec.derived_beta or 1.0
        for name in spec.components:
            got = next((data[k][d] for k in parts[name] if d in data.get(k, {})), None)
            if got is not None:
                row[f"{name}_{'pts' if spec.kind == 'odds' else 'pct'}"] = b * got
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


# below this a move counts as "nothing moved": a twentieth of a typical day for markets (so the rare small steps of
# saving and cashing out do not hide the explanation), half a thousandth of mood, a twentieth of a point of odds
QUIET = {"index": 0.05, "rate": 0.05, "mood": 0.0005, "probability": 0.0005}
ACTIVITY_STATS = ("media_exposures", "wom_aware", "shares", "reposts", "actions_delivered", "odds_beliefs_moved")


def _moved(s: dict[str, Any], value_then: float) -> bool:
    now = s["now"]
    if value_then != value_then or now != now:  # NaN
        return False
    if s["unit"] in ("mood", "probability"):
        return abs(now - value_then) >= QUIET[s["unit"]]
    return value_then != 0 and abs(100.0 * (now / value_then - 1.0)) >= QUIET[s["unit"]]


def quiet_explanation(
    rd: RunDir, world: World, series: list[dict[str, Any]], from_day: int, llm_ready: bool
) -> dict[str, Any]:
    """Why nothing watched moved between ``from_day`` and today, in plain sentences (empty when something moved).

    ``series`` are the rows of :func:`show`; a series counts as moved when it changed by more than ``QUIET``
    from the end of the day before ``from_day``."""
    to_day = world.day
    out: dict[str, Any] = {"from_day": from_day, "to_day": to_day, "quiet": False, "sentences": []}
    if not series or from_day >= to_day:
        return out
    for s in series:
        before = [r for r in s["history"] if r["day"] < from_day]
        then = float(before[-1]["value"]) if before else float(s["since_value"])
        if _moved(s, then):
            return out
    out["quiet"] = True
    say: list[str] = out["sentences"]
    span = f"between day {from_day} and day {to_day}"
    added = [e for e in rd.ledger() if from_day <= int(e["day"]) < to_day]
    act = read_metric_series(rd, {(k, -1, -1) for k in ACTIVITY_STATS}, from_day)
    heard = sum(v for k in ("media_exposures", "wom_aware") for d, v in act[(k, -1, -1)].items() if d < to_day)
    if not added:
        say.append(f"Nothing was added {span}: no what-if, news or question.")
    if heard == 0:
        say.append("Nobody heard any news in these days: no story was being reported or passed on.")
    else:
        say.append(f"Stories still reached people about {int(heard):,} times, but none moved what you watch.")
    ids = [str(s["id"]) for s in series]
    everyday = "economy" in world.macro
    stocks = any(i in ("market:us_stocks", "market:us_tech_stocks") or i.startswith("market:stocks:") for i in ids)
    if stocks and everyday:
        say.append(
            "People act on news as they look at their accounts; with no news nobody's view changes, so only "
            "cash-outs that differ from normal (US) move stocks, and those are rare on any one day."
        )
    elif stocks:
        if "market" not in world.macro:
            say.append(
                "This world was made before people's trading existed, so its US stocks move only on "
                "announcements and what-ifs."
            )
        else:
            say.append("People here buy and sell stocks only when news reaches them; with no news, nobody trades.")
        say.append("This world was made before everyday activity: turn it on with `jahan economy on`.")
    if any(i in ("market:gold", "market:dollar") or i.startswith("market:fx:") for i in ids):
        say.append("Gold and currencies move only when a what-if or an announcement moves them.")
    if any(i.startswith("mood:") for i in ids):
        say.append(
            "Without news, moods move only a little: with job losses and new jobs, and as people talk."
            if everyday
            else "Moods change only with the news people hear."
        )
    if any(i.startswith("odds:") for i in ids):
        say.append("No story bore on the watched questions.")
        if not llm_ready:
            say.append(
                "Claude is not set up, so new headlines are not rated against your questions; only what-ifs "
                "that name a question move it."
            )
    if not llm_ready:
        say.append("Claude is not set up, so public figures did not act on their own (see `jahan doctor`).")
    say.append("To see movement, add a what-if (for example a news story or a market move), then move time forward.")
    return out


def show(
    rd: RunDir,
    world: World,
    days: int = 60,
    only: list[str] | None = None,
    since_day: int | None = None,
    llm_ready: bool = True,
) -> dict[str, Any]:
    """The watch list with each series' recent history, its change since it was watched, and the parts.

    With ``since_day``, also why nothing moved from that day to today (``explanation``), e.g. the day before the
    last time step."""
    explain_from = since_day
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
        keys |= _keys(world, s)
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
        parts = [{"name": n, "label": COMPONENTS[n]} for n in s.components]
        if s.components and s.kind == "odds":  # probability points add up; the rest is word of mouth and turnover
            comp = {f"{n}_pts": float(sum(r.get(f"{n}_pts", 0.0) for r in after)) for n in s.components}
            comp["other_pts"] = (now - since_value) - sum(comp.values())
            parts.append({"name": "other", "label": "word of mouth and other"})
        elif s.components:
            comp = {f"{name}_pct": _compound([r.get(f"{name}_pct", 0.0) for r in after]) for name in s.components}
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
                "parts": parts,
                "illustrative": bool(s.extra.get("illustrative")),
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
                "components": list(s.components),
                "note": s.note,
            }
            for s in cat
        ],
        "series": series,
        "explanation": None if explain_from is None else quiet_explanation(rd, world, series, explain_from, llm_ready),
    }


def add(rd: RunDir, world: World, ids: list[str], since_values: dict[str, float] | None = None) -> list[str]:
    """Watch these series from today (a question must be opened first, as a prompt: `jahan watch add odds:...`)."""
    watched = load_list(rd)
    added = []
    for i in ids:
        i = resolve(world, i).id
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
    economy = m.get("economy")
    out["everyday"] = None
    if economy is not None:
        out["everyday"] = {
            "since_day": int(economy["since_day"]),
            "markets": {
                iso: {
                    "label": mk["cfg"]["label"],
                    "name": mk["cfg"]["name"],
                    "index": float(m["us_stocks_index"]) if iso == "USA" else float(mk["index"]),
                    "illustrative": bool(mk["cfg"].get("illustrative")),
                    "last": {k: v for k, v in mk["last"].items() if k != "saving"},
                }
                for iso, mk in economy["markets"].items()
            },
        }
    return out
