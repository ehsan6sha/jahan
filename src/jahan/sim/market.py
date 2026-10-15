"""Tick phase 5b: people's trading moves the US stock index.

The index has two parts.  Announcements (published estimates times the surprise) and scripted paths move it in
phase 0 (``sim/world_events.apply_macro``).  After the mood phase, simulated US holders of stock trade on the
news they learned that day: on how much better or worse the news they know makes things look
(``news_mood``, the level their mood drifts toward) compared with the morning, plus any scripted mood shock:

    flow  = sum over traders of  pop_weight x holdings x kappa x (news mood tonight - news mood this morning)
    move% = 100 x multiplier x flow / (sum over holders of pop_weight x holdings), capped at max_daily_pct

People trade when they hear the news, not as their mood slowly drifts afterwards (a modelling choice: markets
react when news arrives).  Holders and their holdings come from the Survey of Consumer Finances
(``model/investors.py``); the multiplier from Gabaix and Koijen (2021); kappa is the one fitted number.  The
simulated holders stand in for all US household equity; institutions are not modelled, and there is no random
daily noise.  When a story stops being tracked its effect on news mood ends, so people buy back what they sold
on bad news: this part reverts and cannot make lasting trends.

The market trades on weekdays: the weekend's net flow is carried to Monday (US market holidays are not
modelled).  Worlds made before this phase existed have no ``macro["market"]`` and skip it, so their fingerprints
and recorded sessions stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EventKind
from jahan.model.investors import investor_masks
from jahan.model.world import World
from jahan.sim.params import SimParams
from jahan.sim.world_events import news_mood, stocks_headline
from jahan.store.eventlog import EventLog

LOG_MIN_PCT = 0.01  # people-driven moves smaller than this are not logged as events (metrics keep them)


@dataclass
class DayOpen:
    news: NDArray[np.float64] | None  # news mood of every row at the start of the tick
    index: float  # US stock index at the start of the tick
    shift: list[float] = field(default_factory=list)  # scripted mood shocks so far, per country


def enabled(world: World) -> bool:
    return "market" in world.macro and "us_stocks_index" in world.macro


def open_day(world: World) -> DayOpen:
    if not enabled(world):
        return DayOpen(None, 0.0)
    shift = [float(x) for x in world.macro.get("sentiment_shift", [])]
    return DayOpen(news_mood(world), float(world.macro["us_stocks_index"]), shift)


def settings(world: World, params: SimParams) -> tuple[float, float, bool, float]:
    """(kappa, multiplier, requires_internet, max_daily_pct): the world's copy, overridden by params."""
    cfg = world.macro["market"]["cfg"]
    kappa = float(cfg["kappa"]) if params.market_kappa is None else float(params.market_kappa)
    mult = float(cfg["multiplier"]) if params.market_multiplier is None else float(params.market_multiplier)
    need_net = (
        bool(cfg["requires_internet"]) if params.market_requires_internet is None else params.market_requires_internet
    )
    return kappa, mult, need_net, float(cfg["max_daily_pct"])


def signal(world: World, opened: DayOpen) -> NDArray[np.float64]:
    """Per person: today's change in news mood, plus today's scripted mood shock in their country."""
    assert opened.news is not None
    pop = world.pop
    n0 = opened.news.shape[0]
    out = np.zeros(pop.n)
    out[:n0] = news_mood(world)[:n0] - opened.news  # people born today: no change
    shift = np.asarray(world.macro.get("sentiment_shift", []), dtype=np.float64)
    if shift.size and len(opened.shift) == shift.size:
        out += (shift - np.asarray(opened.shift))[pop["country"].astype(np.int64)]
    return out


def people_move(world: World, params: SimParams, opened: DayOpen) -> tuple[float, float, int, int]:
    """(percent move before the cap, net flow in US dollars, holders, traders) from today's news."""
    kappa, mult, need_net, _cap = settings(world, params)
    inv = investor_masks(world, world.macro["market"]["cfg"], need_net)
    w = world.pop["pop_weight"].astype(np.float64)
    held = float((w * inv.holdings)[inv.holds].sum())
    holders, traders = int(inv.holds.sum()), int(inv.trades.sum())
    if held <= 0.0 or kappa == 0.0:
        return 0.0, 0.0, holders, traders
    flow = float((w * inv.holdings * kappa * signal(world, opened))[inv.trades].sum())
    return 100.0 * mult * flow / held, flow, holders, traders


def close_day(world: World, params: SimParams, log: EventLog, opened: DayOpen) -> dict[str, int]:
    """Split today's index move into its two parts and add people's trading."""
    if opened.news is None or not enabled(world):
        return {}
    m = world.macro["market"]
    index = float(world.macro["us_stocks_index"])
    scripted = 100.0 * (index / opened.index - 1.0) if opened.index > 0 else 0.0
    m["scripted_index"] = float(m["scripted_index"]) * (1.0 + scripted / 100.0)
    last = m["last"]
    people, flow = 0.0, 0.0
    holders, traders = int(last.get("holders", 0)), int(last.get("traders", 0))
    if params.market_people:
        raw, flow, holders, traders = people_move(world, params, opened)
        pending = float(m.get("pending_pct", 0.0)) + raw
        if world.dow() < 5:  # the market is open: today's flow and the weekend's
            people = float(np.clip(pending, -settings(world, params)[3], settings(world, params)[3]))
            pending = 0.0
        m["pending_pct"] = pending
        if people != 0.0:
            after = index * (1.0 + people / 100.0)
            world.macro["us_stocks_index"] = after
            m["people_index"] = float(m["people_index"]) * (1.0 + people / 100.0)
            if abs(people) >= LOG_MIN_PCT:
                log.add(world.day, EventKind.MACRO_UPDATE, -1, -1, -1, after, -1, "us_stocks_people")
            stocks_headline(world, index, after, log)
    total = 100.0 * (float(world.macro["us_stocks_index"]) / opened.index - 1.0) if opened.index > 0 else 0.0
    m["last"] = {
        "day": world.day,
        "scripted_pct": scripted,
        "people_pct": people,
        "total_pct": total,
        "flow_usd": flow,
        "holders": holders,
        "traders": traders,
    }
    return {"market_holders": holders, "market_traders": traders}


def tech_index(total_pcts: list[float], beta: float) -> list[float]:
    """The derived US tech index (Nasdaq-100) from daily total moves of the US stock index: each day moves by
    beta times the day's percent move (beta is a slope of daily percent changes, so daily compounding)."""
    out, level = [], 100.0
    for p in total_pcts:
        level *= 1.0 + beta * p / 100.0
        out.append(level)
    return out
