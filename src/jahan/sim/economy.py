"""Tick phase 5b with everyday activity: each country's stock market moved by its own residents.

A world has everyday activity when ``world.macro["economy"]`` exists: new worlds get it when they are generated,
older worlds through a recorded, undoable switch (`jahan economy on`, an ``EconomyEvent``).  Worlds without it
keep the older engine (sim/market.py) and are byte-identical to before.

Each market (one per country with investor data, ``macro["economy"]["markets"][ISO3]``) has two people-driven
parts:

**News reactions, priced when the news arrives.**  Each holder wants a position of kappa x their view of the
news (their news mood plus any scripted mood shock), as a share of their holdings.  The market prices that
demand when a story first reaches the market's holders: on the day people there first react to it, by the change
it makes to the views of those who know it then; when the story stops being tracked, exactly that amount is
reversed.  People who hear it later, and holders who act late, trade demand the market has already priced, so
their trades do not move the index (as with steady saving, predictable flows are priced when they become
predictable).  Why: in the 2023-2025 backtest, 87% of the simulated reaction fell on the four days after a
release while the real S&P 500 had already reacted on the day (data/backtests/benchmarks.yaml, rule v2 result).

Each holder still acts late on their own account: every day they close the share ``a`` of the gap between
their position and their target, where ``a`` is their lifetime login rate (Sicherman, Loewenstein, Seppi and
Utkus 2016: days logged in out of 730), times 0.905 after a market fall.  Positions are kept per person
(``world.pstate["mkt_pos"]``) and shown as the share of the wanted change still to be traded; they do not move
prices.

**Saving and cashing out (US only).**  Paycheck contributions and retiree drawdowns are steady and already in the
price; what moves it is how far today's cash-outs differ from what is normal for today's population: each
simulated job loser in a retirement plan cashes out (Vanguard: a third of people leaving a job take cash), and
hardship withdrawals happen as daily draws.  The expected amount is recomputed every day from who is employed,
so the part has no trend; its day-to-day steps are larger in small worlds (each simulated person stands for more
real people), and they never count toward the automatic "stocks fall 1%" headlines.

The move is 100 x multiplier x net flow / holdings of all holders (Gabaix and Koijen 2021 for the US; borrowed
elsewhere), traded on the market's trading days (weekend flow carries over), capped per day with the excess
carried.

**Version 2** (new worlds; older ones upgrade with `jahan economy on`): headlines about one event from one prompt
share a story key and are priced once (the largest of them), and the US market gets a third part, other
investors (funds, pensions, insurers, foreign investors: not simulated one by one), see ``others_news`` below.
Version 1 worlds keep everything above unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EmpStatus, EventKind
from jahan.model.investors import Investors, investor_masks
from jahan.model.world import World
from jahan.rng import Stream, normal, u01
from jahan.sim.params import SimParams
from jahan.sim.world_events import news_mood, stocks_headline
from jahan.store.eventlog import EventLog

VERSION = 2
POS = "mkt_pos"
DAYS_PER_YEAR = 365.25
LOG_MIN_PCT = 0.01  # moves smaller than this are not logged as events (metrics keep them)


def enabled(world: World) -> bool:
    return "economy" in world.macro


def state(world: World) -> dict[str, Any]:
    return world.macro["economy"]


def version(world: World) -> int:
    eco = world.macro.get("economy")
    return 0 if eco is None else int(eco.get("version", 1))


def _country(world: World, iso3: str) -> int:
    codes = [c.iso3 for c in world.countries]
    return codes.index(iso3) if iso3 in codes else -1


# ------------------------------------------------------------------------------------------------ the switch


def switch_on(world: World, cfg: dict[str, Any], log: EventLog, prompt_id: int | None) -> None:
    """Turn on everyday activity from today (``cfg`` is the world's own copy, built by `jahan economy on` or at
    generation).  Known news counts as already traded: everybody starts at their target position."""
    m = world.macro
    m.setdefault("us_stocks_index", 100.0)
    markets = {iso: mk for iso, mk in cfg["markets"].items() if _country(world, iso) >= 0}
    if "USA" in markets and "market" not in m:  # a world made before people traded: every move so far was scripted
        m["market"] = {
            "cfg": markets["USA"],
            "people_index": 100.0,
            "scripted_index": float(m["us_stocks_index"]),
            "pending_pct": 0.0,
            "last": {"day": -1, "scripted_pct": 0.0, "people_pct": 0.0, "total_pct": 0.0, "flow_usd": 0.0},
        }
    carried = float(m["market"].get("pending_pct", 0.0)) if "market" in m else 0.0
    if "market" in m:
        m["market"]["pending_pct"] = 0.0
    m["economy"] = {
        "version": int(cfg.get("version", 1)),
        "since_day": world.day,
        "prompt_id": prompt_id,
        "cfg": cfg["everyday"],
        "markets": {
            iso: {
                "cfg": mk,
                "index": 100.0,  # the US level stays in macro["us_stocks_index"]
                "people_index": 100.0,
                "news_index": 100.0,
                "savings_index": 100.0,
                "pending": {"news": carried if iso == "USA" else 0.0, "savings": 0.0},
                # stories known before the switch stay as they are: nothing priced, nothing reversed
                "priced": {str(int(i)): 0.0 for i in world.beliefs.slot_item if i >= 0},
                "pending_cash": 0.0,
                "last_trading_pct": 0.0,
                "last": {},
            }
            for iso, mk in markets.items()
        },
    }
    if m["economy"]["version"] >= 2:
        _start_v2(world, cfg)
    pos = world.pstate.create(POS, "f8", float("nan"), world.pop.capacity)
    view = news_mood(world) + _shift(world)
    for mk in m["economy"]["markets"].values():
        inv = _investors(world, mk["cfg"], None)
        pos[: world.pop.n][inv.holds] = _target(view, _kappa(world, mk, world.config.params))[inv.holds]
    log.add(
        world.day,
        EventKind.ECONOMY_STARTED,
        -1,
        -1,
        -1,
        float(m["economy"]["version"]),
        -1,
        ",".join(sorted(markets)),
    )


def _start_v2(world: World, cfg: dict[str, Any]) -> None:
    """Version 2's own state: the story clusters and, with a US market, the other investors' part.  Stories
    known before today stay as they are (no cluster, no other-investor reaction)."""
    eco = world.macro["economy"]
    eco["stories"] = {}
    if "USA" in eco["markets"] and cfg.get("others") is not None:
        eco["others"] = {
            "cfg": dict(cfg["others"]),
            "priced": {},  # story cluster -> its move, kept for the cluster's life (never given back)
            "news_index": 100.0,
            "quiet_index": 100.0,
            "tech_index": 100.0,  # the tech line's own part (tech stories), on top of beta x the S&P move
            "pending": {"news": 0.0, "tech": 0.0},
            "quiet": {"h": None},
            "last": {},
        }


def upgrade(world: World, cfg: dict[str, Any], log: EventLog, prompt_id: int | None) -> None:
    """Move a version 1 world to version 2 from today (recorded as a prompt, undone by resetting before it).
    Markets keep their levels and everything priced so far."""
    eco = world.macro["economy"]
    eco["version"] = int(cfg["version"])
    eco["upgraded"] = {"day": world.day, "prompt_id": prompt_id}
    _start_v2(world, cfg)
    log.add(world.day, EventKind.ECONOMY_STARTED, -1, -1, -1, float(eco["version"]), -1, "upgrade")


# ------------------------------------------------------------------------------------------------- helpers


def _shift(world: World) -> NDArray[np.float64]:
    """Each person's scripted mood shock so far (macro sentiment steps in their home country)."""
    shift = np.asarray(world.macro.get("sentiment_shift", []), dtype=np.float64)
    if shift.size == 0:
        return np.zeros(world.pop.n)
    return shift[world.pop["country"].astype(np.int64)]


def _kappa(world: World, mk: dict[str, Any], params: SimParams) -> float:
    return float(mk["cfg"]["kappa"]) if params.market_kappa is None else float(params.market_kappa)


def _target(view: NDArray[np.float64], kappa: float) -> NDArray[np.float64]:
    return np.clip(kappa * view, -1.0, 1.0)


def _investors(world: World, mcfg: dict[str, Any], params: SimParams | None) -> Investors:
    need = None if params is None else params.market_requires_internet
    return investor_masks(world, mcfg, need)


def login_rates(world: World, att: dict[str, Any]) -> NDArray[np.float64]:
    """Each person's lifetime share of days they look at their account (a draw from the published spread)."""
    u = u01(world.sim_seed, Stream.ATTENTION, 0, world.pop["id"])
    days = np.interp(100.0 * u, np.asarray(att["pct"], np.float64), np.asarray(att["days"], np.float64))
    return np.asarray(days / float(att["of_days"]), dtype=np.float64)


def trading_day(world: World, mcfg: dict[str, Any]) -> bool:
    return world.dow() in set(int(d) for d in mcfg.get("weekdays", [0, 1, 2, 3, 4]))


# ------------------------------------------------------------------------------------------ saving (US only)


def _wage_quartile(world: World, employed: NDArray[np.bool_]) -> NDArray[np.int8]:
    """Quartile (0-3) of own monthly income among the employed (population-weighted), -1 for others."""
    pop = world.pop
    out = np.full(pop.n, -1, np.int8)
    rows = np.flatnonzero(employed)
    if rows.size == 0:
        return out
    inc = pop["income_month"][rows].astype(np.float64)
    w = pop["pop_weight"][rows].astype(np.float64)
    order = np.lexsort((pop["id"][rows], inc))
    cum = np.cumsum(w[order])
    pct = np.empty(rows.size)
    pct[order] = (cum - 0.5 * w[order]) / cum[-1]
    out[rows] = np.minimum((pct * 4).astype(np.int8), 3)
    return out


def participants(world: World, c: int, sav: dict[str, Any]) -> NDArray[np.bool_]:
    """Employed residents of country ``c`` who save in a workplace retirement plan (lifetime draw against the
    participation rate of their wage quartile)."""
    pop = world.pop
    employed = pop["alive"] & (pop["country"] == c) & (pop["emp_status"] == EmpStatus.EMPLOYED)
    q = _wage_quartile(world, employed)
    rate = np.asarray(sav["participation"], np.float64) / 100.0
    u = u01(world.sim_seed, Stream.SAVE, 0, pop["id"], k=0)
    return np.asarray(employed & (u < rate[np.maximum(q, 0)]), dtype=bool)


def _cash_per_person(world: World, inv: Investors, sav: dict[str, Any]) -> NDArray[np.float64]:
    """Stock sold when a holder in a plan leaves a job: their plan's stock times the share of people with a
    balance that size who take cash (weighted by the people the agent stands for)."""
    w = world.pop["pop_weight"].astype(np.float64)
    plan_stock = float(sav["dc_share"]) * inv.holdings
    balance = plan_stock / float(sav["equity_share"])
    k = np.searchsorted(np.asarray(sav["cashout_thresholds"], np.float64), balance, side="right")
    share = np.asarray(sav["cashout_shares"], np.float64)[k]
    return np.asarray(w * share * plan_stock, dtype=np.float64)


def _daily_job_loss(world: World, c: int) -> float:
    v = world.countries[c].vital
    return 0.0 if v is None else 1.0 - (1.0 - float(v.job_loss_per_month)) ** (12.0 / DAYS_PER_YEAR)


def note_job_losses(world: World, c: int, losers: NDArray[np.int64]) -> None:
    """Called by the labour phase before people lose their jobs: their cash-outs reach the market on the next
    day's market phase (phase 7 runs after it)."""
    if not enabled(world) or losers.size == 0 or not world.config.params.market_savings:
        return
    iso = world.countries[c].iso3
    mk = state(world)["markets"].get(iso)
    sav = state(world)["cfg"].get("saving")
    if mk is None or sav is None or iso not in sav["countries"]:
        return
    inv = _investors(world, mk["cfg"], world.config.params)
    part = participants(world, c, sav)
    who = np.zeros(world.pop.n, bool)
    who[losers] = True
    who &= part & inv.holds
    mk["pending_cash"] = float(mk.get("pending_cash", 0.0)) + float(_cash_per_person(world, inv, sav)[who].sum())


def saving_flow(world: World, c: int, mk: dict[str, Any], inv: Investors, held: float) -> dict[str, float]:
    """Today's cash-outs and hardship withdrawals against what is normal for today's population (US dollars;
    negative = net selling), and the steady flows for display."""
    sav = state(world)["cfg"].get("saving")
    iso = world.countries[c].iso3
    zero = {"net": 0.0, "realised": 0.0, "expected": 0.0, "contributions": 0.0, "drawdowns": 0.0}
    if sav is None or iso not in sav["countries"] or held <= 0:
        return zero
    pop = world.pop
    w = pop["pop_weight"].astype(np.float64)
    part = participants(world, c, sav) & inv.holds
    cash = _cash_per_person(world, inv, sav)
    expected_cash = _daily_job_loss(world, c) * float(cash[part].sum())
    realised_cash = float(mk.get("pending_cash", 0.0))
    mk["pending_cash"] = 0.0
    rate = float(sav["hardship_rate"]) / DAYS_PER_YEAR
    balance = float(sav["dc_share"]) * inv.holdings / float(sav["equity_share"])
    amount = np.minimum(float(sav["hardship_amount"]), balance) * float(sav["equity_share"])  # the stock sold
    hard = part & (u01(world.sim_seed, Stream.SAVE, world.day, pop["id"], k=1) < rate)
    realised_hard = float((w * amount)[hard].sum())
    expected_hard = rate * float((w * amount)[part].sum())
    pay = pop["income_month"].astype(np.float64) * 12.0 / DAYS_PER_YEAR
    fx = float(world.macro["fx"][c]) if "fx" in world.macro else 1.0
    contributions = float(
        (w * pay / fx * float(sav["contribution"]) * float(sav["equity_share"]))[participants(world, c, sav)].sum()
    )
    retired = inv.holds & (pop["emp_status"] == EmpStatus.RETIRED)
    drawdowns = float((w * inv.holdings)[retired].sum()) * float(sav.get("drawdown_rate", 0.0)) / DAYS_PER_YEAR
    realised = realised_cash + realised_hard
    expected = expected_cash + expected_hard
    return {
        "net": -(realised - expected),
        "realised": realised,
        "expected": expected,
        "contributions": contributions,
        "drawdowns": drawdowns,
    }


# --------------------------------------------------------------------------------------------- the market


@dataclass
class DayOpen:
    index: float  # US stock level at the start of the tick (announcements and scripted moves come after)


def open_day(world: World) -> DayOpen:
    return DayOpen(float(world.macro.get("us_stocks_index", 100.0)))


def _apply(level: float, news: float, savings: float) -> float:
    return level * (1.0 + news / 100.0) * (1.0 + savings / 100.0)


def _news_sums(world: World) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]]:
    """Each person's news sum before squashing (news mood = tanh of it) and each live item's part of it."""
    bs = world.beliefs
    n = world.pop.n
    total = np.zeros(n)
    parts: dict[int, NDArray[np.float64]] = {}
    for s in np.flatnonzero(bs.slot_item >= 0):
        item = int(bs.slot_item[s])
        wt = float(world.items["sentiment_weight"][item])
        if wt == 0.0:
            continue
        c = (bs.m["belief"][:n, s].astype(np.float64) - 0.5) * 2.0 * bs.m["aware"][:n, s].astype(np.float64) * wt
        parts[item] = c
        total += c
    return total, parts


def _cluster(stories: dict[str, Any], item: int) -> str:
    """The key a story is priced under: its cluster (version 2: headlines of one event from one prompt), or the
    item itself (stories known before version 2, and every story in a version 1 world)."""
    e = stories.get(str(item))
    return str(item) if e is None else f"c:{e['story']}"


def priced_news_flow(
    world: World, mk: dict[str, Any], inv: Investors, kappa: float, c: int, sums: tuple[NDArray[np.float64], dict]
) -> float:
    """Today's demand the market prices in this market: stories its holders first react to today (the change in
    their views), minus what was priced for stories no longer tracked, plus scripted mood shocks in the country."""
    w = world.pop["pop_weight"].astype(np.float64)
    total, parts = sums
    priced: dict[str, float] = mk.setdefault("priced", {})
    stories = state(world).get("stories", {})
    who = inv.trades & inv.holds
    base = w * inv.holdings * kappa
    flow = 0.0
    groups: dict[str, list[int]] = {}  # clusters not priced yet -> their stories
    for item in parts:
        key = _cluster(stories, item)
        if key not in priced:
            groups.setdefault(key, []).append(item)
    for key, items in groups.items():
        if len(items) == 1:
            part = parts[items[0]]
            knows = who & (part != 0.0)
            if not knows.any():
                continue
            dv = np.tanh(total[knows]) - np.tanh(total[knows] - part[knows])  # the view this story adds
            priced[key] = float((base[knows] * dv).sum())
        else:  # one event told in several headlines: priced as its strongest-rated headline alone (its own
            # reach, the event's other headlines left out), so extra headlines add neither reach nor size
            rating = world.items["valence"].astype(np.float64) * world.items["sentiment_weight"].astype(np.float64)
            top = max(items, key=lambda i: abs(float(rating[i])))  # ties: the first headline
            knows = who & (parts[top] != 0.0)
            if not knows.any():
                continue
            rest = total - np.sum([parts[i] for i in items], axis=0)
            dv = np.tanh(rest[knows] + parts[top][knows]) - np.tanh(rest[knows])
            priced[key] = float((base[knows] * dv).sum())
        flow += priced[key]
    live = {_cluster(stories, int(i)) for i in world.beliefs.slot_item if i >= 0}
    for key in [k for k in priced if k not in live]:  # no longer tracked
        flow -= priced.pop(key)
    shift = list(world.macro.get("sentiment_shift", []))
    seen = float(mk.get("shift_seen", shift[c] if c < len(shift) else 0.0))
    if c < len(shift) and shift[c] != seen:
        flow += float(base[who].sum()) * (float(shift[c]) - seen)
        mk["shift_seen"] = float(shift[c])
    elif c < len(shift):
        mk["shift_seen"] = float(shift[c])
    return flow


# ------------------------------------------------------------------- other investors (version 2, US only)


def others_news(world: World, eco: dict[str, Any], oth: dict[str, Any], params: SimParams) -> tuple[float, float]:
    """Other investors' reaction to news (data/global/investors.yaml other_investors.rule_v4): each US story with
    a market rating moves the S&P 500 on its first trading day after release (at or after the close: the next
    day), once per story cluster (its strongest member), permanently.  impact = s x market_valence x salience on
    the part the story is about: a broad story moves the S&P 500 by impact (the tech line follows by its beta);
    a tech story moves the tech line by impact and the S&P 500 by information technology's index weight x
    impact.  Returns today's percent move of the S&P 500 and the tech line's own extra percent move."""
    cfg = oth["cfg"]
    s = float(cfg["s"]) if params.market_others_s is None else float(params.market_others_s)
    stories = eco.get("stories", {})
    priced: dict[str, Any] = oth["priced"]
    it = world.items
    close = int(cfg.get("close_utc_hour", 21))
    w_it = cfg.get("it_weight")
    beta = float(cfg.get("tech_beta") or 0.0)
    best: dict[str, tuple[float, float, float]] = {}  # cluster -> (S&P part, tech extra, impact)
    for k, e in stories.items():
        if e.get("mv") is None or not e.get("us") or e["story"] in priced:
            continue
        i = int(k)
        if int(it["slot"][i]) < 0:
            continue
        if int(it["created_day"][i]) == world.day and int(it["release_utc_hour"][i]) >= close:
            continue  # after the close: traded tomorrow
        impact = s * float(e["mv"]) * float(it["salience"][i])
        if e.get("sector") == "tech" and w_it:
            part = (float(w_it) * impact, impact * (1.0 - beta * float(w_it)), impact)
        else:
            part = (impact, 0.0, impact)
        if e["story"] not in best or abs(impact) > abs(best[e["story"]][2]):
            best[e["story"]] = part
    sp = tech = 0.0
    for key, (a, b, _) in best.items():
        priced[key] = {"pct": a, "tech": b}
        sp += a
        tech += b
    live = {e["story"] for k, e in stories.items() if int(it["slot"][int(k)]) >= 0}
    for key in [k for k in priced if k not in live]:  # kept for good: nothing to give back, only forgotten
        del priced[key]
    return sp, tech


def quiet_shocks(world: World, oth: dict[str, Any]) -> dict[str, float]:
    """Today's ordinary-day moves in percent (other_investors.quiet_days): the S&P 500 from a GARCH(1,1)
    variance with Student-t shocks; gold and the euro's dollar rate with their own volatility scaled by the same
    variance ratio and shocks correlated with the S&P 500's (one shared t scale, so tails come together).  No
    drift.  Updates the variance state."""
    q = oth["cfg"]["quiet"]
    g = q["sp"]
    h = oth["quiet"]["h"]
    h = float(g["var"]) if h is None else float(h)
    nu = int(q["nu"])
    keys = np.arange(3 + nu)
    z = normal(world.sim_seed, Stream.QUIET, world.day, keys)
    e = np.asarray(q["chol"], dtype=np.float64) @ z[:3]  # correlated: S&P 500, gold, euro
    chi = float(np.sum(z[3:] ** 2))
    t = e * np.sqrt((nu - 2) / max(chi, 1e-12))  # unit-variance Student-t, one shared scale
    ratio = np.sqrt(h / float(g["var"]))
    sp = float(np.sqrt(h) * t[0])
    oth["quiet"]["h"] = float(g["omega"]) + float(g["alpha"]) * sp * sp + float(g["beta"]) * h
    return {"sp": sp, "gold": float(q["gold_sd"] * ratio * t[1]), "eur": float(q["eur_sd"] * ratio * t[2])}


def _apply_quiet_macro(world: World, shocks: dict[str, float]) -> None:
    """Gold and the euro's dollar rate (euros per dollar move opposite to dollars per euro).  Their headline
    references move along, so ordinary days never count toward a "gold rises" or currency headline."""
    m = world.macro
    refs = m.get("news_ref", {})
    if "gold_index" in m:
        m["gold_index"] = float(m["gold_index"]) * (1.0 + shocks["gold"] / 100.0)
        if "gold" in refs:
            refs["gold"] = float(refs["gold"]) * (1.0 + shocks["gold"] / 100.0)
    fx = m.get("fx")
    if fx:
        for c, spec in enumerate(world.countries):
            if spec.currency == "EUR":
                fx[c] = float(fx[c]) / (1.0 + shocks["eur"] / 100.0)
                if f"fx:{c}" in refs:
                    refs[f"fx:{c}"] = float(refs[f"fx:{c}"]) / (1.0 + shocks["eur"] / 100.0)


def close_day(
    world: World, params: SimParams, log: EventLog, opened: DayOpen, news: NDArray[np.float64]
) -> dict[str, int]:
    """Move every market by its residents' trading (and record the US index's scripted part)."""
    st = state(world)
    everyday = st["cfg"]
    world.pstate.ensure_rows(world.pop.n)
    pos = world.pstate[POS]
    view = news[: world.pop.n] + _shift(world)
    rates = login_rates(world, everyday["attention"])
    sums = _news_sums(world)
    w = world.pop["pop_weight"].astype(np.float64)
    m = world.macro
    stats: dict[str, int] = {}
    if "USA" in st["markets"] and "market" in m:  # the scripted part: announcements and what-ifs, phase 0
        scripted = 100.0 * (float(m["us_stocks_index"]) / opened.index - 1.0) if opened.index > 0 else 0.0
        m["market"]["scripted_index"] = float(m["market"]["scripted_index"]) * (1.0 + scripted / 100.0)
    else:
        scripted = 0.0
    oth_us = st.get("others") if "USA" in st["markets"] and "market" in m else None
    shocks = None
    if oth_us is not None and params.market_quiet and oth_us["cfg"].get("quiet") and world.dow() < 5:
        shocks = quiet_shocks(world, oth_us)
    for iso, mk in st["markets"].items():
        c = _country(world, iso)
        if c < 0:
            continue
        oth = oth_us if iso == "USA" else None
        mcfg = mk["cfg"]
        inv = _investors(world, mcfg, params)
        p = pos[: world.pop.n]
        target = _target(view, _kappa(world, mk, params))
        in_c = world.pop["country"] == c
        p[in_c & ~inv.holds] = np.nan  # no longer holding (too young, emigrated, died, income band changed)
        new = inv.holds & np.isnan(p)
        p[new] = target[new]  # a new holder starts where they want to be: no trade
        if params.market_attention:
            a = np.minimum(1.0, rates * float(mcfg.get("attention_scale", 1.0)))
            if world.dow() >= 5:
                a = a * float(everyday["attention"].get("weekend_factor", 1.0))
            if float(mk.get("last_trading_pct", 0.0)) < 0:
                a = a * float(everyday["attention"]["down_day_factor"])
        else:
            a = np.ones(world.pop.n)
        a = np.where(inv.trades, a, 0.0)
        cont = inv.holds & ~new
        step = np.zeros(world.pop.n)
        step[cont] = a[cont] * (target[cont] - p[cont])
        p[cont] += step[cont]
        held = float((w * inv.holdings)[inv.holds].sum())
        mult = float(mcfg["multiplier"]) if params.market_multiplier is None else float(params.market_multiplier)
        flow = float((w * inv.holdings * step)[cont].sum())  # what people actually traded (not priced again)
        priced = priced_news_flow(world, mk, inv, _kappa(world, mk, params), c, sums)
        news_raw = 100.0 * mult * priced / held if held > 0 else 0.0
        sav = saving_flow(world, c, mk, inv, held) if params.market_savings else saving_flow_off()
        savings_raw = 100.0 * mult * sav["net"] / held if held > 0 else 0.0
        if not params.market_people:
            news_raw = savings_raw = 0.0
        pend = mk["pending"]
        pend["news"] = float(pend["news"]) + news_raw
        pend["savings"] = float(pend["savings"]) + savings_raw
        news_pct = savings_pct = others_pct = tech_pct = quiet_pct = 0.0
        if oth is not None:  # other investors: priced today, traded on the next trading day if today is not one
            o_raw, t_raw = others_news(world, st, oth, params) if params.market_others else (0.0, 0.0)
            oth["pending"]["news"] = float(oth["pending"]["news"]) + o_raw
            oth["pending"]["tech"] = float(oth["pending"]["tech"]) + t_raw
        if trading_day(world, mcfg):
            cap = float(mcfg["max_daily_pct"])
            total = pend["news"] + pend["savings"]
            if oth is not None:
                total += oth["pending"]["news"]
            if abs(total) > cap and not params.market_shadow:  # the excess waits, split in proportion
                keep = cap / abs(total)
                news_pct, savings_pct = pend["news"] * keep, pend["savings"] * keep
                pend["news"], pend["savings"] = pend["news"] - news_pct, pend["savings"] - savings_pct
                if oth is not None:
                    op = oth["pending"]
                    others_pct, tech_pct = op["news"] * keep, op["tech"] * keep
                    op["news"], op["tech"] = op["news"] - others_pct, op["tech"] - tech_pct
            else:
                news_pct, savings_pct = pend["news"], pend["savings"]
                pend["news"] = pend["savings"] = 0.0
                if oth is not None:
                    others_pct, tech_pct = oth["pending"]["news"], oth["pending"]["tech"]
                    oth["pending"]["news"] = oth["pending"]["tech"] = 0.0
            if oth is not None and shocks is not None:
                quiet_pct = shocks["sp"]
        people_pct = 100.0 * ((1 + news_pct / 100.0) * (1 + savings_pct / 100.0) - 1.0)
        mk["people_index"] = float(mk["people_index"]) * (1.0 + people_pct / 100.0)
        mk["news_index"] = float(mk["news_index"]) * (1.0 + news_pct / 100.0)
        mk["savings_index"] = float(mk["savings_index"]) * (1.0 + savings_pct / 100.0)
        if iso == "USA" and "market" in m:
            before = float(m["us_stocks_index"])
            key, label, where = "us_stocks", None, None
        else:
            before = float(mk["index"])
            key, label, where = f"stocks:{iso}", str(mcfg.get("news_label", f"{iso} stocks")), [c]
        after = _apply(before, news_pct, savings_pct)
        if oth is not None:
            after = after * (1.0 + others_pct / 100.0) * (1.0 + quiet_pct / 100.0)
        if params.market_shadow:  # the kappa fit: flows are recorded, nothing moves (no feedback)
            after = before
        elif people_pct != 0.0 or others_pct != 0.0 or quiet_pct != 0.0:
            refs = m.get("news_ref", {})
            if quiet_pct != 0.0:  # ordinary-day moves never make news: start the reference here if none exists
                refs = m.setdefault("news_ref", {})
                refs.setdefault(key, before)
            if key in refs:  # saving steps and ordinary-day moves never count toward the automatic market headline
                refs[key] = float(refs[key]) * (1.0 + savings_pct / 100.0)
                if quiet_pct != 0.0:
                    refs[key] = float(refs[key]) * (1.0 + quiet_pct / 100.0)
            if iso == "USA" and "market" in m:
                m["us_stocks_index"] = after
                m["market"]["people_index"] = float(m["market"]["people_index"]) * (1.0 + people_pct / 100.0)
            else:
                mk["index"] = after
            if abs(people_pct) >= LOG_MIN_PCT:
                log.add(world.day, EventKind.MACRO_UPDATE, -1, c, -1, after, c, f"stocks_people:{iso}")
            if abs(others_pct) >= LOG_MIN_PCT:
                log.add(world.day, EventKind.MACRO_UPDATE, -1, c, -1, after, c, f"stocks_others:{iso}")
            stocks_headline(world, before, after, log, where, key=key, label=label)
        us = iso == "USA" and "market" in m
        total_pct = 100.0 * ((1 + (scripted if us else 0.0) / 100.0) * (1 + people_pct / 100.0) - 1.0)
        if oth is not None:
            total_pct = 100.0 * ((1 + total_pct / 100.0) * (1 + others_pct / 100.0) * (1 + quiet_pct / 100.0) - 1.0)
            if not params.market_shadow:
                oth["news_index"] = float(oth["news_index"]) * (1.0 + others_pct / 100.0)
                oth["quiet_index"] = float(oth["quiet_index"]) * (1.0 + quiet_pct / 100.0)
                oth["tech_index"] = float(oth["tech_index"]) * (1.0 + tech_pct / 100.0)
            oth["last"] = {"day": world.day, "others_pct": others_pct, "tech_pct": tech_pct, "quiet_pct": quiet_pct}
        if trading_day(world, mcfg):  # logins fall after a down day (read tomorrow); in shadow mode only the
            # index's own move counts, so who looks never depends on kappa
            mk["last_trading_pct"] = (scripted if us else 0.0) if params.market_shadow else total_pct
        gap = float((w * inv.holdings * np.abs(target - p))[inv.holds].sum() / held) if held > 0 else 0.0
        mk["last"] = {
            "day": world.day,
            "news_pct": news_pct,
            "savings_pct": savings_pct,
            "people_pct": people_pct,
            "total_pct": total_pct,
            "flow": flow,
            "priced": priced,
            "saving": sav,
            "holders": int(inv.holds.sum()),
            "few_holders": int(inv.holds.sum()) < float(everyday.get("min_holders", 0)),
            "traders": int(inv.trades.sum()),
            "attention": float(np.average(a[inv.holds], weights=w[inv.holds])) if inv.holds.any() else 0.0,
            "gap": gap,
        }
        if iso == "USA" and "market" in m:
            m["market"]["last"] = {
                "day": world.day,
                "scripted_pct": scripted,
                "people_pct": people_pct,
                "news_pct": news_pct,
                "savings_pct": savings_pct,
                "total_pct": 100.0 * (float(m["us_stocks_index"]) / opened.index - 1.0) if opened.index > 0 else 0.0,
                "flow_usd": flow,
                "holders": int(inv.holds.sum()),
                "traders": int(inv.trades.sum()),
            }
            stats["market_holders"] = int(inv.holds.sum())
            stats["market_traders"] = int(inv.trades.sum())
    if oth_us is not None:
        last = oth_us.setdefault("last", {})
        last["gold_pct"] = last["dollar_pct"] = 0.0
        if shocks is not None and not params.market_shadow:
            _apply_quiet_macro(world, shocks)
            last["gold_pct"] = shocks["gold"]
            last["dollar_pct"] = 100.0 * (1.0 / (1.0 + shocks["eur"] / 100.0) - 1.0)  # euros per dollar
    if "stories" in st:  # forget stories nobody tracks any more (their clusters were reversed or kept above)
        slot = world.items["slot"]
        for k in [k for k in st["stories"] if int(slot[int(k)]) < 0]:
            del st["stories"][k]
    return stats


def saving_flow_off() -> dict[str, float]:
    return {"net": 0.0, "realised": 0.0, "expected": 0.0, "contributions": 0.0, "drawdowns": 0.0}
