"""Read prediction-market questions and prices from Polymarket's public data API (no key, read only).

Every request goes through ``feeds.fetch_url``: the raw response is recorded under ``runs/<name>/recordings``
(git-ignored) once per real calendar day, and replays never touch the network.  Only what a world needs is
kept from a response: the question, its page, the resolve date and the prices read, with the fetch date.
Check Polymarket's terms of use before relying on this; Jahan does not trade and needs no account.

Endpoints (https://gamma-api.polymarket.com, documented at docs.polymarket.com):
  /public-search?q=...&limit_per_type=N     events matching a text
  /events?active=true&closed=false&order=volume24hr&ascending=false&limit=N[&tag_slug=...]
                                            busiest open events (the ordering parameters are not documented)
  /markets/slug/<slug>                      one market
  /events/slug/<slug>                       one event and its markets
``outcomes``, ``outcomePrices`` and ``clobTokenIds`` arrive as JSON text inside the JSON and are decoded here.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from jahan.ingest.feeds import fetch_url
from jahan.llm.recordings import Recordings

GAMMA = "https://gamma-api.polymarket.com"
SITE = "https://polymarket.com"
PURPOSE = "polymarket"


class PolymarketError(RuntimeError):
    pass


@dataclass
class MarketRow:
    question: str
    slug: str
    event_slug: str
    event_title: str
    url: str
    resolves_by: str | None
    yes_price: float
    best_bid: float | None
    best_ask: float | None
    volume: float | None
    liquidity: float | None
    closed: bool
    description: str
    clob_token_id: str | None
    fetch_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonish(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except json.JSONDecodeError:
            return []
        return v if isinstance(v, list) else []
    return []


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _resolves_by(m: dict[str, Any]) -> str | None:
    """The resolve date in New York time: an end time of 03:59-04:59 UTC belongs to the evening before."""
    raw = m.get("endDate") or m.get("endDateIso")
    if not raw:
        return None
    s = str(raw)
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s[:10]
    if t.tzinfo is None:
        return t.date().isoformat()
    return (t - dt.timedelta(hours=5)).date().isoformat()


def _row(m: dict[str, Any], event: dict[str, Any] | None, fetch_date: str) -> MarketRow | None:
    outcomes = [str(o).strip().lower() for o in _jsonish(m.get("outcomes"))]
    prices = [_num(p) for p in _jsonish(m.get("outcomePrices"))]
    if sorted(outcomes) != ["no", "yes"] or len(prices) != 2:
        return None  # only yes/no questions
    yes = prices[outcomes.index("yes")]
    if yes is None or not 0.0 <= yes <= 1.0:
        return None
    tokens = _jsonish(m.get("clobTokenIds"))
    ev = event or (m.get("events") or [None])[0] or {}
    event_slug = str(ev.get("slug") or "")
    slug = str(m.get("slug") or "")
    return MarketRow(
        question=str(m.get("question") or "").strip(),
        slug=slug,
        event_slug=event_slug,
        event_title=str(ev.get("title") or ""),
        url=f"{SITE}/event/{event_slug}" if event_slug else f"{SITE}/market/{slug}",
        resolves_by=_resolves_by(m),
        yes_price=float(yes),
        best_bid=_num(m.get("bestBid")),
        best_ask=_num(m.get("bestAsk")),
        volume=_num(m.get("volume") if m.get("volume") is not None else m.get("volumeNum")),
        liquidity=_num(m.get("liquidity") if m.get("liquidity") is not None else m.get("liquidityNum")),
        closed=bool(m.get("closed", False)),
        description=str(m.get("description") or "")[:400],
        clob_token_id=str(tokens[outcomes.index("yes")]) if len(tokens) == 2 else None,
        fetch_date=fetch_date,
    )


def parse_markets(body: bytes, fetch_date: str, include_closed: bool = False) -> list[MarketRow]:
    """Yes/no markets from any Gamma response: a search result, a list of events, one event or one market."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PolymarketError(f"not a Polymarket data response: {e}") from e
    if isinstance(data, dict) and "events" in data and "question" not in data:
        events = data["events"] or []
    elif isinstance(data, list):
        events = data
    else:
        events = [data]
    rows: list[MarketRow] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        markets = ev.get("markets") if "markets" in ev else [ev]
        parent = ev if "markets" in ev else None
        for m in markets or []:
            if not isinstance(m, dict):
                continue
            r = _row(m, parent, fetch_date)
            if r is not None and r.question and r.slug and (include_closed or not r.closed):
                rows.append(r)
    return rows


Fetcher = Callable[[str], bytes] | None


def search(query: str, rec: Recordings, fetch_day: str, limit: int = 10, fetcher: Fetcher = None) -> list[MarketRow]:
    q = urllib.parse.urlencode({"q": query, "limit_per_type": limit})
    return parse_markets(fetch_url(f"{GAMMA}/public-search?{q}", rec, fetch_day, PURPOSE, fetcher), fetch_day)


def top_events(
    rec: Recordings, fetch_day: str, tag: str | None = None, limit: int = 20, fetcher: Fetcher = None
) -> list[MarketRow]:
    params: dict[str, Any] = {"limit": limit, "active": "true", "closed": "false", "order": "volume24hr"}
    params["ascending"] = "false"
    if tag:
        params["tag_slug"] = tag
    url = f"{GAMMA}/events?{urllib.parse.urlencode(params)}"
    return parse_markets(fetch_url(url, rec, fetch_day, PURPOSE, fetcher), fetch_day)


def market_by_slug(slug: str, rec: Recordings, fetch_day: str, fetcher: Fetcher = None) -> MarketRow:
    """One market (closed ones included, so a resolution can be read); an event slug lists its markets."""
    safe = urllib.parse.quote(slug, safe="")
    first: Exception | None = None
    try:
        rows = parse_markets(
            fetch_url(f"{GAMMA}/markets/slug/{safe}", rec, fetch_day, PURPOSE, fetcher), fetch_day, True
        )
    except Exception as e:  # noqa: BLE001 - try the event, then explain
        rows, first = [], e
    for r in rows:
        if r.slug == slug:
            return r
    try:
        ev = parse_markets(fetch_url(f"{GAMMA}/events/slug/{safe}", rec, fetch_day, PURPOSE, fetcher), fetch_day)
    except Exception as e:  # noqa: BLE001
        raise PolymarketError(f"no Polymarket market or event {slug!r} ({first or e})") from e
    if not ev:
        raise PolymarketError(f"no open yes/no market for {slug!r}")
    names = "; ".join(f"{r.slug} ({r.question})" for r in ev[:12])
    raise PolymarketError(f"{slug!r} is an event with several markets; watch one of them: {names}")


def refresh(
    slugs: list[str], rec: Recordings, fetch_day: str, fetcher: Fetcher = None
) -> tuple[dict[str, MarketRow], list[str]]:
    """Current prices for these markets (one request each); problems are collected, never fatal."""
    out: dict[str, MarketRow] = {}
    problems: list[str] = []
    for s in slugs:
        try:
            out[s] = market_by_slug(s, rec, fetch_day, fetcher)
        except Exception as e:  # noqa: BLE001 - one failing market must not stop the others
            problems.append(f"{s}: {e}")
    return out, problems
