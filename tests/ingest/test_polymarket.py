"""Polymarket's public data API: parsing (no network), recording and replay, and `watch` with --replay."""

import json

import pytest

from jahan.ingest.feeds import fetch_url
from jahan.ingest.polymarket import GAMMA, PURPOSE, PolymarketError, market_by_slug, parse_markets, refresh
from jahan.llm.recordings import Recordings
from tests.conftest import cli

DAY = "2026-10-15"
SLUG = "us-x-iran-ceasefire-continues-through-october-31-20260917"


def _market(slug=SLUG, outcomes=("Yes", "No"), prices=("0.795", "0.205"), closed=False, **kw):
    return {
        "question": kw.get("question", "US x Iran ceasefire continues through October 31?"),
        "slug": slug,
        "endDate": kw.get("endDate", "2026-10-31T23:59:00Z"),
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps(list(prices)),
        "clobTokenIds": json.dumps(["111", "222"]),
        "volume": "2982743.43",
        "liquidity": "486077.56",
        "bestBid": 0.79,
        "bestAsk": 0.8,
        "closed": closed,
        "description": "Resolves No if the US takes qualifying military action against Iran.",
        "events": [
            {"slug": "us-iran-ceasefire-continues-throughptptpt", "title": "US-Iran ceasefire continues through...?"}
        ],
    }


def _bytes(obj) -> bytes:
    return json.dumps(obj).encode()


def test_parse_double_encoded_markets_and_events():
    rows = parse_markets(_bytes(_market()), DAY)
    assert len(rows) == 1
    r = rows[0]
    assert r.yes_price == 0.795 and r.clob_token_id == "111" and r.resolves_by == "2026-10-31"
    assert r.url == "https://polymarket.com/event/us-iran-ceasefire-continues-throughptptpt"
    flipped = parse_markets(_bytes(_market(outcomes=("No", "Yes"), prices=("0.3", "0.7"))), DAY)[0]
    assert flipped.yes_price == 0.7 and flipped.clob_token_id == "222"
    event = {
        "slug": "ev",
        "title": "Event",
        "markets": [
            _market("a"),
            _market("b", closed=True),
            _market("c", outcomes=("Trump", "Harris", "Other"), prices=("0.5", "0.4", "0.1")),
            _market("d", endDate="2026-11-01T03:59:00Z"),
        ],
    }
    rows = parse_markets(_bytes({"events": [event], "tags": [], "pagination": {}}), DAY)
    assert [r.slug for r in rows] == ["a", "d"]  # closed and three-way markets are left out
    assert rows[1].resolves_by == "2026-10-31" and rows[0].event_slug == "ev"
    with pytest.raises(PolymarketError):
        parse_markets(b"<html>blocked</html>", DAY)


def test_recorded_then_replayed_without_the_network(tmp_path):
    rec = Recordings(tmp_path / "rec")
    seen = []

    def fetcher(url):
        seen.append(url)
        if "/markets/slug/" in url:
            return _bytes(_market())
        raise OSError("404")

    row = market_by_slug(SLUG, rec, DAY, fetcher)
    assert row.yes_price == 0.795 and seen == [f"{GAMMA}/markets/slug/{SLUG}"]
    assert DAY not in seen[0]  # the URL never carries a date

    def offline(url):
        raise AssertionError("replay must not fetch")

    rec.replay_only = True
    assert market_by_slug(SLUG, rec, DAY, offline).yes_price == 0.795
    rows, problems = refresh([SLUG, "missing-market"], rec, DAY, offline)
    assert set(rows) == {SLUG} and problems and "missing-market" in problems[0]


def test_an_event_slug_lists_its_markets(tmp_path):
    rec = Recordings(tmp_path / "rec")
    event = {"slug": "ev", "title": "Event", "markets": [_market("m1"), _market("m2")]}

    def fetcher(url):
        if "/markets/slug/" in url:
            raise OSError("404")
        return _bytes(event)

    with pytest.raises(PolymarketError, match="watch one of them: m1"):
        market_by_slug("ev", rec, DAY, fetcher)


def test_watch_with_replayed_polymarket_answers(tmp_path):
    run = tmp_path / "w"
    cli("gen", "--run", str(run), "--agents", "800", "--countries", "USA,IRN", "--seed", "3", "--start-date", DAY)
    with pytest.raises(AssertionError, match="needs the network"):
        cli("watch", "add", "--run", str(run), f"odds:{SLUG}")
    rec = Recordings(run / "recordings")
    fetch_url(f"{GAMMA}/markets/slug/{SLUG}", rec, DAY, PURPOSE, lambda url: _bytes(_market()))
    search = f"{GAMMA}/public-search?q=Iran&limit_per_type=15"
    fetch_url(
        search, rec, DAY, PURPOSE, lambda url: _bytes({"events": [{"slug": "e", "title": "E", "markets": [_market()]}]})
    )
    cat = json.loads(
        cli(
            "watch",
            "catalog",
            "--run",
            str(run),
            "--polymarket",
            "--query",
            "Iran",
            "--replay",
            "--fetch-day",
            DAY,
            "--json",
        )
        .strip()
        .splitlines()[-1]
    )
    assert cat["polymarket"][0]["series"] == f"odds:{SLUG}"
    out = cli("watch", "add", "--run", str(run), f"odds:{SLUG}", "--replay", "--fetch-day", DAY)
    assert "market 79.5% on 2026-10-15" in out
    fetch_url(
        f"{GAMMA}/markets/slug/{SLUG}",
        rec,
        "2026-10-16",
        PURPOSE,
        lambda url: _bytes(_market(prices=("1", "0"), closed=True)),
    )
    out = cli("watch", "fetch", "--run", str(run), "--replay", "--fetch-day", "2026-10-16")
    assert "real 100.0% on 2026-10-16" in out and "resolved YES on Polymarket" in out
    real = json.loads((run / "odds" / "real.json").read_text(encoding="utf-8"))
    assert [r["fetch_date"] for r in real["questions"][SLUG]] == ["2026-10-15", "2026-10-16"]
