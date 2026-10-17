"""`jahan watch`: the list, the history read back from the timeline's metrics, and undo."""

import json

import pytest

from tests.conftest import cli


def _json(out: str):
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    run = tmp_path_factory.mktemp("watch") / "w"
    cli(
        "gen",
        "--run",
        str(run),
        "--agents",
        "400",
        "--countries",
        "USA,DEU,IRN",
        "--seed",
        "4",
        "--start-date",
        "2026-10-13",
        "--no-economy",  # people trade only on news: a quiet world stays exactly flat (see test_economy_markets)
    )
    cli("run", "--run", str(run), "--days", "5", "--no-autonomy")
    return run


def test_watch_list_history_and_parts(world):
    run = str(world)
    cat = _json(cli("watch", "catalog", "--run", run, "--json"))["catalog"]
    ids = {c["id"] for c in cat}
    assert {"market:us_stocks", "market:us_tech_stocks", "market:gold", "market:dollar", "market:fx:IRN"} <= ids
    assert "mood:USA" in ids
    out = cli("watch", "add", "--run", run, "market:us_stocks", "market:dollar", "mood:USA")
    assert "Watching US stocks (S&P 500) from day 5" in out
    cli("run", "--run", run, "--days", "3", "--no-autonomy")
    data = _json(cli("watch", "show", "--run", run, "--json"))
    by = {s["id"]: s for s in data["series"]}
    stocks = by["market:us_stocks"]
    assert [r["day"] for r in stocks["history"]] == list(range(8))
    assert stocks["since_day"] == 5 and stocks["now"] == pytest.approx(100.0)
    assert stocks["components"] == {"people_pct": 0.0, "scripted_pct": 0.0}  # a quiet world: kappa 0 or no news
    assert by["market:dollar"]["components"] is None and by["market:dollar"]["unit"] == "rate"
    assert "not a forecast" in data["disclaimer"]
    status = _json(cli("status", "--run", run, "--json"))
    assert status["markets"]["us_stocks"]["index"] == pytest.approx(100.0)
    assert status["markets"]["people_trading"] is True
    assert status["markets"]["us_tech_stocks"]["derived"] is True
    assert status["watch"] == ["market:dollar", "market:us_stocks", "mood:USA"]
    text = cli("watch", "show", "--run", run)
    assert "US stocks (S&P 500): 100.00, +0.00% since day 5" in text
    assert data["explanation"] is None  # only asked for with --since-day
    ex = _json(cli("watch", "show", "--run", run, "--json", "--since-day", "5"))["explanation"]
    assert ex["quiet"] and (ex["from_day"], ex["to_day"]) == (5, 8)
    said = " ".join(ex["sentences"])
    assert "Nothing was added between day 5 and day 8" in said and "Nobody heard any news" in said
    assert "only when news reaches them" in said and "Moods change only" in said
    assert "Claude is not set up" in said  # tests run offline
    text = cli("watch", "show", "--run", run, "--since-day", "5")
    assert "Nothing you watch moved between day 5 and day 8:" in text


def test_unknown_series_lists_the_catalog(world):
    with pytest.raises(AssertionError, match="unknown series 'market:nikkei'; this world has"):
        cli("watch", "add", "--run", str(world), "market:nikkei")


def test_scripted_move_and_undo(world, tmp_path):
    run = str(world)
    sc = tmp_path / "gold.yaml"
    sc.write_text(
        "prompt: gold up\nevents:\n- type: macro\n  variable: gold\n  steps:\n  - {days: 2, pct_per_day: 1.0}\n",
        encoding="utf-8",
    )
    before = _json(cli("status", "--run", run, "--json"))
    cli("watch", "add", "--run", run, "market:gold", "market:us_tech_stocks")
    cli("inject", "--run", run, "--scenario", str(sc), "--yes")
    cli("run", "--run", run, "--days", "3", "--no-autonomy")
    by = {s["id"]: s for s in _json(cli("watch", "show", "--run", run, "--json"))["series"]}
    assert by["market:gold"]["since_watched_pct"] == pytest.approx(100 * (1.01**2 - 1))
    # the gold headline is bad news to people, who may sell US stocks; tech stocks follow at beta x each day's move
    stocks = {r["day"]: r["value"] for r in _json(cli("watch", "show", "--run", run, "--json"))["series"][0]["history"]}
    tech = by["market:us_tech_stocks"]
    beta = _json(cli("status", "--run", run, "--json"))["markets"]["us_tech_stocks"]["beta"]
    level, prev = 100.0, 100.0
    for d in sorted(stocks):
        level *= 1 + beta * (stocks[d] / prev - 1)
        prev = stocks[d]
    assert tech["now"] == pytest.approx(level)
    prompt = max(p["prompt_id"] for p in _json(cli("status", "--run", run, "--json"))["prompts"])
    cli("reset", "--run", run, "--before-prompt", str(prompt))
    after = {s["id"]: s for s in _json(cli("watch", "show", "--run", run, "--json"))["series"]}
    assert after["market:gold"]["history"][-1]["day"] == before["day"] - 1  # history ends where the world is
    assert after["market:gold"]["now"] == pytest.approx(100.0)
    cli("watch", "remove", "--run", run, "market:gold")
    assert "market:gold" not in _json(cli("watch", "list", "--run", run, "--json"))
