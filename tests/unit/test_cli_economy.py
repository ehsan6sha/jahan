"""`jahan economy on`, local markets on the watch list, and `jahan watch set` through the command line."""

import json

from tests.conftest import cli


def _json(out: str):
    return json.loads(out.strip().splitlines()[-1])


def test_switch_an_old_world_on_and_undo_it(tmp_path):
    run = str(tmp_path / "old")
    cli(
        "gen",
        "--run",
        run,
        "--agents",
        "500",
        "--countries",
        "USA,DEU",
        "--seed",
        "3",
        "--start-date",
        "2026-10-13",
        "--no-economy",
    )
    cli("run", "--run", run, "--days", "3", "--no-autonomy")
    st = _json(cli("status", "--run", run, "--json"))
    assert st["markets"]["everyday"] is None
    assert "Everyday activity: off" in cli("status", "--run", run)
    out = cli("economy", "on", "--run", run, "--yes")
    assert "Everyday activity is on from day 3 (prompt 1)" in out and "German stocks (DAX)" in out
    assert "already on" in cli("economy", "on", "--run", run, "--yes")
    cli("run", "--run", run, "--days", "4", "--no-autonomy")
    st = _json(cli("status", "--run", run, "--json"))
    assert st["markets"]["everyday"]["since_day"] == 3 and set(st["markets"]["everyday"]["markets"]) == {"USA", "DEU"}
    cat = {c["id"]: c for c in _json(cli("watch", "catalog", "--run", run, "--json"))["catalog"]}
    assert cat["market:stocks:DEU"]["components"] == ["news"]
    assert cat["market:us_stocks"]["components"] == ["news", "savings", "scripted"]
    cli("watch", "add", "--run", run, "market:stocks:USA", "market:stocks:DEU")  # the alias means US stocks
    watched = _json(cli("watch", "list", "--run", run, "--json"))
    assert set(watched) == {"market:us_stocks", "market:stocks:DEU"}
    cli("run", "--run", run, "--days", "2", "--no-autonomy")
    show = {s["id"]: s for s in _json(cli("watch", "show", "--run", run, "--json"))["series"]}
    assert set(show["market:us_stocks"]["components"]) == {"news_pct", "savings_pct", "scripted_pct"}
    assert [p["name"] for p in show["market:stocks:DEU"]["parts"]] == ["news"]
    cli("reset", "--run", run, "--before-prompt", "1")
    st = _json(cli("status", "--run", run, "--json"))
    assert st["day"] == 3 and st["markets"]["everyday"] is None


def test_watch_set_changes_a_questions_time_effect(tmp_path):
    run = str(tmp_path / "q")
    cli("gen", "--run", run, "--agents", "600", "--countries", "USA,IRN", "--seed", "3", "--start-date", "2026-10-15")
    seed = "us-x-iran-ceasefire-continues-through-december-31-20260918"
    out = cli("watch", "add", "--run", run, f"odds:seed:{seed}")
    assert "holds through a date" in out and "(seed:" in out
    custom = cli(
        "watch",
        "add",
        "--run",
        run,
        "odds:custom:deal-by-december",
        "--question",
        "Will the US and Iran sign a deal by December 31?",
        "--resolves-by",
        "2026-12-31",
        "--prior",
        "0.3",
    )
    assert "happens by a date" in custom and 'wording: "by"' in custom
    cli("watch", "set", "--run", run, "odds:deal-by-december", "--kind", "other")
    rows = {r["id"]: r for r in _json(cli("watch", "show", "--run", run, "--json"))["series"] if r["question"]}
    q = rows["odds:deal-by-december"]["question"]
    assert q["kind"] == "other" and q["kind_by"] == "user"
    cli("run", "--run", run, "--days", "5", "--no-autonomy")
    rows = {r["id"]: r for r in _json(cli("watch", "show", "--run", run, "--json"))["series"]}
    held = rows[f"odds:{seed}"]
    assert held["components"]["time_pts"] > 0  # quiet days make "holds through" more likely
    assert {p["name"] for p in held["parts"]} == {"time", "news", "talk", "other"}
