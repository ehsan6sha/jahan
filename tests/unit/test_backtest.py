"""Backtests against real 2026 announcements (`jahan backtest`)."""

import json

import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import load_registry
from jahan.model.markets import load_markets
from jahan.scenario.backtest import (
    BacktestFile,
    EventResult,
    _combine,
    choose_kappa,
    default_events_file,
    default_events_files,
    hindsight,
    score,
    simulate,
)
from tests.conftest import cli


@pytest.fixture(scope="module")
def events():
    return BacktestFile.load_all(default_events_files())


def test_event_files_merge_in_date_order_and_must_agree(tmp_path):
    files = default_events_files()
    assert len(files) >= 2 and default_events_file() in files
    merged = BacktestFile.load_all(files)
    dates = [e.date for e in merged.events]
    assert dates == sorted(dates)
    fed = [e for e in merged.events if e.id.startswith("fomc-")]
    assert len(fed) >= 30  # every Fed decision 2023-2026
    other = BacktestFile.load(files[0]).model_copy(
        update={"noise_sd": merged.noise_sd.model_copy(update={"gold": 9.9})}
    )
    path = tmp_path / "events_x.yaml"
    import yaml

    path.write_text(yaml.safe_dump(other.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="noise_sd differs"):
        BacktestFile.load_all([files[0], path])


def _event(file, eid):
    return next(e for e in file.events if e.id == eid)


def test_events_file_is_complete(events):
    reg = load_registry(paths.data_dir())
    ids = {e.id for e in events.events}
    assert len(ids) == len(events.events)
    for e in events.events:
        assert e.sources, e.id
        assert any(v is not None for v in e.actual.model_dump().values()), e.id
        assert e.actual.us_stocks is not None or "closed" in e.note, e.id  # e.g. Good Friday
        if e.kind != "not_modelled":
            assert e.announcement is not None and e.expected_note, e.id
    for chk in events.society:
        assert chk.event in ids and chk.source in reg and chk.low < chk.high


def test_hindsight_uses_the_two_year_move_beyond_the_decision(events):
    cfg = load_markets(paths.data_dir())
    sep = _event(events, "fomc-2026-09-16")
    after, side = hindsight(sep, cfg)
    guidance = 7 - 0.4636 * 2  # the 2-year moved 7 bp; the 2 bp surprise explains about 1 bp of it
    assert after["us_stocks"] == pytest.approx(-3.961 * 2 / 100 - 0.14 * guidance / 4.61)
    assert after["dollar"] == pytest.approx(0.19 * 2 / 6 + 0.21 * guidance / 4.61)
    assert side["gold"] == pytest.approx(-0.31 * 2 / 6)  # the inactive estimate, shown for comparison


def test_score_rules():
    assert score(-0.08, -0.45, 0.92)["major"] is False
    assert score(0.0, 2.0, 0.92)["major"] is True  # more than 2 s.d.
    assert score(0.33, -1.52, 0.92)["major"] is True  # 2.01 s.d.
    assert score(0.3, -1.0, 0.92)["major"] is True  # wrong direction on a move over 1 s.d.
    assert score(0.1, -1.0, 0.92)["major"] is False  # a forecast under 0.25 s.d. has no meaningful sign


def test_simulate_one_event(events):
    base = RunConfig(n_agents=1200, countries=["USA", "DEU"], seed=11)
    res = simulate(_event(events, "fomc-2026-09-16"), base)
    assert res.surprise == 2
    assert res.jahan["us_stocks"] == pytest.approx(-0.07922, abs=1e-4)
    assert res.jahan["dollar"] == pytest.approx(0.0633, abs=1e-4)
    assert res.jahan["gold"] == 0.0
    shares = res.aware[0]
    assert len(shares) == 3 and shares[0] <= shares[1] <= shares[2] and shares[2] > 0
    assert set(res.people) == {"us_stocks"} and set(res.hybrid) == set(res.jahan)


def test_people_column_is_linear_in_kappa_on_the_release_day(events):
    base = RunConfig(n_agents=2000, countries=["USA", "DEU"], seed=11)
    ev = _event(events, "jobs-2026-06-05")  # a big upside surprise: holders buy on good news
    one = simulate(ev, base, kappa=1.0)
    half = simulate(ev, base, kappa=0.5)
    assert one.jahan == half.jahan  # the estimates part never depends on people
    assert one.people["us_stocks"] > 0
    assert half.people["us_stocks"] == pytest.approx(0.5 * one.people["us_stocks"], rel=1e-9)
    assert one.hybrid["us_stocks"] == pytest.approx(_combine(one.jahan["us_stocks"], one.people["us_stocks"]))


def test_choose_kappa_minimises_the_hybrid_miss(events):
    ev = _event(events, "jobs-2026-06-05")

    def row(scripted, people, real):
        r = EventResult(ev.model_copy(update={"actual": ev.actual.model_copy(update={"us_stocks": real})}), 1.0)
        r.jahan, r.people = {"us_stocks": scripted}, {"us_stocks": people}
        return r

    k, losses = choose_kappa([row(0.0, 1.0, 0.3), row(0.0, -1.0, -0.3)], 0.92)
    assert k == pytest.approx(0.3) and len(losses) == 2001
    k0, _ = choose_kappa([row(0.0, 1.0, -0.5)], 0.92)  # people point the wrong way: kappa 0
    assert k0 == 0.0


def test_cli_backtest_json_and_markdown(tmp_path, events):
    small = events.model_copy(
        update={
            "events": [_event(events, "fomc-2026-09-16"), _event(events, "hormuz-2026-09-28")],
        }
    )
    path = tmp_path / "events.yaml"
    import yaml

    path.write_text(yaml.safe_dump(small.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    out = cli(
        "backtest", "--events", str(path), "--agents", "400", "--seeds", "2", "--json", "--out", str(tmp_path / "b.md")
    )
    data = json.loads(out.strip().splitlines()[-1])
    rows = {r["id"]: r for r in data["events"]}
    assert rows["fomc-2026-09-16"]["scores"]["us_stocks"]["major"] is False
    assert len(rows["fomc-2026-09-16"]["aware"]) == 2  # two seeds for the news-reach check
    assert rows["hormuz-2026-09-28"]["jahan"] == {}
    assert set(rows["fomc-2026-09-16"]["hybrid"]) == {"us_stocks", "gold", "dollar"}
    assert "summary_hybrid" in data and "kappa" in data
    md = (tmp_path / "b.md").read_text(encoding="utf-8")
    assert "| 2026-09-16 |" in md and "Not modelled" in md and "News reach" in md
    assert "US stocks with people's trading" in md


def test_gen_before_the_roster_date_needs_the_flag(tmp_path):
    with pytest.raises(AssertionError, match="is after the world start date"):
        cli("gen", "--run", str(tmp_path / "a"), "--agents", "300", "--start-date", "2026-09-15")
    out = cli(
        "gen", "--run", str(tmp_path / "b"), "--agents", "300", "--start-date", "2026-09-15", "--allow-later-roster"
    )
    assert "Warning: notables/USA" in out and "World day 0 = 2026-09-15" in out
