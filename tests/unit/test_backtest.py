"""Backtests against real 2026 announcements (`jahan backtest`)."""

import json

import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import load_registry
from jahan.model.markets import load_markets
from jahan.scenario.backtest import BacktestFile, default_events_file, hindsight, score, simulate
from tests.conftest import cli


@pytest.fixture(scope="module")
def events():
    return BacktestFile.load(default_events_file())


def _event(file, eid):
    return next(e for e in file.events if e.id == eid)


def test_events_file_is_complete(events):
    reg = load_registry(paths.data_dir())
    ids = {e.id for e in events.events}
    assert len(ids) == len(events.events)
    for e in events.events:
        assert e.sources, e.id
        assert e.actual.us_stocks is not None, e.id
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
    md = (tmp_path / "b.md").read_text(encoding="utf-8")
    assert "| 2026-09-16 |" in md and "Not modelled" in md and "News reach" in md


def test_gen_before_the_roster_date_needs_the_flag(tmp_path):
    with pytest.raises(AssertionError, match="is after the world start date"):
        cli("gen", "--run", str(tmp_path / "a"), "--agents", "300", "--start-date", "2026-09-15")
    out = cli(
        "gen", "--run", str(tmp_path / "b"), "--agents", "300", "--start-date", "2026-09-15", "--allow-later-roster"
    )
    assert "Warning: notables/USA" in out and "World day 0 = 2026-09-15" in out
