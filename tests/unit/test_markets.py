"""Market reactions to announcements: published estimates times the surprise, applied by the engine."""

import json

import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import load_registry, validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind
from jahan.model.markets import combined, load_markets, market_moves
from jahan.scenario.inject import apply_scenario, preview_moves
from jahan.scenario.spec import Announcement, MacroPath, MacroStep, PromptScenario, Scenario
from jahan.sim.params import SimParams
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from jahan.store.snapshot import data_hashes
from tests.conftest import cli


@pytest.fixture(scope="module")
def cfg():
    out = load_markets(paths.data_dir())
    assert out is not None
    return out


def _world(countries=("USA", "DEU"), n=1500):
    # people's trading off: these tests pin the published estimates alone (tests/unit/test_market_people.py)
    params = SimParams(market_people=False)
    return generate_world(
        RunConfig(n_agents=n, countries=list(countries), seed=5, start_date="2026-09-15", params=params)
    )


def _by_target(moves):
    return {(m.target, m.term): m for m in moves}


def test_fed_hike_of_16_sep_2026(cfg):
    # +25 bp with a 92% chance priced: expected 23, surprise +2 bp
    moves = _by_target(market_moves(cfg, "us_policy_rate", 25, 23))
    assert moves[("us_stocks", "surprise")].pct == pytest.approx(-3.961 * 2 / 100)
    assert moves[("dollar", "surprise")].pct == pytest.approx(0.19 * 2 / 6)
    gold = moves[("gold", "surprise")]
    assert gold.pct == pytest.approx(-0.31 * 2 / 6) and not gold.active  # shown, never applied
    assert set(combined(list(moves.values()))) == {"us_stocks", "dollar"}


def test_forward_guidance_and_payrolls(cfg):
    g = _by_target(market_moves(cfg, "us_policy_rate", 0, 0, guidance_bp=10))
    assert g[("us_stocks", "guidance")].pct == pytest.approx(-0.14 * 10 / 4.61)
    assert g[("dollar", "guidance")].pct == pytest.approx(0.21 * 10 / 4.61)
    assert g[("us_stocks", "surprise")].pct == 0.0
    p = combined(market_moves(cfg, "us_payrolls", 172, 85))  # 5 Jun 2026: +87k above consensus
    assert p["us_stocks"] == pytest.approx(-0.220 * 87 / 77.5)
    assert p["gold"] == pytest.approx(-0.18 * 87 / 77.5)
    assert p["dollar"] == pytest.approx(0.13 * 87 / 77.5)


def test_every_estimate_cites_the_registry(cfg):
    reg = load_registry(paths.data_dir())
    assert cfg.source_ids() and all(s in reg for s in cfg.source_ids())
    assert not [p for p in validate_data(paths.data_dir(), ["USA", "DEU"]) if "markets" in p]


def test_announcement_moves_markets_and_becomes_news():
    w = _world()
    run_in_memory(w, 1)
    eur0 = w.macro["fx"][1]
    ev = Announcement(indicator="us_policy_rate", actual=25, expected=23, headline="Fed raises rates", valence=-0.3)
    apply_scenario(w, Scenario(prompt="fed", events=[ev]), EventLog(), 1)
    rec = run_in_memory(w, 2)
    assert w.macro["us_stocks_index"] == pytest.approx(100 * (1 - 3.961 * 0.02 / 100))
    assert w.macro["gold_index"] == 100.0  # the disputed gold estimate is inactive
    assert w.macro["fx"][1] / eur0 - 1 == pytest.approx(0.19 * 2 / 6 / 100)
    assert w.macro["announcements"][-1]["indicator"] == "us_policy_rate"
    t = rec.metrics.table().to_pydict()
    aware = [
        v
        for m, it, c, v in zip(t["metric"], t["item"], t["country"], t["value"], strict=True)
        if m == "aware_share" and c == 0
    ]
    assert aware and max(aware) > 0  # the headline spreads in the USA


def test_fully_expected_decision_moves_nothing_and_world_without_euro():
    w = _world(("USA",), n=800)
    ev = Announcement(indicator="us_policy_rate", actual=0, expected=0, headline="Fed holds rates")
    apply_scenario(w, Scenario(prompt="hold", events=[ev]), EventLog(), 1)
    run_in_memory(w, 1)
    assert w.macro["us_stocks_index"] == 100.0 and w.macro["gold_index"] == 100.0
    ev2 = Announcement(indicator="us_payrolls", actual=-23, expected=80, headline="Jobs fall")
    apply_scenario(w, Scenario(prompt="jobs", events=[ev2]), EventLog(), 2)  # no Germany: the euro is skipped
    run_in_memory(w, 1)
    assert w.macro["us_stocks_index"] > 100.0 and w.macro["gold_index"] > 100.0


def test_old_world_gets_the_stock_index_only_when_stocks_move():
    w = _world(("USA",), n=800)
    del w.macro["us_stocks_index"]  # a world made before the index existed
    del w.macro["market"]
    gold = MacroPath(variable="gold", steps=[MacroStep(days=1, pct_per_day=1.0)])
    apply_scenario(w, Scenario(prompt="gold", events=[gold]), EventLog(), 1)
    run_in_memory(w, 1)
    assert "us_stocks_index" not in w.macro  # fingerprints of recorded sessions stay valid
    stocks = MacroPath(variable="us_stocks", steps=[MacroStep(days=1, pct_per_day=-1.2)])
    apply_scenario(w, Scenario(prompt="stocks", events=[stocks]), EventLog(), 2)
    rec = run_in_memory(w, 1)
    assert w.macro["us_stocks_index"] == pytest.approx(98.8)
    t = rec.events.table().to_pydict()
    created = [d for k, d in zip(t["kind"], t["detail"], strict=True) if k == EventKind.ITEM_CREATED]
    assert any(d.startswith("US stocks fall 1.2%") for d in created)


def test_small_stock_moves_make_no_headline():
    w = _world(("USA",), n=800)
    stocks = MacroPath(variable="us_stocks", steps=[MacroStep(days=1, pct_per_day=-0.5)])
    apply_scenario(w, Scenario(prompt="stocks", events=[stocks]), EventLog(), 1)
    rec = run_in_memory(w, 1)
    kinds = rec.events.table().column("kind").to_pylist()
    assert EventKind.ITEM_CREATED not in kinds


def test_schema_hashes_and_preview(tmp_path):
    assert "announcement" in json.dumps(PromptScenario.model_json_schema())
    hashes = data_hashes(RunConfig(countries=["USA"]), paths.data_dir())
    assert "global/markets.yaml" in hashes
    w = _world(("USA",), n=800)
    ev = Announcement(indicator="us_payrolls", actual=172, expected=85, headline="Jobs beat")
    rows = preview_moves(w, Scenario(prompt="jobs", events=[ev]))
    assert rows[0]["surprise"] == 87 and {m["target"] for m in rows[0]["moves"]} == {"us_stocks", "gold", "dollar"}


def test_cli_preview_shows_market_moves(tmp_path):
    run = tmp_path / "w"
    cli(
        "gen",
        "--run",
        str(run),
        "--agents",
        "300",
        "--countries",
        "USA,DEU",
        "--seed",
        "2",
        "--start-date",
        "2026-10-13",
    )
    sc = tmp_path / "fed.yaml"
    sc.write_text(
        "prompt: fed\nevents:\n- type: announcement\n  indicator: us_policy_rate\n  actual: 25\n  expected: 23\n"
        "  headline: Fed raises rates\n",
        encoding="utf-8",
    )
    out = cli("inject", "--run", str(run), "--scenario", str(sc), "--preview", "--json")
    data = json.loads(out.strip().splitlines()[-1])
    stocks = [m for m in data["market_moves"][0]["moves"] if m["target"] == "us_stocks"][0]
    assert stocks["pct"] == pytest.approx(-0.07922)
    text = cli("inject", "--run", str(run), "--scenario", str(sc), "--yes")
    assert "published estimates, not the simulated people" in text
