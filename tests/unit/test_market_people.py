"""People's trading in US stocks (sim/market.py): who trades, how mood moves the index, and old worlds."""

import datetime as dt

import numpy as np
import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.investors import investor_masks, load_investors, ownership_check
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Announcement, MacroPath, MacroStep, NewsEvent, Scenario
from jahan.sim.market import tech_index
from jahan.sim.params import SimParams
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from jahan.store.snapshot import data_hashes

START = "2026-10-13"  # a Tuesday


def _world(countries=("USA",), n=1200, seed=5, **params):
    cfg = RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date=START, params=SimParams(**params))
    return generate_world(cfg)


def _bad_news(w, valence=-0.8):
    ev = NewsEvent(
        headline="US economy enters recession as layoffs surge",
        topic="economy",
        valence=valence,
        salience=0.9,
        countries=["USA"],
        duration_days=7,
    )
    apply_scenario(w, Scenario(prompt="recession", events=[ev]), EventLog(), 1)


def _series(rec, metric):
    t = rec.metrics.table().to_pydict()
    return {d: v for d, m, c, v in zip(t["day"], t["metric"], t["country"], t["value"], strict=True) if m == metric}


def test_new_worlds_carry_their_own_copy_of_the_config():
    w = _world()
    cfg = w.macro["market"]["cfg"]
    data = load_investors(paths.data_dir())
    assert data is not None and cfg == data.to_world()
    assert "global/investors.yaml" not in data_hashes(RunConfig(countries=["USA"]), paths.data_dir())


def test_who_holds_and_who_trades():
    w = _world(("USA", "DEU"), n=2000)
    inv = investor_masks(w, w.macro["market"]["cfg"])
    pop = w.pop
    usa = [c.iso3 for c in w.countries].index("USA")
    assert inv.holds.any()
    assert not (inv.holds & (pop["country"] != usa)).any()  # no sourced share outside the USA
    assert not (inv.holds & pop["is_notable"]).any()
    assert not (inv.holds & (w.ages() < 18)).any()
    assert not (inv.trades & ~pop["internet"]).any() and not (inv.trades & ~inv.holds).any()
    held = np.unique(inv.holdings[inv.holds])
    assert set(held) <= {12600.0, 53200.0, 608000.0} and len(held) == 3


def test_ownership_matches_the_survey_and_the_draw_is_stable():
    w = _world()
    check = ownership_check(w, w.macro["market"]["cfg"])
    assert 0.50 <= check["adults"] <= 0.66  # SCF bands imply 57.7%; Gallup reports 61-62%
    again = _world()
    a = investor_masks(w, w.macro["market"]["cfg"]).holds
    assert np.array_equal(a, investor_masks(again, again.macro["market"]["cfg"]).holds)
    before = investor_masks(w, w.macro["market"]["cfg"])
    run_in_memory(w, 10)
    after = investor_masks(w, w.macro["market"]["cfg"])
    n = before.holds.shape[0]
    same_band = (before.band == after.band[:n]) & w.pop["alive"][:n] & before.adults & after.adults[:n]
    assert np.array_equal(before.holds[same_band], after.holds[:n][same_band])


def test_quiet_world_stays_flat():
    w = _world(market_kappa=1.0)
    rec = run_in_memory(w, 20)
    assert w.macro["us_stocks_index"] == 100.0 and w.macro["market"]["people_index"] == 100.0
    t = rec.events.table().to_pydict()
    assert "us_stocks_people" not in t["detail"]


def test_bad_news_makes_holders_sell_on_weekdays_only():
    w = _world(market_kappa=0.2)
    _bad_news(w)
    rec = run_in_memory(w, 12)
    people = _series(rec, "us_stocks_people_pct")
    assert min(people.values()) < 0 and w.macro["us_stocks_index"] < 100.0
    assert w.macro["market"]["scripted_index"] == 100.0  # no announcement or scripted path
    start = dt.date.fromisoformat(START)
    for day, pct in people.items():
        if (start + dt.timedelta(days=day)).weekday() >= 5:
            assert pct == 0.0, f"people traded on a weekend (day {day})"
    m = w.macro["market"]
    assert w.macro["us_stocks_index"] == pytest.approx(m["scripted_index"] * m["people_index"] / 100.0)
    t = rec.events.table().to_pydict()
    assert "us_stocks_people" in t["detail"]


def test_weekend_news_is_traded_on_monday():
    w = _world(market_kappa=0.2)
    run_in_memory(w, 4)  # Tuesday to Friday
    assert w.dow() == 5  # Saturday
    _bad_news(w)
    rec = run_in_memory(w, 3)  # Saturday, Sunday, Monday
    people = _series(rec, "us_stocks_people_pct")
    assert people[4] == 0.0 and people[5] == 0.0  # the market is closed
    assert people[6] < 0.0  # Monday trades the weekend's news
    assert w.macro["market"]["pending_pct"] == 0.0


def test_one_shock_stays_bounded_and_reverts():
    w = _world(market_kappa=0.2)
    shock = MacroPath(variable="us_stocks", steps=[MacroStep(days=1, pct_per_day=-3.0)])
    apply_scenario(w, Scenario(prompt="crash", events=[shock]), EventLog(), 1)
    rec = run_in_memory(w, 70)
    people = _series(rec, "us_stocks_people_pct")
    index = _series(rec, "us_stocks_people_index")
    assert min(index.values()) > 50.0  # bounded: no spiral
    assert min(people.values()) < 0 < max(people.values())  # selling while mood falls, buying as it recovers
    late = [abs(v) for d, v in people.items() if d >= 60]
    assert max(late) < 0.5 * max(abs(v) for v in people.values())


def test_scripted_moves_unchanged_when_people_do_not_trade():
    def run(**params):
        w = _world(("USA", "DEU"), n=1500, **params)
        run_in_memory(w, 1)
        ev = Announcement(indicator="us_policy_rate", actual=25, expected=23, headline="Fed raises rates", valence=-0.3)
        apply_scenario(w, Scenario(prompt="fed", events=[ev]), EventLog(), 1)
        run_in_memory(w, 3)
        return w

    off = run(market_people=False, market_kappa=1.0)
    assert off.macro["us_stocks_index"] == pytest.approx(100 * (1 - 3.961 * 0.02 / 100))
    assert off.macro["market"]["people_index"] == 100.0
    on = run(market_kappa=1.0)
    assert on.macro["market"]["scripted_index"] == off.macro["us_stocks_index"]


def test_the_daily_move_is_capped():
    w = _world(market_kappa=1.0)
    gloom = MacroPath(variable="sentiment", steps=[MacroStep(days=1, pct_per_day=-100.0)])
    apply_scenario(w, Scenario(prompt="gloom", events=[gloom]), EventLog(), 1)
    run_in_memory(w, 1)
    assert w.macro["market"]["last"]["people_pct"] == -10.0


def test_old_worlds_and_worlds_without_the_usa():
    w = _world(market_kappa=1.0)
    del w.macro["market"]  # a world made before people traded
    _bad_news(w)
    rec = run_in_memory(w, 8)
    assert "market" not in w.macro and w.macro["us_stocks_index"] == 100.0
    assert not _series(rec, "us_stocks_people_pct")
    deu = _world(("DEU",), n=600, market_kappa=1.0)
    rec = run_in_memory(deu, 3)
    assert set(_series(rec, "us_stocks_people_pct").values()) == {0.0}


def test_tech_index_is_beta_times_daily_moves():
    assert tech_index([1.0, -2.0], 1.2) == pytest.approx([101.2, 101.2 * (1 - 0.024)])


def test_investors_data_cites_the_registry():
    problems = validate_data(paths.data_dir(), ["USA"])
    assert not [p for p in problems if "investors" in p]
    strict = validate_data(paths.data_dir(), ["USA"], strict=True)
    assert any("investors.kappa_fit" in p for p in strict)  # the fitted number is flagged on purpose
