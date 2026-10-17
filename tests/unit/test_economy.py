"""Everyday activity (sim/economy.py): late reactions, saving and cashing out, each country's market, moods moved
by life events and conversation, and the switch for older worlds."""

import datetime as dt

import numpy as np
import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.economy import economy_cfg, load_by_country, load_everyday, shift_bands
from jahan.model.enums import EmpStatus, EventKind
from jahan.scenario.inject import InjectError, apply_scenario
from jahan.scenario.spec import EconomyEvent, MacroPath, MacroStep, NewsEvent, Scenario
from jahan.sim import economy
from jahan.sim.params import SimParams
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from jahan.store.snapshot import world_fingerprint

START = "2026-10-13"  # a Tuesday


def _world(countries=("USA",), n=1500, seed=5, economy_on=True, **params):
    cfg = RunConfig(
        n_agents=n,
        countries=list(countries),
        seed=seed,
        start_date=START,
        economy=economy_on,
        params=SimParams(**params),
    )
    return generate_world(cfg)


def _news(w, countries=("USA",), valence=-0.8, days=7, offset=0):
    ev = NewsEvent(
        headline="Economy enters recession as layoffs surge",
        topic="economy",
        valence=valence,
        salience=0.9,
        countries=list(countries),
        duration_days=days,
        start_day_offset=offset,
    )
    apply_scenario(w, Scenario(prompt="recession", events=[ev]), EventLog(), 1)


def _series(rec, metric, country=None):
    t = rec.metrics.table().to_pydict()
    return {
        d: v
        for d, m, c, v in zip(t["day"], t["metric"], t["country"], t["value"], strict=True)
        if m == metric and (country is None or c == country)
    }


def test_new_worlds_start_with_everyday_activity():
    w = _world(("USA", "DEU", "IRN"), n=1200)
    st = w.macro["economy"]
    assert st["since_day"] == 0 and st["prompt_id"] is None and set(st["markets"]) == {"USA", "DEU", "IRN"}
    assert economy.POS in w.pstate
    pos = w.pstate[economy.POS][: w.pop.n]
    holds = ~np.isnan(pos)
    assert holds.any() and np.all(pos[holds] == 0.0)  # no news yet: everybody's target is 0
    assert st["markets"]["IRN"]["cfg"]["weekdays"] == [5, 6, 0, 1, 2]
    old = _world(economy_on=False)
    assert "economy" not in old.macro and not old.pstate


def test_quiet_world_news_part_flat_and_mood_moves():
    w = _world(n=1500)
    rec = run_in_memory(w, 40)
    usa = [c.iso3 for c in w.countries].index("USA")
    assert set(_series(rec, "stocks_news_pct", usa).values()) == {0.0}
    mood = _series(rec, "sentiment", usa)
    assert mood[39] < 0.0  # the unemployed are gloomier, with nothing to cheer anyone up
    assert abs(mood[39]) < 0.02


def test_late_reactions_build_and_fade():
    w = _world(n=2000, market_kappa=0.2, market_savings=False)
    _news(w, days=7)
    rec = run_in_memory(w, 120)
    usa = [c.iso3 for c in w.countries].index("USA")
    news = _series(rec, "stocks_news_pct", usa)
    level = 100 * np.cumprod([1 + news[d] / 100 for d in sorted(news)])
    low = int(np.argmin(level))
    assert level[low] < 99.5 and low >= 3  # the fall builds over several days, not just the first
    assert level[-1] > level[low]  # as the stories stop being tracked, holders buy back as they look
    gap = _series(rec, "stocks_gap", usa)  # the share of the wanted change still to be traded
    peak = max(gap, key=gap.get)
    assert gap[1] < gap[peak] and gap[119] < 0.5 * gap[peak]  # builds during the story, then is traded away


def test_attention_off_reacts_the_day_people_learn():
    slow = _world(n=1500, market_kappa=0.2, market_savings=False)
    fast = _world(n=1500, market_kappa=0.2, market_savings=False, market_attention=False)
    for w in (slow, fast):
        _news(w)
    rs, rf = run_in_memory(slow, 3), run_in_memory(fast, 3)
    usa = [c.iso3 for c in slow.countries].index("USA")
    s, f = _series(rs, "stocks_news_pct", usa), _series(rf, "stocks_news_pct", usa)
    assert sum(f.values()) < sum(s.values()) < 0  # everyone acting at once moves more, faster
    pos = fast.pstate[economy.POS][: fast.pop.n]
    inv = economy._investors(fast, fast.macro["economy"]["markets"]["USA"]["cfg"], fast.config.params)
    from jahan.sim.world_events import news_mood

    trades = inv.trades & inv.holds
    assert np.allclose(pos[trades], np.clip(0.2 * news_mood(fast)[trades], -1, 1))


def test_logins_fall_after_a_down_day():
    w = _world(n=1500, market_savings=False)
    mk = w.macro["economy"]["markets"]["USA"]
    run_in_memory(w, 1)
    base = mk["last"]["attention"]
    mk["last_trading_pct"] = -1.0
    run_in_memory(w, 1)  # Wednesday, after a (pretend) down Tuesday
    assert mk["last"]["attention"] == pytest.approx(0.905 * base, rel=0.02)


def test_switch_for_an_old_world():
    w = _world(n=1200, economy_on=False)
    run_in_memory(w, 3)
    _news(w)
    run_in_memory(w, 2)
    before = world_fingerprint(w)
    cfg = economy_cfg(paths.data_dir(), ["USA"])
    log = EventLog()
    apply_scenario(w, Scenario(prompt="on", events=[EconomyEvent(cfg=cfg)]), log, 2)
    assert world_fingerprint(w) != before and w.macro["economy"]["since_day"] == 5
    t = log.table().to_pydict()
    assert EventKind.ECONOMY_STARTED in t["kind"]
    index = w.macro["us_stocks_index"]
    pos = w.pstate[economy.POS][: w.pop.n]
    from jahan.sim.world_events import news_mood

    held = ~np.isnan(pos)
    kappa = cfg["markets"]["USA"]["kappa"]
    assert np.allclose(pos[held], np.clip(kappa * news_mood(w)[held], -1, 1))  # known news is already traded
    rec = run_in_memory(w, 1)
    usa = [c.iso3 for c in w.countries].index("USA")
    assert abs(_series(rec, "stocks_news_pct", usa)[5]) < 0.05  # no flood of trades on the switch day
    assert w.macro["us_stocks_index"] == pytest.approx(index, abs=0.5)
    with pytest.raises(InjectError, match="already on"):
        apply_scenario(w, Scenario(prompt="again", events=[EconomyEvent(cfg=cfg)]), EventLog(), 3)


def test_unemployed_people_are_gloomier():
    w = _world(n=1500, mood_contagion=0.0)
    run_in_memory(w, 60)
    s = w.pop["sentiment"].astype(float)
    un = w.pop["alive"] & (w.pop["emp_status"] == EmpStatus.UNEMPLOYED)
    em = w.pop["alive"] & (w.pop["emp_status"] == EmpStatus.EMPLOYED)
    assert un.any() and s[un].mean() < -0.02 and abs(s[em].mean()) < 0.01


def test_conversation_spreads_mood_without_collapsing_it():
    w = _world(n=1500, mood_life=False, mood_contagion=0.05)
    _news(w, valence=-0.8)
    run_in_memory(w, 20)
    s = w.pop["sentiment"].astype(float)
    assert np.all((s >= -1) & (s <= 1))
    alone = _world(n=1500, mood_life=False, mood_contagion=0.0)
    _news(alone, valence=-0.8)
    run_in_memory(alone, 20)
    a = alone.pop["sentiment"].astype(float)
    # talk reaches people the news did not: fewer people sit at exactly 0, and moods still differ
    assert (np.abs(s) > 1e-6).sum() > (np.abs(a) > 1e-6).sum()
    assert s.std() > 0.3 * a.std()


def test_layoffs_lever_and_cash_outs():
    base = _world(n=2000)
    hit = _world(n=2000)
    lever = MacroPath(variable="job_loss_rate", steps=[MacroStep(days=1, pct_per_day=300.0)])
    apply_scenario(hit, Scenario(prompt="layoffs quadruple", events=[lever]), EventLog(), 1)
    rb, rh = run_in_memory(base, 60), run_in_memory(hit, 60)
    tb, th = rb.events.table().to_pydict(), rh.events.table().to_pydict()

    def losers(t):
        return {a for a, d in zip(t["agent"], t["detail"], strict=True) if d == "job_loss"}

    first = {a for a, d, day in zip(tb["agent"], tb["detail"], tb["day"], strict=True) if d == "job_loss" and day == 0}
    assert len(losers(th)) > 2 * len(losers(tb))
    assert first <= losers(th)  # the same draws: baseline job losers lose their jobs in the scenario too
    usa = [c.iso3 for c in hit.countries].index("USA")
    sav = _series(rh, "stocks_savings_pct", usa)
    assert sum(sav.values()) < sum(_series(rb, "stocks_savings_pct", usa).values())
    assert hit.macro["labour"]["job_loss_mult"][usa] == pytest.approx(4.0)


def test_each_country_trades_on_its_own_news():
    w = _world(("USA", "DEU"), n=2000, market_kappa=0.3, market_savings=False)
    _news(w, countries=["DEU"], valence=-0.9)
    rec = run_in_memory(w, 20)
    codes = [c.iso3 for c in w.countries]
    deu = _series(rec, "stocks_news_pct", codes.index("DEU"))
    usa = _series(rec, "stocks_news_pct", codes.index("USA"))
    assert min(deu.values()) < 0 and w.macro["economy"]["markets"]["DEU"]["index"] < 100
    assert max(abs(v) for v in usa.values()) < 0.25 * max(abs(v) for v in deu.values())
    labels = w.macro.get("item_labels", {}).values()
    if w.macro["economy"]["markets"]["DEU"]["index"] < 99.0:  # a 1% fall makes German news, in Germany
        assert any(str(x).startswith("German stocks fall") for x in labels)


def test_tehran_trades_saturday_to_wednesday():
    w = _world(("IRN",), n=1500, market_kappa=0.5, market_savings=False)
    _news(w, countries=["IRN"], valence=-0.9, days=14)
    rec = run_in_memory(w, 14)
    irn = [c.iso3 for c in w.countries].index("IRN")
    people = _series(rec, "stocks_people_pct", irn)
    start = dt.date.fromisoformat(START)
    for day, pct in people.items():
        if (start + dt.timedelta(days=day)).weekday() in (3, 4):  # Thursday, Friday
            assert pct == 0.0, f"Tehran traded on day {day}"
    assert min(people.values()) < 0


def test_shadow_mode_moves_nothing():
    w = _world(n=1500, market_kappa=0.3, market_shadow=True)
    _news(w)
    rec = run_in_memory(w, 10)
    usa = [c.iso3 for c in w.countries].index("USA")
    assert min(_series(rec, "stocks_news_pct", usa).values()) < 0
    assert w.macro["us_stocks_index"] == 100.0


def test_bands_shift_to_the_country_share():
    bands = [
        {"pct_from": 0, "pct_to": 50, "own_share": 34},
        {"pct_from": 50, "pct_to": 90, "own_share": 78},
        {"pct_from": 90, "pct_to": 100, "own_share": 95},
    ]
    s = shift_bands(bands, 5.0)
    assert 0.5 * s[0] + 0.4 * s[1] + 0.1 * s[2] == pytest.approx(5.0, abs=1e-6)
    assert s[0] < s[1] < s[2]


def test_data_files_and_registry():
    by = load_by_country(paths.data_dir())
    ev = load_everyday(paths.data_dir())
    assert by is not None and ev is not None
    assert set(by.countries) == {"USA", "DEU", "BRA", "NGA", "IRN", "IND", "CHN", "IDN"}
    assert 85 <= ev.attention.mean_days() <= 95  # published mean 85.2
    cfg = economy_cfg(paths.data_dir(), list(by.countries))
    for iso, mk in cfg["markets"].items():
        if mk["adult_share"] is not None:
            assert 0 < mk["adult_share"] < 30, iso
    assert cfg["markets"]["NGA"]["illustrative"] and not cfg["markets"]["DEU"]["illustrative"]
    problems = validate_data(paths.data_dir(), list(by.countries))
    assert not [p for p in problems if "everyday" in p or "investors_by_country" in p], problems
