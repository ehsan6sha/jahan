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


def test_news_is_priced_when_it_arrives_and_reversed_when_it_fades():
    w = _world(n=2000, market_kappa=0.05, market_savings=False)
    _news(w, days=7)
    rec = run_in_memory(w, 40)
    usa = [c.iso3 for c in w.countries].index("USA")
    news = _series(rec, "stocks_news_pct", usa)
    moved = {d: v for d, v in news.items() if abs(v) > 1e-9}
    assert min(moved) <= 1 and news[min(moved)] < 0  # priced when people first react, not weeks later
    assert any(v > 0 for v in moved.values())  # and given back when the story stops being tracked
    gap = _series(rec, "stocks_gap", usa)  # holders still act late on their own accounts
    assert gap[2] > 0 and gap[2] > gap[39]


def test_late_account_checks_do_not_move_prices():
    slow = _world(n=1500, market_kappa=0.05, market_savings=False)
    fast = _world(n=1500, market_kappa=0.05, market_savings=False, market_attention=False)
    for w in (slow, fast):
        _news(w)
    rs, rf = run_in_memory(slow, 10), run_in_memory(fast, 10)
    usa = [c.iso3 for c in slow.countries].index("USA")
    assert _series(rs, "stocks_news_pct", usa) == pytest.approx(_series(rf, "stocks_news_pct", usa))
    assert _series(rs, "stocks_gap", usa)[3] > _series(rf, "stocks_gap", usa)[3]  # positions differ


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


def _headlines(w, n, story=None, mv=None, pid=1):
    evs = [
        NewsEvent(
            headline=f"Lawmakers call for a moratorium on new data centers ({i})",
            topic="economy",
            valence=-0.8,
            market_valence=mv,
            salience=0.9,
            countries=["USA"],
            duration_days=7,
            story=story,
        )
        for i in range(n)
    ]
    apply_scenario(w, Scenario(prompt="moratorium", events=evs), EventLog(), pid)


def test_headlines_of_one_event_are_priced_once():
    """Version 2: three headlines sharing a story key move the market about as much as one; without the key
    they add up (as every story did before)."""
    moves = []
    for n, story in ((1, "m"), (3, "m"), (3, None)):
        w = _world(n=1500, market_kappa=0.05, market_savings=False)
        assert w.macro["economy"]["version"] == 2
        _headlines(w, n, story)
        rec = run_in_memory(w, 3)
        usa = [c.iso3 for c in w.countries].index("USA")
        moves.append(min(_series(rec, "stocks_news_pct", usa).values()))
    one, clustered, split = moves
    assert one < 0 and clustered == pytest.approx(one, rel=0.35)
    assert split < 1.5 * one  # more negative: three separate stories
    stories = w.macro["economy"]["stories"]
    assert len({e["story"] for e in stories.values()}) == 3 and all(e["us"] for e in stories.values())


def test_market_fields_are_stored_only_when_given():
    ev = NewsEvent(headline="h", valence=0.1)
    assert set(ev.dump()) == set(ev.model_dump()) - {"market_valence", "sectors", "story"}
    full = NewsEvent(headline="h", valence=0.1, market_valence=-0.4, sectors="tech", story="k").dump()
    assert (full["market_valence"], full["sectors"], full["story"]) == (-0.4, "tech", "k")


def test_version_1_world_upgrades_in_place():
    w = _world(n=1200, economy_on=False)
    cfg = economy_cfg(paths.data_dir(), ["USA"])
    apply_scenario(w, Scenario(prompt="on", events=[EconomyEvent(version=1, cfg=cfg)]), EventLog(), 1)
    _news(w)
    run_in_memory(w, 2)
    priced = dict(w.macro["economy"]["markets"]["USA"]["priced"])
    assert "stories" not in w.macro["economy"] and priced  # version 1 prices every story on its own
    log = EventLog()
    apply_scenario(w, Scenario(prompt="up", events=[EconomyEvent(version=2, cfg=cfg)]), log, 2)
    eco = w.macro["economy"]
    assert eco["version"] == 2 and eco["upgraded"]["day"] == 2 and eco["stories"] == {} and "others" in eco
    assert eco["markets"]["USA"]["priced"] == priced  # what was priced stays
    _headlines(w, 2, story="m", pid=3)
    run_in_memory(w, 2)
    assert "c:3:m" in eco["markets"]["USA"]["priced"]
    with pytest.raises(InjectError, match="already on"):
        apply_scenario(w, Scenario(prompt="again", events=[EconomyEvent(version=2, cfg=cfg)]), EventLog(), 3)


def _story(w, mv, sector="broad", hour=12, salience=0.8, story=None, pid=1, headline="Congress passes a tax bill"):
    ev = NewsEvent(
        headline=headline,
        topic="economy",
        valence=0.0,
        market_valence=mv,
        sectors=sector,
        salience=salience,
        countries=["USA"],
        duration_days=3,
        release_utc_hour=hour,
        story=story,
    )
    apply_scenario(w, Scenario(prompt="story", events=[ev]), EventLog(), pid)


def _others_world(**params):
    return _world(n=1200, market_kappa=0.0, market_savings=False, market_quiet=False, market_others_s=2.0, **params)


def test_other_investors_react_at_once_and_keep_it():
    w = _others_world()
    _story(w, -0.5)  # Tuesday noon UTC: before the New York close
    rec = run_in_memory(w, 30)
    others = _series(rec, "us_stocks_others_pct")
    assert others[0] == pytest.approx(2.0 * -0.5 * 0.8)
    assert all(v == 0.0 for d, v in others.items() if d > 0)  # once, and never given back
    assert w.macro["us_stocks_index"] == pytest.approx(100.0 * (1 - 0.008))
    assert w.macro["economy"]["others"]["priced"] == {}  # forgotten once the story is no longer tracked


def test_tech_story_moves_the_tech_line_in_full():
    w = _others_world()
    oc = w.macro["economy"]["others"]["cfg"]
    _story(w, -1.0, sector="tech", salience=1.0)
    rec = run_in_memory(w, 1)
    impact = -2.0
    assert _series(rec, "us_stocks_others_pct")[0] == pytest.approx(oc["it_weight"] * impact)
    extra = _series(rec, "us_stocks_tech_pct")[0]
    assert oc["tech_beta"] * oc["it_weight"] * impact + extra == pytest.approx(impact)


def test_other_investors_timing_clusters_and_what_they_ignore():
    late = _others_world()
    _story(late, -0.5, hour=22)  # after the close: traded the next day
    rec = run_in_memory(late, 2)
    assert _series(rec, "us_stocks_others_pct") == pytest.approx({0: 0.0, 1: -0.8})
    pair = _others_world()
    _story(pair, -0.25, story="bill", headline="Congress passes a tax bill")
    _story(pair, -0.5, story="bill", headline="The tax bill raises corporate taxes")
    rec = run_in_memory(pair, 1)
    assert _series(rec, "us_stocks_others_pct")[0] == pytest.approx(-0.8)  # the strongest headline only
    plain = _others_world()
    _story(plain, None)  # no market rating: other investors do not react
    from jahan.scenario.spec import Announcement

    ann = Announcement(indicator="us_payrolls", actual=165, expected=65, headline="Jobs beat", valence=0.5)
    apply_scenario(plain, Scenario(prompt="jobs", events=[ann]), EventLog(), 2)
    rec = run_in_memory(plain, 2)
    assert set(_series(rec, "us_stocks_others_pct").values()) == {0.0}


QUIET = {  # made-up coefficients for the test (the real ones come from scripts/quiet_days_fit.py)
    "sp": {"omega": 0.02, "alpha": 0.1, "beta": 0.88, "var": 1.0},
    "nu": 5,
    "chol": [[1.0, 0.0, 0.0], [0.1, 0.99, 0.0], [-0.2, 0.0, 0.98]],
    "gold_sd": 1.2,
    "eur_sd": 0.45,
}


def test_ordinary_days_move_like_markets_do():
    def run(**params):
        w = _world(n=800, market_kappa=0.0, market_savings=False, countries=("USA", "DEU"), **params)
        w.macro["economy"]["others"]["cfg"]["quiet"] = QUIET
        return w, run_in_memory(w, 60)

    w, rec = run()
    quiet = _series(rec, "us_stocks_quiet_pct")
    weekdays = {d for d in quiet if w.date(d).weekday() < 5}
    assert all(quiet[d] == 0.0 for d in quiet if d not in weekdays)
    moves = np.array([quiet[d] for d in sorted(weekdays)])
    assert 0.3 < np.mean(np.abs(moves)) < 2.0 and abs(np.mean(moves)) < 0.5
    assert w.macro["gold_index"] != 100.0 and w.macro["fx"][1] != w.macro["fx0"][1]
    labels = w.macro.get("item_labels", {}).values()
    assert not any("stocks" in v.lower() for v in labels)  # ordinary days never make "stocks fall" headlines
    w2, rec2 = run()
    assert _series(rec2, "us_stocks_quiet_pct") == quiet  # deterministic per seed
    w3, rec3 = run(market_quiet=False)
    assert set(_series(rec3, "us_stocks_quiet_pct").values()) == {0.0} and w3.macro["gold_index"] == 100.0


def test_no_ordinary_movement_on_announcement_days():
    from jahan.scenario.spec import Announcement

    w = _world(n=600, market_kappa=0.0, market_savings=False, countries=("USA", "DEU"))
    w.macro["economy"]["others"]["cfg"]["quiet"] = QUIET
    ann = Announcement(indicator="us_payrolls", actual=165, expected=65, headline="Jobs beat", start_day_offset=1)
    apply_scenario(w, Scenario(prompt="jobs", events=[ann]), EventLog(), 1)
    rec = run_in_memory(w, 3)
    quiet = _series(rec, "us_stocks_quiet_pct")
    assert quiet[1] == 0.0 and quiet[0] != 0.0 and quiet[2] != 0.0  # Wednesday is the release day


def test_ordinary_days_do_not_count_toward_a_scripted_gold_headline():
    w = _world(n=600, market_kappa=0.0, market_savings=False, countries=("USA", "DEU"))
    w.macro["economy"]["others"]["cfg"]["quiet"] = QUIET
    step = MacroPath(variable="gold", steps=[MacroStep(days=1, pct_per_day=2.0)], auto_news=True)
    apply_scenario(w, Scenario(prompt="gold", events=[step]), EventLog(), 1)
    run_in_memory(w, 1)  # the scripted step makes a headline and sets the reference
    assert w.macro["news_ref"]["gold"] == pytest.approx(w.macro["gold_index"])
    run_in_memory(w, 20)  # ordinary days move gold, and its reference with it
    assert w.macro["gold_index"] != pytest.approx(100.0 * 1.02, abs=0.01)
    assert w.macro["news_ref"]["gold"] == pytest.approx(w.macro["gold_index"])


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
