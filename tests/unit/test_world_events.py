from pathlib import Path

import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind, OriginKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

EX = Path(__file__).resolve().parents[2] / "examples" / "scenarios"


def _world(countries, n=2000, seed=4):
    return generate_world(RunConfig(n_agents=n, countries=countries, seed=seed, start_date="2026-10-13"))


def _series(rec, metric, country):
    t = rec.metrics.table().to_pydict()
    return {
        d: v
        for d, c, m, v in zip(t["day"], t["country"], t["metric"], t["value"], strict=True)
        if m == metric and c == country
    }


def test_gold_path_and_auto_news():
    w = _world(["USA", "IRN"])
    apply_scenario(w, Scenario.load(EX / "gold_spike.yaml"), EventLog(), 1)
    rec = run_in_memory(w, 10)
    assert abs(w.macro["gold_index"] - 100 * 1.01**7 * 0.9) < 1e-9
    created = [
        d
        for d, k in zip(
            rec.events.table().column("detail").to_pylist(), rec.events.table().column("kind").to_pylist(), strict=True
        )
        if k == EventKind.ITEM_CREATED
    ]
    assert any("Gold price jumps" in d for d in created) and any("Gold price drops" in d for d in created)
    macro_items = np.flatnonzero(w.items["origin_kind"] == OriginKind.MACRO)
    # repeated daily moves refresh one headline instead of creating seven
    assert 1 <= macro_items.size <= 3


def test_dollar_rise_raises_local_prices_and_lowers_adoption():
    shares = {}
    prices = {}
    for label, path in (("base", EX / "launch_product_x.yaml"), ("dollar", EX / "dollar_rises.yaml")):
        w = _world(["USA", "IRN", "IND"], n=4000, seed=9)
        sc = Scenario.load(path)
        if label == "base":  # same launch, worldwide, without the currency move
            sc = Scenario.load(EX / "dollar_rises.yaml")
            sc.events = [e for e in sc.events if e.type == "product_launch"]
            sc.events[0].countries = ["USA", "IRN", "IND"]
        else:
            sc.events[0].countries = ["USA", "IRN", "IND"]
        apply_scenario(w, sc, EventLog(), 1)
        run_in_memory(w, 60)
        irn = 1
        prices[label] = float(w.product_price[0, irn])
        m = (w.pop["country"] == irn) & (w.ages() >= 16)
        shares[label] = float((w.beliefs.m["adopted_day"][: w.pop.n, 0][m] >= 0).mean())
        if label == "dollar":
            assert abs(w.product_price[0, 0] - 199) < 1e-3  # USD price unchanged in the USA
    assert abs(prices["dollar"] / prices["base"] - 1.2) < 0.01
    assert shares["dollar"] <= shares["base"]


def test_false_rumour_lowers_views_of_the_product():
    views = {}
    for label in ("launch_only", "rumour"):
        w = _world(["USA"], n=3000, seed=12)
        sc = Scenario.load(EX / "product_rumour.yaml")
        if label == "launch_only":
            sc.events = sc.events[:1]
        apply_scenario(w, sc, EventLog(), 1)
        run_in_memory(w, 45)
        aware = w.beliefs.m["aware"][: w.pop.n, 0] == 1
        views[label] = float(w.beliefs.m["belief"][: w.pop.n, 0][aware].mean())
    assert views["rumour"] < views["launch_only"]


def test_running_currency_story_reports_cumulative_move():
    w = _world(["USA", "IRN"], n=1000, seed=3)
    apply_scenario(w, Scenario.load(EX / "iran_us_tension.yaml"), EventLog(), 1)
    rec = run_in_memory(w, 12)
    t = rec.events.table().to_pydict()
    labels = [
        d
        for k, d in zip(t["kind"], t["detail"], strict=True)
        if k in (EventKind.ITEM_CREATED, EventKind.ITEM_UPDATED) and "against the US dollar" in d
    ]
    assert labels, "no currency headline"
    final = labels[-1]
    pct = float(final.split(" falls ")[1].split("%")[0])
    assert abs(pct - 100 * (1.0142**10 - 1)) < 3.0  # the story ends near the full 15% slide


def test_bad_news_lowers_sentiment_in_target_country():
    base = _world(["USA", "IRN"], n=3000, seed=2)
    hit = _world(["USA", "IRN"], n=3000, seed=2)
    apply_scenario(hit, Scenario.load(EX / "iran_us_tension.yaml"), EventLog(), 1)
    rb = run_in_memory(base, 20)
    rh = run_in_memory(hit, 20)
    irn = 1
    sb = _series(rb, "sentiment", irn)
    sh = _series(rh, "sentiment", irn)
    assert sh[19] < sb[19] - 0.01
    fx = _series(rh, "fx_local_per_usd", irn)
    assert abs(fx[19] / fx[0] - 1.0142**9) < 0.01  # day-0 value already includes the first step
