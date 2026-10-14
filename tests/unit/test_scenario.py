"""Scenario smoke tests (sanity of mechanics, not empirical validation)."""

from pathlib import Path

import numpy as np

from jahan.model.enums import EventKind
from jahan.rng import Stream
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.availability import compute_masks
from jahan.sim.media import media_exposure
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

SCEN = Path(__file__).resolve().parents[2] / "examples" / "scenarios" / "launch_product_x.yaml"


def _series(rec, metric):
    t = rec.metrics.table().to_pydict()
    return {
        d: v
        for d, c, it, m, v in zip(t["day"], t["country"], t["item"], t["metric"], t["value"], strict=True)
        if m == metric and it == 0 and c == 0
    }


def test_baseline_has_no_adoption_and_treated_is_consistent(small_world_factory):
    base = small_world_factory(1500, seed=5)
    treated = small_world_factory(1500, seed=5)
    run_in_memory(base, 3)
    run_in_memory(treated, 3)
    apply_scenario(treated, Scenario.load(SCEN), EventLog(), 1)
    rb = run_in_memory(base, 90)
    rt = run_in_memory(treated, 90)
    kinds_b = np.array(rb.events.table().column("kind").to_pylist())
    assert not (kinds_b == EventKind.ADOPTED).any()
    aware = _series(rt, "aware_share")
    adopted = _series(rt, "adopted_share_adults")
    n_aware = _series(rt, "n_aware")
    n_adopted = _series(rt, "n_adopted")
    assert all(n_aware[d] >= n_adopted[d] for d in n_aware)
    days = sorted(aware)
    assert all(aware[a] <= aware[b] + 1e-12 for a, b in zip(days, days[1:], strict=False))  # awareness never falls
    assert adopted[days[-1]] > 0
    age = treated.ages()
    adopters = treated.beliefs.m["adopted_day"][: treated.pop.n, 0] >= 0
    assert (age[adopters] >= treated.config.params.min_adopt_age - 90 / 365).all()


def test_common_random_numbers_keep_untouched_draws_identical(small_world_factory):
    a = small_world_factory(800, seed=9)
    b = small_world_factory(800, seed=9)
    ra = run_in_memory(a, 10)
    apply_scenario(b, Scenario.load(SCEN), EventLog(), 1)
    rb = run_in_memory(b, 10)
    ca = _contacts(ra)
    cb = _contacts(rb)
    assert ca == cb  # contact sampling does not depend on the scenario


def _contacts(rec):
    t = rec.metrics.table().to_pydict()
    return [v for m, v in zip(t["metric"], t["value"], strict=True) if m == "contacts"]


def test_price_lowers_adoption(small_world_factory):
    shares = []
    for price in (49, 1999):
        w = small_world_factory(1500, seed=21)
        sc = Scenario.load(SCEN)
        sc.events[0].price_usd = price
        apply_scenario(w, sc, EventLog(), 1)
        run_in_memory(w, 120)
        shares.append((w.beliefs.m["adopted_day"][: w.pop.n, 0] >= 0).mean())
    assert shares[0] > shares[1]


def test_asleep_at_release_sees_tv_later(small_world_factory):
    """TV airs 18-23 local.  An item released 04:00 UTC is 23:00 in US Eastern (after the evening news)
    but 20:00 in US Pacific (during it): on release day only Pacific viewers can see it on TV."""
    w = small_world_factory(3000, seed=4)
    p = w.config.params.model_copy()
    p.media_attention = 1.0
    sc = Scenario.load(SCEN)
    ev = sc.events[0]
    ev.release_utc_hour = 4
    ev.salience = 1.0
    ev.pushes[0].channel_kinds = ["tv"]
    apply_scenario(w, sc, EventLog(), 1)
    log = EventLog()
    media_exposure(w, compute_masks(w, p), p, log)
    t = log.table().to_pydict()
    exposed = np.array(t["agent"], dtype=np.int64)
    tz = w.pop["tz_offset_min"][exposed]
    assert exposed.size > 0
    assert not (tz == -300).any()
    assert (tz == -480).any()
    _ = Stream
