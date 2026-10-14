"""Ordinary people repost what they believe: eligibility, caps, determinism and what it changes."""

import numpy as np

from jahan.actors import engine
from jahan.actors.actions import Claim, Control, CreateAgent, Post
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import CHANNEL_KIND_NAMES, Big5, EventKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import NewsEvent, ProductLaunch, Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

SOCIAL = int(CHANNEL_KIND_NAMES["social"])
NEWS = [
    NewsEvent(headline="Inflation jumps", topic="economy", valence=-0.6, salience=0.8, countries=["USA"]),
    NewsEvent(headline="Team wins the cup", topic="sports", valence=0.7, salience=0.7, countries=["USA"]),
]


def world(n=3000, seed=7, **params):
    cfg = RunConfig(n_agents=n, countries=["USA"], seed=seed, start_date="2026-10-13")
    for k, v in params.items():
        setattr(cfg.params, k, v)
    return generate_world(cfg)


def shared(rec):
    t = rec.events.table().to_pydict()
    return [
        (d, a, i, o)
        for k, d, a, i, o in zip(t["kind"], t["day"], t["agent"], t["item"], t["other"], strict=True)
        if k == EventKind.SOCIAL_SHARED
    ]


def run(days=10, events=NEWS, **params):
    w = world(**params)
    apply_scenario(w, Scenario(prompt="t", events=list(events)), EventLog(), 1)
    return w, run_in_memory(w, days)


def test_reposts_are_logged_and_never_lower_awareness():
    on_w, on = run()
    off_w, off = run(social_reposts=False)
    assert shared(on) and not shared(off)
    for item in range(2):
        slot = int(on_w.items["slot"][item])
        a_on = on_w.beliefs.m["aware"][: on_w.pop.n, slot].sum()
        a_off = off_w.beliefs.m["aware"][: off_w.pop.n, int(off_w.items["slot"][item])].sum()
        assert a_on >= a_off
    assert on_w.macro["social"]["reposts"] and sum(on_w.macro["social"]["reposts_by_day"].values()) == len(shared(on))
    online = int((on_w.pop["alive"] & on_w.pop["internet"]).sum())
    per_day = np.bincount([d for d, *_ in shared(on)])
    assert per_day.max() < 0.05 * online  # tripwire from the plan: reposts stay a small share of people online


def test_caps_hold_when_everyone_wants_to_repost():
    w, rec = run(days=20, repost_b0=5.0)
    rows = shared(rec)
    assert rows
    pairs = [(a, i) for _, a, i, _ in rows]
    assert len(pairs) == len(set(pairs))  # once per person and item
    days = [(d, a) for d, a, _, _ in rows]
    assert len(days) == len(set(days))  # one repost per person per day
    assert (w.pushes["intensity"][: w.pushes.n] <= 1.0 + 1e-6).all()
    for item in range(2):
        assert int(w.items["active_until"][item]) <= int(w.items["created_day"][item]) + 30
    tails = np.flatnonzero(w.pushes["start_day"][: w.pushes.n] > 0)
    assert tails.size and (w.channels["kind"][w.pushes["channel_id"][tails]] == SOCIAL).all()


def test_who_reposts():
    w, rec = run(days=10, repost_b0=-2.0)
    rows = shared(rec)
    who = np.array(sorted({a for _, a, _, _ in rows}))
    pop = w.pop
    assert not pop["is_notable"][who].any() and (pop["controlled_by"][who] < 0).all() and pop["internet"][who].all()
    assert (w.ages()[who] >= w.config.params.share_min_age).all()
    adults = np.flatnonzero(pop["alive"] & ~pop["is_notable"] & pop["internet"] & (w.ages() >= 18))
    assert pop["big5"][who, Big5.EXTRAVERSION].mean() > pop["big5"][adults, Big5.EXTRAVERSION].mean()


def test_reposts_are_deterministic():
    assert shared(run(days=8)[1]) == shared(run(days=8)[1])


def test_products_are_never_reposted():
    launch = ProductLaunch(
        name="Ring", topic="tech", price_usd=199, true_quality=0.8, countries=["USA"], valence=0.8, salience=0.9
    )
    w, rec = run(days=12, events=[launch], repost_b0=5.0)
    assert not shared(rec)


def test_an_ordinary_post_reaches_social_media_only_through_reposts():
    def go(**params):
        w = world(n=2500, **params)
        me = engine.create_agent(w, CreateAgent(country="USA", visibility=3.0, friends=20), EventLog(), 1)
        apply_scenario(w, Scenario(prompt="t", events=[Control(agent=me, session=1)]), EventLog(), 1)
        claim = Claim(headline="The river is polluted again", topic="local", stance=-1.0)
        post = Post(actor=me, session=1, claim=claim, audience="community", utc_hour=10)
        apply_scenario(w, Scenario(prompt="p", events=[post]), EventLog(), 2)
        run_in_memory(w, 10)
        item = int(next(iter(w.macro["posts"].values()))["item"])
        pushed = np.flatnonzero(w.pushes["item_id"][: w.pushes.n] == item)
        return w, item, pushed

    # word of mouth moves views only partway, so the conviction bar is lowered to make reposts likely here
    w, item, pushed = go(repost_b0=5.0, repost_min_conviction=0.05)
    assert pushed.size and (w.channels["kind"][w.pushes["channel_id"][pushed]] == SOCIAL).all()
    rec = next(iter(w.macro["posts"].values()))
    assert rec["reposts"] == w.macro["social"]["reposts"][str(item)]["n"] > 0
    _, _, none = go(social_reposts=False, repost_min_conviction=0.05)
    assert none.size == 0
