"""Real social posts inside the world: pushes, origins, feed weighting, reach and the tracked-item pool."""

import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import CHANNEL_KIND_NAMES, EventKind, OriginKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import NewsEvent, ProductLaunch, Scenario, SocialDigest, SocialPostEvent
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

SOCIAL = int(CHANNEL_KIND_NAMES["social"])


def world(countries=("USA",), n=3000, seed=4, **params):
    cfg = RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13")
    for k, v in params.items():
        setattr(cfg if k == "max_slots" else cfg.params, k, v)
    return generate_world(cfg)


def social(handle="POTUS", key="USA.head_of_state", vis=8.7, text="We are cutting taxes", **kw):
    base = {
        "platform": "x",
        "post_id": f"{handle}-{text[:8]}",
        "author_handle": handle,
        "author_key": key,
        "author_visibility": vis,
        "engagement": 0.6,
        "headline": text,
        "topic": "economy",
        "valence": 0.5,
        "language": "en",
        "countries": ["USA"],
        "url": f"https://x.com/{handle}/status/1",
        "mentions": ["@SecScottBessent"],
    }
    base.update(kw)
    return SocialPostEvent(**base)


def apply(w, *events, pid=1):
    log = EventLog()
    apply_scenario(w, Scenario(prompt="t", events=list(events)), log, pid)
    return log


def events_of(log, kind):
    t = log.table().to_pydict()
    return [
        (a, o, i, d)
        for k, a, o, i, d in zip(t["kind"], t["agent"], t["other"], t["item"], t["detail"], strict=True)
        if k == kind
    ]


def test_posts_push_only_social_channels_in_their_language():
    w = world(("USA", "IRN"), n=2000)
    apply(
        w, social(), social(handle="", key=None, vis=2.0, text="Prices are up", author_name="a user with 300 followers")
    )
    figure, anon = w.items.n - 2, w.items.n - 1
    for item in (figure, anon):
        rows = np.flatnonzero(w.pushes["item_id"] == item)
        assert rows.size and (w.channels["kind"][w.pushes["channel_id"][rows]] == SOCIAL).all()
        assert (w.channels["lang"][w.pushes["channel_id"][rows]] == w.lang_id("en")).all()
        assert (w.channels["country"][w.pushes["channel_id"][rows]] == 0).all()
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    assert w.items["origin_kind"][figure] == OriginKind.NOTABLE and w.items["origin_id"][figure] == pres["agent_id"]
    assert w.items["origin_kind"][anon] == OriginKind.SOCIAL
    labels = w.macro["item_labels"]
    assert labels[str(figure)].startswith("@POTUS on X: ") and labels[str(anon)].startswith(
        "a user with 300 followers on X"
    )
    meta = w.macro["social"]["items"][str(figure)]
    assert meta["mentions"] == ["@SecScottBessent"] and meta["url"].startswith("https://x.com/POTUS")
    assert w.items["salience"][figure] > w.items["salience"][anon]  # reach from visibility, not wording


def test_a_post_in_a_language_nobody_reads_is_skipped_and_logged():
    w = world(n=1500)
    log = apply(w, social(language="fa"))
    assert w.items.n == 0
    assert any("no social channel in fa" in d for *_, d in events_of(log, EventKind.ACTION_SKIPPED))


def test_digest_and_scheduled_posts():
    w = world(n=1500)
    digest = SocialDigest(
        fetch_day="2026-10-14",
        seen=["x:1", "x:2"],
        figures={"USA.head_of_state": [{"platform": "x", "url": "u1", "created_utc": "2026-10-13", "text": "a"}]},
        trending={"USA": [{"topic": "economy", "n": 3, "mean_valence": -0.2, "top_handles": []}]},
        feed={"USA": [{"text": "a", "tracked": True}]},
    )
    apply(w, digest, social(start_day_offset=1))
    st = w.macro["social"]
    assert st["seen"] == ["x:1", "x:2"] and st["last_fetch"]["fetch_day"] == "2026-10-14"
    assert st["feed"]["USA"][0]["world_day"] == 0 and st["trending"]["USA"][0]["topic"] == "economy"
    assert w.items.n == 0  # starts tomorrow
    run_in_memory(w, 2)
    assert w.items.n == 1 and w.items["origin_kind"][0] == OriginKind.NOTABLE
    apply(w, digest)  # the same digest again changes nothing
    assert w.macro["social"]["seen"] == ["x:1", "x:2"] and len(w.macro["social"]["figures"]["USA.head_of_state"]) == 1


def exposures(rec, item):
    t = rec.events.table().to_pydict()
    return {
        (a, d)
        for k, a, i, d in zip(t["kind"], t["agent"], t["item"], t["day"], strict=True)
        if k == EventKind.INFO_EXPOSED and i == item
    }


def test_feed_weighting_touches_social_posts_only():
    news = NewsEvent(
        headline="Storm warning", topic="local", valence=-0.5, salience=0.8, countries=["USA"], channel_kinds=["social"]
    )
    got = []
    for weight in (0.0, 1.0):
        w = world(n=2000, social_interest_weight=weight, social_affinity_base=0.5 * weight)
        apply(w, news)
        got.append(exposures(run_in_memory(w, 5), 0))
    assert got[0] == got[1] and got[0]  # news on social channels is unchanged
    reached = []
    for weight in (0.0, 1.0):
        w = world(n=4000, social_interest_weight=weight, social_affinity_base=0.0)
        apply(w, social(handle="", key=None, vis=7.0, topic="sports", text="Big match tonight"))
        rec = run_in_memory(w, 5)
        who = np.array(sorted(a for a, _ in exposures(rec, 0)))
        reached.append(float(w.pop["interests"][who, 4].mean()))  # topic 4 = sports
    assert reached[1] > reached[0]  # with the weight on, the people reached care more about the topic


def test_a_presidents_post_reaches_far_more_than_an_unknown_account():
    w = world(n=4000)
    apply(w, social(text="Big announcement"), social(handle="nobody", key=None, vis=2.0, text="Big announcement too"))
    run_in_memory(w, 5)
    aware = [int(w.beliefs.m["aware"][: w.pop.n, int(w.items["slot"][i])].sum()) for i in (0, 1)]
    assert aware[0] >= 3 * max(aware[1], 1)


def test_daily_ingests_never_lose_new_items_and_evict_only_stale_ones():
    w = world(n=1500, max_slots=48)
    apply(w, ProductLaunch(name="Phone", topic="tech", price_usd=500, true_quality=0.7, countries=["USA"]))
    evicted, created_fail = [], 0
    for day in range(14):
        events = [
            NewsEvent(headline=f"News {day}-{k}", topic="world", valence=0.0, countries=["USA"], duration_days=2)
            for k in range(4)
        ]
        events += [social(text=f"Post {day}-{k}", duration_days=2) for k in range(3)]
        log = apply(w, *events, pid=day + 2)
        created_fail += sum("no free slot" in d for *_, d in events_of(log, EventKind.ITEM_CREATED))
        for _, _, item, _ in events_of(log, EventKind.ITEM_EVICTED):
            assert w.items["created_day"][item] < w.day
            rows = np.flatnonzero(w.pushes["item_id"] == item)
            assert not (w.pushes["end_day"][rows] > w.day).any()  # never while still broadcast
            evicted.append(item)
        run_in_memory(w, 1)
    assert created_fail == 0
    assert evicted, "the pool was full, so old items should have given up their slots"
    assert int(w.products["item_id"][0]) not in evicted  # products keep their slot
    assert w.items["slot"][int(w.products["item_id"][0])] >= 0
