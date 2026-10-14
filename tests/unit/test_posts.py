"""Posts made inside the world: records with reach, the briefing, figures' real posts as grounding."""

import numpy as np

from jahan.actors import autonomy, engine
from jahan.actors.actions import Claim, Control, CreateAgent, Post
from jahan.actors.briefing import briefing
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.query.persona import interview
from jahan.query.social_feed import feed_rows, feed_text
from jahan.query.status import eligible_audience
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario, SocialDigest, Statement
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog


def world(n=2000, countries=("USA",), seed=3):
    return generate_world(RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13"))


def apply(w, *events, pid=1):
    log = EventLog()
    apply_scenario(w, Scenario(prompt="t", events=list(events)), log, pid)
    return log


def test_a_players_post_is_recorded_with_its_reach_and_shown_in_the_briefing():
    w = world()
    me = engine.create_agent(w, CreateAgent(country="USA", visibility=3.0, friends=8), EventLog(), 1)
    apply(w, Control(agent=me, session=1))
    claim = Claim(headline="The new clinic downtown is excellent", topic="health", stance=1.0)
    apply(w, Post(actor=me, session=1, claim=claim, audience="community", utc_hour=15), pid=2)
    run_in_memory(w, 3)
    (rec,) = w.macro["posts"].values()
    assert rec["actor"] == me and rec["kind"] == "post" and rec["audience"] == "community"
    assert rec["reached"] > 0 and rec["reached_people"] > 0 and 0 <= rec["believed"] <= rec["reached"]
    text = briefing(w, me)
    assert "Your posts (last 7 days):" in text and "The new clinic downtown is excellent" in text
    rows = feed_rows(w, kind="virtual")
    assert rows[0]["name"] and rows[0]["reached"] == rec["reached"]
    assert "[simulated]" in feed_text(rows)


def test_a_figures_statement_is_a_post_and_their_real_posts_ground_their_turns():
    w = world()
    apply(w, Statement(speaker="USA.head_of_state", headline="Tariffs are working", topic="economy", valence=0.8))
    run_in_memory(w, 2)
    rec = next(iter(w.macro["posts"].values()))
    assert rec["kind"] == "statement" and rec["reached"] > 0
    digest = SocialDigest(
        fetch_day="2026-10-14",
        figures={
            "USA.head_of_state": [
                {
                    "platform": "x",
                    "url": "u",
                    "created_utc": "2026-10-13T10:00:00",
                    "text": "Real post text",
                    "is_repost": False,
                }
            ]
        },
        trending={"USA": [{"topic": "economy", "n": 12, "mean_valence": -0.3, "top_handles": []}]},
    )
    apply(w, digest, pid=2)
    col = next(i for i, e in enumerate(w.notables) if e["key"] == "USA.head_of_state")
    cand = {"col": col, "agent": w.notables[col]["agent_id"], "salience": 1.0, "reasons": [(1.0, "k", "test")]}
    brief = autonomy._brief(w, cand)
    assert "What you said publicly lately (real posts, fetched 2026-10-14):" in brief and "Real post text" in brief
    assert "economy: 12 posts" in brief and "uncalibrated" in brief
    snap = interview(None, w, int(w.notables[col]["agent_id"]), "How is the economy?").snapshot
    assert "Real post text" in snap


def test_eligible_audience_grows_with_the_audience():
    w = world()
    me = engine.create_agent(w, CreateAgent(country="USA", friends=6, acquaintances=6), EventLog(), 1)
    reach = eligible_audience(w, w.agent_row(me))
    sizes = [reach[a]["people"] for a in ("circle", "community", "national", "global")]
    assert sizes[0] >= 6 and sizes == sorted(sizes) and reach["national"]["represents"] > 1e8
    assert np.isclose(reach["national"]["people"], int((w.pop["alive"] & ~w.pop["is_notable"]).sum()))
