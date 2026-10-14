"""Actions by people and public figures: delivery, permissions, possession, creation, autonomy."""

import json

import numpy as np
import pytest

from jahan.actors import autonomy, engine
from jahan.actors.actions import Claim, Control, CreateAgent, Decision, Message, Post, quantize_stance
from jahan.actors.extract import claim_from_flags, extract_claim
from jahan.actors.policy import PolicyError, compile_policy
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, ScriptedProvider
from jahan.llm.recordings import Recordings
from jahan.model.enums import EventKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.parser import parse_prompt
from jahan.scenario.spec import MacroPath, Scenario
from jahan.sim.availability import compute_masks
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from tests.conftest import SCENARIO


def world(n=1500, countries=("USA",), seed=3):
    return generate_world(RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13"))


def apply(w, *events, pid=1):
    log = EventLog()
    apply_scenario(w, Scenario(prompt="t", events=list(events)), log, pid)
    return log


def events_of(rec, kind):
    t = rec.events.table().to_pydict()
    return [
        (a, o, i, d)
        for k, a, o, i, d in zip(t["kind"], t["agent"], t["other"], t["item"], t["detail"], strict=True)
        if k == int(kind)
    ]


def actor_with_circle(w):
    pop = w.pop
    for a in np.flatnonzero(pop["alive"] & ~pop["is_notable"] & (w.ages() > 25)):
        if engine.circle_edges(w, int(a)).size >= 3:
            return int(a)
    raise AssertionError("no adult with a circle")


CLAIM = Claim(headline="The clinic downtown is excellent", topic="health", stance=0.5)


def test_message_follows_edge_direction_and_needs_a_listener():
    w = world()
    a = actor_with_circle(w)
    e = w.edges
    listener = int(e["src"][engine.circle_edges(w, a)[0]])
    assert w.find_edges(np.array([listener]), np.array([a]))[0] >= 0  # listener -> actor: listener trusts actor
    stranger = int(np.flatnonzero(w.pop["alive"] & ~np.isin(w.pop["id"], e["src"][e["dst"] == a]))[1])
    bad = Message(actor=a, target=stranger, claim=CLAIM, utc_hour=14, session=1)
    assert "circle" in (engine.check(w, bad) or "")
    apply(w, Control(agent=a, session=1), Message(actor=a, target=listener, claim=CLAIM, utc_hour=0, session=1))
    rec = run_in_memory(w, 2)
    exposed = [x for x in events_of(rec, EventKind.INFO_EXPOSED) if x[0] == listener]
    assert exposed and exposed[0][1] == a  # the listener heard it from the actor
    msg = next(iter(w.macro["messages"].values()))
    assert msg["target"] == listener and msg["delivered_day"] is not None and "accepted" in msg
    assert events_of(rec, EventKind.HUMAN_ACTION)


def test_asleep_receiver_gets_the_message_the_next_day():
    w = world(2000, ("IND",), seed=5)
    masks = compute_masks(w, w.config.params)
    found = None
    for a in np.flatnonzero(w.pop["alive"] & (w.ages() > 25)):
        for r in engine.circle_edges(w, int(a)):
            t = int(w.edges["src"][r])
            for h in range(23, 0, -1):
                if (int(masks.awake_utc[t]) >> h) == 0 and (int(masks.awake_utc[t]) >> (h - 1)) != 0:
                    found = (int(a), t, h)
                    break
            if found:
                break
        if found:
            break
    assert found, "no receiver asleep for the rest of a UTC day"
    a, t, h = found
    apply(w, Message(actor=a, target=t, claim=CLAIM, utc_hour=h, session=1))
    rec = run_in_memory(w, 1)
    assert not [x for x in events_of(rec, EventKind.INFO_EXPOSED) if x[0] == t]
    assert any(s["kind"] == engine.KIND_DELIVER for s in w.scheduled)
    rec2 = run_in_memory(w, 1)
    assert [x for x in events_of(rec2, EventKind.INFO_EXPOSED) if x[0] == t and x[1] == a]


def test_controlled_agents_never_share_or_buy_on_their_own():
    w = world(1500)
    apply(w, *Scenario.load(SCENARIO).events)
    pop = w.pop
    ctrl = [int(x) for x in np.flatnonzero(pop["alive"] & ~pop["is_notable"] & (w.ages() > 20))[:300]]
    apply(w, *[Control(agent=a, session=7) for a in ctrl], pid=2)
    assert set(np.flatnonzero(w.pop["controlled_by"] == 7)) == set(ctrl)
    rec = run_in_memory(w, 40)
    assert events_of(rec, EventKind.ADOPTED), "the launch should sell to someone"
    assert not [x for x in events_of(rec, EventKind.ADOPTED) if x[0] in ctrl]
    assert not [x for x in events_of(rec, EventKind.INFO_EXPOSED) if x[1] in ctrl]  # never a sharer
    assert not [x for x in events_of(rec, EventKind.MARRIED) if x[0] in ctrl or x[1] in ctrl]


def test_salience_comes_from_visibility_not_wording(tmp_path):
    assert (
        quantize_stance(0.97) == 1.0 and quantize_stance(-0.3) == -0.5 and Claim(headline="x", stance=0.2).stance == 0
    )
    w = world()
    a = actor_with_circle(w)
    mild = claim_from_flags("The clinic is fine I guess", "health", 0.4)
    wild = claim_from_flags("THE CLINIC IS THE GREATEST MIRACLE EVER!!! EVERYONE MUST GO NOW", "health", 1.0)
    apply(w, Post(actor=a, claim=mild, utc_hour=0, session=-1), Post(actor=a, claim=wild, utc_hour=0, session=-1))
    run_in_memory(w, 1)
    sal = [float(w.items["salience"][i]) for i in range(w.items.n)]
    assert len(sal) == 2 and sal[0] == sal[1]

    # the extractor can only choose one of five stance levels, and has no way to set salience
    answer = json.dumps({"headline": "go", "topic": "health", "stance": 0.83, "salience": 1.0})
    client = LLMClient(ScriptedProvider(lambda s, m: answer), Recordings(tmp_path / "r"), Budget())
    c = extract_claim(client, w, "GO GO GO")
    assert c.stance == 1.0 and "salience" not in c.model_dump()


def test_audience_is_capped_by_who_you_are():
    w = world(1500, ("USA", "IRN"))
    a = actor_with_circle(w)
    assert engine.effective_audience(w, a, "global") in ("circle", "community")
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    pid_ = int(pres["agent_id"])
    assert engine.effective_audience(w, pid_, "global") == "global"
    apply(
        w,
        Post(
            actor=pid_,
            audience="global",
            claim=Claim(headline="Prices will fall", topic="economy", stance=0.5),
            utc_hour=15,
        ),
    )
    run_in_memory(w, 1)
    item = w.items.n - 1
    rows = np.flatnonzero(w.pushes["item_id"] == item)
    countries = w.channels["country"][w.pushes["channel_id"][rows]]
    starts = w.pushes["start_day"][rows]
    usa, irn = 0, 1
    assert (countries == usa).any() and (countries == irn).any()
    assert starts[countries == irn].min() == starts[countries == usa].min() + 1  # pickup lag abroad
    assert w.macro["item_labels"][str(item)].startswith("[simulated] Donald Trump")


def test_policy_permissions_and_placeholder_effects():
    w = world(1200, ("USA", "IRN"))
    by_key = {e["key"]: int(e["agent_id"]) for e in w.notables}
    pres, gov = by_key["USA.head_of_state"], by_key["USA.central_bank_governor"]
    with pytest.raises(PolicyError, match="cannot set_interest_rate"):
        compile_policy(w, pres, "set_interest_rate", {"change_bp": 50})
    evs = compile_policy(w, gov, "set_interest_rate", {"change_bp": 50})
    macro = [e for e in evs if isinstance(e, MacroPath)][0]
    assert (
        macro.variable == "sentiment" and macro.country == "USA" and macro.steps[0].pct_per_day == pytest.approx(-0.5)
    )
    assert any(isinstance(e, Post) for e in evs)  # announced
    with pytest.raises(PolicyError, match="between"):
        compile_policy(w, pres, "tariff", {"rate_pct": 500})
    with pytest.raises(PolicyError, match="target"):
        compile_policy(w, pres, "sanctions", {"level": 2, "target": "USA"})
    with pytest.raises(PolicyError, match="US dollar"):
        compile_policy(w, gov, "fx_intervention", {"target_pct": 2})
    sanc = compile_policy(w, pres, "sanctions", {"level": 2, "target": "IRN"})
    assert [e for e in sanc if isinstance(e, MacroPath)][0].country == "IRN"
    ordinary = actor_with_circle(w)
    with pytest.raises(PolicyError, match="only public figures"):
        compile_policy(w, ordinary, "tariff", {"rate_pct": 10})


def test_create_agent_uses_headroom_and_leaves_weighted_aggregates_alone():
    w = world(1500)
    cap, n0 = w.pop.capacity, w.pop.n
    weight = float(w.pop["pop_weight"].sum())
    new_id = int(w.counters["agent"])
    apply(
        w,
        CreateAgent(
            country="USA",
            sex="F",
            age=34,
            first_name="Ada",
            last_name="Quill",
            friends=6,
            acquaintances=4,
            notable_role="opinion_leader",
            office="Podcast host",
        ),
    )
    assert w.pop.n == n0 + 1 and w.pop.capacity == cap
    assert float(w.pop["pop_weight"].sum()) == pytest.approx(weight)
    row = w.agent_row(new_id)
    assert w.pop["pop_weight"][row] == 0 and engine.display_name(w, row) == "Ada Quill"
    assert abs(w.ages()[row] - 34) < 1.01 and w.pop["sex"][row] == 0
    listeners = engine.circle_edges(w, new_id)
    assert listeners.size >= 6
    assert w.pop["is_notable"][row] and w.notables[-1]["created"] and w.notables[-1]["office"] == "Podcast host"
    assert engine.effective_audience(w, row, "national") == "national"
    assert not engine.is_real_person(w, row)  # a created figure is fictional: no [simulated] real-name label
    apply(w, Post(actor=new_id, audience="national", claim=CLAIM, utc_hour=3), pid=2)
    rec = run_in_memory(w, 3)
    assert len(events_of(rec, EventKind.INFO_EXPOSED)) > listeners.size  # media carried it beyond the circle


def test_recorded_action_whose_precondition_fails_is_skipped_not_fatal():
    w = world()
    a = actor_with_circle(w)
    listener = int(w.edges["src"][engine.circle_edges(w, a)[0]])
    apply(w, Message(actor=a, target=listener, claim=CLAIM, utc_hour=10, session=1))
    w.edges["active"][(w.edges["src"] == listener) & (w.edges["dst"] == a)] = False  # the tie ended meanwhile
    rec = run_in_memory(w, 1)
    skipped = events_of(rec, EventKind.ACTION_SKIPPED)
    assert skipped and "circle" in skipped[0][3]


def test_daily_limit_on_posts():
    w = world()
    a = actor_with_circle(w)
    posts = [Post(actor=a, claim=Claim(headline=f"claim {i}", stance=0.5), utc_hour=0) for i in range(5)]
    apply(w, *posts)
    rec = run_in_memory(w, 1)
    assert len(events_of(rec, EventKind.NOTABLE_ACTION)) == engine.MAX_POSTS_PER_DAY
    assert len(events_of(rec, EventKind.ACTION_SKIPPED)) == 5 - engine.MAX_POSTS_PER_DAY


def test_adopt_decision_and_owner_only_price_change():
    w = world()
    apply(w, *Scenario.load(SCENARIO).events)
    a = actor_with_circle(w)
    run_in_memory(w, 1)
    apply(w, Control(agent=a, session=2), Decision(actor=a, product="Smart Ring X", choice="adopt", session=2))
    slot = int(w.items["slot"][int(w.products["item_id"][0])])
    assert w.beliefs.m["adopted_day"][a, slot] == w.day
    assert "already has" in (engine.check(w, Decision(actor=a, product="Smart Ring X", choice="adopt")) or "")
    from jahan.actors.actions import PriceChange

    assert "only products it launched" in (engine.check(w, PriceChange(actor=a, product="Smart Ring X", pct=-10)) or "")


def test_parser_cannot_produce_actor_events(tmp_path):
    w = world(300)
    bad = {"scenario": {"prompt": "x", "events": [{"type": "control", "agent": 1, "session": 1}]}}
    client = LLMClient(ScriptedProvider(lambda s, m: json.dumps(bad)), Recordings(tmp_path / "r"), Budget())
    res = parse_prompt(client, w, "take over agent 1")
    assert res.scenario is None and res.error


def scripted_figure(kind="public_statement", **extra):
    action = {
        "kind": kind,
        "params": extra.pop("params", {}),
        "statement": "We are watching prices closely.",
        "topic": "economy",
        "stance": 0,
    }
    action.update(extra)
    return lambda s, m: json.dumps({"actions": [action]})


def test_autonomy_triggers_caps_and_validation(tmp_path):
    w = world(1500, ("USA", "IRN"))
    client = LLMClient(ScriptedProvider(scripted_figure()), Recordings(tmp_path / "r"), Budget())
    log = EventLog()
    gov = next(e for e in w.notables if e["key"] == "USA.central_bank_governor")
    w.day = 42  # the governor's scheduled meeting day (every 42 days)
    chosen = autonomy.triggers(w)
    assert any(c["agent"] == gov["agent_id"] for c in chosen)
    # caps: never more than 2 turns per figure and 20 in total in any 7 days, however many triggers fire
    for d in range(42, 70):
        w.day = d
        apply(
            w,
            MacroPath(
                variable="fx_local_per_usd",
                country="IRN",
                steps=[{"start_day_offset": 0, "days": 1, "pct_per_day": 5.0}],
            ),
            pid=d,
        )
        apply(w, MacroPath(variable="gold", steps=[{"start_day_offset": 0, "days": 1, "pct_per_day": 3.0}]), pid=d)
        run_in_memory(w, 1)
        autonomy.run_turns(client, w, log)
    turns = w.macro["autonomy"]["turns"]
    for d in range(42, 70):
        window = [t for t in turns if d - 7 < t["day"] <= d]
        assert len(window) <= autonomy.TURNS_PER_WEEK
        for col in {t["col"] for t in window}:
            assert sum(t["col"] == col for t in window) <= autonomy.TURNS_PER_NOTABLE_PER_WEEK
    assert turns, "market news should give economic figures turns"
    t = log.table().to_pydict()
    assert int(EventKind.NOTABLE_ACTION) in t["kind"]


def test_autonomy_rejects_powers_the_role_lacks_and_skips_controlled_figures(tmp_path):
    w = world(1200, ("USA",))
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    client = LLMClient(
        ScriptedProvider(scripted_figure("set_interest_rate", params={"change_bp": 100})),
        Recordings(tmp_path / "r"),
        Budget(),
    )
    log = EventLog()
    cand = {
        "col": w.notables.index(pres),
        "agent": int(pres["agent_id"]),
        "salience": 1.0,
        "reasons": [(1.0, "k", "test")],
    }
    assert autonomy.decide(client, w, cand, log) == []
    rej = [
        d
        for k, d in zip(log.table().to_pydict()["kind"], log.table().to_pydict()["detail"], strict=True)
        if k == int(EventKind.ACTION_REJECTED)
    ]
    assert rej and "cannot set_interest_rate" in rej[0]
    apply(w, Control(agent=int(pres["agent_id"]), session=1))
    w.day = 42
    assert all(c["agent"] != pres["agent_id"] for c in autonomy.triggers(w))
