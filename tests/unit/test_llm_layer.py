import json
from pathlib import Path

import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.budget import Budget, BudgetExceeded
from jahan.llm.provider import LLMClient, LLMForbidden, ScriptedProvider, call_count, extract_json, forbid_llm
from jahan.llm.recordings import MissingRecording, Recordings
from jahan.model.notables import SIMULATED_LABEL
from jahan.query.persona import interview, mind_snapshot
from jahan.scenario.inject import apply_scenario
from jahan.scenario.parser import parse_prompt
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

EX = Path(__file__).resolve().parents[2] / "examples" / "scenarios"


def client(tmp_path, responder, replay_only=False, budget=None):
    return LLMClient(ScriptedProvider(responder), Recordings(tmp_path / "rec", replay_only), budget or Budget())


def test_recordings_replay_and_fail_closed(tmp_path):
    calls = []

    def responder(system, messages):
        calls.append(1)
        return "hello"

    c = client(tmp_path, responder)
    before = call_count()
    assert c.complete("persona", "sys", [{"role": "user", "content": "hi"}]) == "hello"
    assert c.complete("persona", "sys", [{"role": "user", "content": "hi"}]) == "hello"  # recorded
    assert len(calls) == 1 and call_count() == before + 1
    replay = client(tmp_path, responder, replay_only=True)
    assert replay.complete("persona", "sys", [{"role": "user", "content": "hi"}]) == "hello"
    with pytest.raises(MissingRecording):
        replay.complete("persona", "sys", [{"role": "user", "content": "something new"}])
    with forbid_llm(), pytest.raises(LLMForbidden):
        c.complete("persona", "sys", [{"role": "user", "content": "uncached"}])


def test_budget_caps(tmp_path):
    c = client(tmp_path, lambda s, m: "x " * 50, budget=Budget(max_input_tokens=10_000, max_output_tokens=60))
    c.complete("persona", "s", [{"role": "user", "content": "a"}], max_tokens=40)
    with pytest.raises(BudgetExceeded):
        c.complete("persona", "s", [{"role": "user", "content": "b"}], max_tokens=40)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": [1, 2, {"b": "}"}]}\n```') == {"a": [1, 2, {"b": "}"}]}


def test_parser_questions_then_scenario(tmp_path):
    w = generate_world(RunConfig(n_agents=300, countries=["USA", "IRN"], seed=1, start_date="2026-10-13"))
    good = {
        "scenario": {
            "prompt": "dollar up",
            "horizon_days": 30,
            "events": [
                {
                    "type": "macro",
                    "variable": "fx_local_per_usd",
                    "country": "IRN",
                    "steps": [{"start_day_offset": 0, "days": 5, "pct_per_day": 2.0}],
                }
            ],
        }
    }

    def responder(system, messages):
        last = messages[-1]["content"]
        if "answered your questions" in last:
            return json.dumps(good)
        if "That was invalid" in last:
            return json.dumps(good)
        return json.dumps({"clarifying_questions": ["How large is the move?"]})

    c = client(tmp_path, responder)
    first = parse_prompt(c, w, "the dollar rises in Iran")
    assert first.scenario is None and first.questions == ["How large is the move?"]
    second = parse_prompt(c, w, "the dollar rises in Iran", answers={"How large is the move?": "10% in 5 days"})
    assert second.scenario is not None and second.scenario.events[0].type == "macro"


def test_parser_retries_invalid_output_and_feed_restriction(tmp_path):
    w = generate_world(RunConfig(n_agents=300, countries=["USA"], seed=1, start_date="2026-10-13"))
    product = {
        "scenario": {
            "prompt": "x",
            "events": [
                {"type": "product_launch", "name": "Gizmo", "true_quality": 0.5, "price_usd": 10, "countries": ["USA"]}
            ],
        }
    }
    replies = iter(["not json at all", json.dumps(product)])
    c = client(tmp_path, lambda s, m: next(replies))
    res = parse_prompt(c, w, "launch a gizmo")
    assert res.scenario is not None and res.scenario.events[0].type == "product_launch"
    c2 = client(tmp_path / "b", lambda s, m: json.dumps(product))
    res2 = parse_prompt(c2, w, "launch a gizmo", feed=True)
    assert res2.scenario is None and "news and macro" in res2.error


def test_persona_grounding_and_unaware_shortcut(tmp_path):
    w = generate_world(RunConfig(n_agents=2000, countries=["USA"], seed=4, start_date="2026-10-13"))
    apply_scenario(w, Scenario.load(EX / "launch_product_x.yaml"), EventLog(), 1)
    run_in_memory(w, 25)
    slot = int(w.items["slot"][int(w.products["item_id"][0])])
    aware = w.beliefs.m["aware"][: w.pop.n, slot] == 1
    adult = w.ages() >= 18
    seen = []

    def responder(system, messages):
        seen.append(messages[-1]["content"])
        return "It sounds useful but pricey."

    c = client(tmp_path, responder)
    unaware_id = int(w.pop["id"][(~aware) & adult & ~w.pop["is_notable"]][0])
    res = interview(c, w, unaware_id, "Would you buy Smart Ring X?")
    assert not res.used_llm and "haven't heard" in res.answer and not seen
    aware_id = int(w.pop["id"][aware & adult][0])
    res = interview(c, w, aware_id, "Would you buy Smart Ring X?")
    assert res.used_llm and seen and "Smart Ring X" in seen[0]
    snap = mind_snapshot(w, unaware_id)
    assert "Smart Ring X" not in snap  # closed world: never mentions items the person did not encounter


def test_notable_interview_is_labelled_simulated(tmp_path):
    w = generate_world(RunConfig(n_agents=300, countries=["USA"], seed=2, start_date="2026-10-13"))
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    c = client(tmp_path, lambda s, m: "The figure has not addressed this publicly.")
    res = interview(c, w, pres["agent_id"], "What do you think of Smart Ring X?")
    assert res.answer.startswith(SIMULATED_LABEL)
    assert "Documented public positions" in res.snapshot
