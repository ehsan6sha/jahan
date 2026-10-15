"""Claude rates new stories against watched questions (scenario/odds_rater.py), recorded for replay."""

import json
import re

import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, ScriptedProvider
from jahan.llm.recordings import MissingRecording, Recordings
from jahan.model.enums import EventKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.odds_rater import BATCH, rate_new_items
from jahan.scenario.spec import NewsEvent, QuestionEvent, Scenario
from jahan.sim.questions import current_odds
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

QID = "us-x-iran-ceasefire-test"


def _world(stories: int = 3):
    w = generate_world(RunConfig(n_agents=1500, countries=["USA", "IRN"], seed=8, start_date="2026-10-15"))
    q = QuestionEvent(id=QID, question="US x Iran ceasefire continues through October 31?", price=0.795)
    news = [
        NewsEvent(headline=f"Story {k}: US warships fire on Iranian boats", topic="world", valence=-0.7, salience=0.9)
        for k in range(stories)
    ]
    apply_scenario(w, Scenario(prompt="w", events=[q, *news]), EventLog(), 1)
    return w


def _client(tmp_path, responder, replay_only=False):
    return LLMClient(ScriptedProvider(responder), Recordings(tmp_path / "rec", replay_only), Budget())


def _rate_all(system, messages):
    lines = re.findall(r"^(\d+)\. ", messages[-1]["content"], flags=re.M)
    return json.dumps([{"i": int(i), "direction": -0.9, "why": "fighting breaks the ceasefire"} for i in lines])


def test_ratings_become_claims_and_move_the_odds(tmp_path):
    w = _world()
    log = EventLog()
    calls = []

    def responder(system, messages):
        calls.append(messages[-1]["content"])
        assert "never estimate the probability" in system
        return _rate_all(system, messages)

    assert rate_new_items(_client(tmp_path, responder), w, log) == 3
    claims = w.macro["odds"]["questions"][QID]["claims"]
    assert len(claims) == 3 and all(c["by"] == "llm" and c["direction"] == -0.9 for c in claims.values())
    assert len(calls) == 1 and "Question: US x Iran ceasefire" in calls[0]
    kinds = log.table().column("kind").to_pylist()
    assert kinds.count(EventKind.ODDS_RATED) == 3
    assert rate_new_items(_client(tmp_path, responder), w, log) == 0  # never asked twice
    before = current_odds(w)[QID]["odds"]
    run_in_memory(w, 7)
    assert current_odds(w)[QID]["odds"] < before - 0.01


def test_batches_fallbacks_and_clipping(tmp_path):
    w = _world(stories=BATCH + 5)
    calls = []

    def responder(system, messages):
        calls.append(1)
        lines = [int(i) for i in re.findall(r"^(\d+)\. ", messages[-1]["content"], flags=re.M)]
        rows = [{"i": i, "direction": 3.0} for i in lines[:-1]]  # out of range; the last row is missing
        return json.dumps(rows)

    rate_new_items(_client(tmp_path, responder), w, EventLog())
    claims = list(w.macro["odds"]["questions"][QID]["claims"].values())
    assert len(calls) == 2 and len(claims) == BATCH + 5
    assert {c["direction"] for c in claims if c["by"] == "llm"} == {1.0}
    assert sum(c["by"] == "fallback" and c["direction"] == 0.0 for c in claims) == 2


def test_replay_reproduces_the_ratings_without_calls(tmp_path):
    first = _world()
    rate_new_items(_client(tmp_path, _rate_all), first, EventLog())
    again = _world()

    def refuse(system, messages):
        raise AssertionError("replay must not call the model")

    rate_new_items(_client(tmp_path, refuse, replay_only=True), again, EventLog())
    assert again.macro["odds"] == first.macro["odds"]
    other = _world(stories=4)  # a story the recording never saw
    with pytest.raises(MissingRecording):
        rate_new_items(_client(tmp_path, refuse, replay_only=True), other, EventLog())


def test_a_failing_provider_stops_rating_for_the_run(tmp_path):
    w = _world()
    log = EventLog()

    def broken(system, messages):
        raise RuntimeError("provider down")

    assert rate_new_items(_client(tmp_path, broken), w, log) == 0
    assert "provider down" in w.macro["odds"]["rating_stopped"]
    assert any("odds rating stopped" in d for d in log.table().column("detail").to_pylist())
