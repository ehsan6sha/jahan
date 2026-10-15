"""Watched questions: traders, seeding, stories that bear on a question, the odds metric, closing."""

import numpy as np
import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind
from jahan.scenario.inject import InjectError, apply_scenario, preview_bearings
from jahan.scenario.spec import Bearing, NewsEvent, QuestionEvent, Scenario
from jahan.sim.questions import MAX_OPEN_QUESTIONS, current_odds, trader_mask
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

QID = "us-x-iran-ceasefire-test"


def _world(n=3000, seed=5, countries=("USA", "IRN"), **cfg):
    return generate_world(RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13", **cfg))


def _question(qid=QID, **kw):
    base = dict(
        id=qid,
        question="US x Iran ceasefire continues through October 31?",
        price=0.795,
        fetch_date="2026-10-15",
        resolves_by="2026-12-31",
        countries=["USA", "IRN"],
    )
    return QuestionEvent(**{**base, **kw})


def _open(w, *extra, qid=QID, **kw):
    apply_scenario(w, Scenario(prompt="watch", events=[_question(qid, **kw), *extra]), EventLog(), 1)


def _odds(w, qid=QID):
    return current_odds(w)[qid]["odds"]


def test_traders_start_near_the_market_price():
    w = _world()
    _open(w)
    st = w.macro["odds"]
    q = st["questions"][QID]
    slot = int(w.items["slot"][q["item"]])
    tm = trader_mask(w)
    aware = w.beliefs.m["aware"][: w.pop.n, slot] == 1
    assert tm.sum() > 50 and np.array_equal(aware, tm)  # traders know it, nobody else yet
    assert abs(_odds(w) - 0.795) < 0.03
    online_adults = w.pop["internet"] & (w.ages() >= 18) & ~w.pop["is_notable"]
    share = tm.sum() / online_adults.sum()
    assert 0.06 < share < 0.15  # about 10% of internet-using adults
    assert np.isnan(w.items["truth"][q["item"]]) and w.items["sentiment_weight"][q["item"]] == 0.0


def test_quiet_world_keeps_the_odds():
    w = _world()
    _open(w)
    start = _odds(w)
    path = []
    for _ in range(60):
        run_in_memory(w, 1)
        path.append(_odds(w))
    assert max(abs(p - start) for p in path) <= 0.02
    assert all(np.isfinite(w.beliefs.m["belief"][: w.pop.n].astype(np.float64)).ravel())


@pytest.mark.parametrize("direction", [0.8, -0.8])
def test_a_story_that_bears_on_it_moves_the_odds(direction):
    w = _world()
    story = NewsEvent(
        headline="US and Iran agree on Hormuz" if direction > 0 else "US strikes Iranian navy",
        topic="world",
        valence=0.6 if direction > 0 else -0.7,
        salience=0.9,
        countries=["USA", "IRN"],
        bears_on=[Bearing(question=QID, direction=direction)],
    )
    _open(w, story)
    start = _odds(w)
    rec = run_in_memory(w, 7)
    assert (_odds(w) - start) * np.sign(direction) > 0.01
    claim = next(iter(w.macro["odds"]["questions"][QID]["claims"].values()))
    assert claim["by"] == "scripted" and claim["moved"] * direction > 0
    t = rec.metrics.table().to_pydict()
    rows = [
        (d, c, v)
        for d, c, m, v in zip(t["day"], t["country"], t["metric"], t["value"], strict=True)
        if m == f"odds.{QID}"
    ]
    assert {c for _, c, _ in rows} == {-1, 0, 1} and all(0 < v < 1 for *_, v in rows)


def test_non_traders_do_not_count():
    w = _world()
    _open(w)
    before = _odds(w)
    q = w.macro["odds"]["questions"][QID]
    slot = int(w.items["slot"][q["item"]])
    others = np.flatnonzero(~trader_mask(w) & w.pop["alive"])[:500]
    w.beliefs.m["aware"][others, slot] = 1
    w.beliefs.m["belief"][others, slot] = 0.99
    assert _odds(w) == pytest.approx(before)


def test_word_of_mouth_spreads_the_question():
    w = _world()
    _open(w)
    run_in_memory(w, 30)
    slot = int(w.items["slot"][w.macro["odds"]["questions"][QID]["item"]])
    aware = w.beliefs.m["aware"][: w.pop.n, slot] == 1
    assert (aware & ~trader_mask(w)).sum() > 0


def test_the_question_keeps_its_slot_when_the_pool_is_full():
    w = _world(n=800, countries=("USA",), max_slots=4)
    _open(w, countries=None)
    stories = [NewsEvent(headline=f"Story {i}", valence=0.1, duration_days=1) for i in range(10)]
    for k, s in enumerate(stories):
        apply_scenario(w, Scenario(prompt="s", events=[s]), EventLog(), 10 + k)
        rec = run_in_memory(w, 2)
        t = rec.events.table().to_pydict()
        item = w.macro["odds"]["questions"][QID]["item"]
        assert not [
            1 for kd, it in zip(t["kind"], t["item"], strict=True) if kd == EventKind.ITEM_EVICTED and it == item
        ]
    assert int(w.items["slot"][w.macro["odds"]["questions"][QID]["item"]]) >= 0


def test_closing_and_resolving():
    w = _world()
    _open(w, resolves_by="2026-10-17")
    rec = run_in_memory(w, 6)
    q = w.macro["odds"]["questions"][QID]
    assert q["status"] == "closed" and q["closed_reason"] == "resolve date passed" and q["final_odds"] is not None
    assert int(w.items["slot"][q["item"]]) == -1
    t = rec.metrics.table().to_pydict()
    days = [d for d, m in zip(t["day"], t["metric"], strict=True) if m == f"odds.{QID}"]
    assert max(days) == 4  # the 17th is day 4; from the 18th on the odds are frozen
    # watch it again, then remove it by hand
    _open(w)
    apply_scenario(w, Scenario(prompt="stop", events=[QuestionEvent(id=QID, action="close")]), EventLog(), 3)
    assert w.macro["odds"]["questions"][QID]["closed_reason"] == "removed by the user"


def test_refusals_before_anything_changes():
    w = _world(n=800, countries=("USA",))
    _open(w, countries=None)
    with pytest.raises(InjectError, match="already watched"):
        _open(w, countries=None)
    bad = NewsEvent(headline="x", valence=0.1, bears_on=[Bearing(question="no-such-question", direction=1)])
    items = w.items.n
    with pytest.raises(InjectError, match="not a watched question"):
        apply_scenario(w, Scenario(prompt="x", events=[bad]), EventLog(), 2)
    assert w.items.n == items  # nothing was applied
    for k in range(MAX_OPEN_QUESTIONS - 1):
        _open(w, qid=f"question-{k}", countries=None)
    with pytest.raises(InjectError, match="at most"):
        _open(w, qid="one-too-many", countries=None)


def test_a_story_scheduled_after_its_question_closed_is_skipped():
    w = _world(n=800, countries=("USA",))
    later = NewsEvent(headline="later", valence=0.2, start_day_offset=3, bears_on=[Bearing(question=QID, direction=1)])
    _open(w, later, countries=None)
    apply_scenario(w, Scenario(prompt="stop", events=[QuestionEvent(id=QID, action="close")]), EventLog(), 2)
    rec = run_in_memory(w, 4)
    t = rec.events.table().to_pydict()
    assert any("bears_on" in d and "not open" in d for d in t["detail"])


def test_preview_lists_bearings():
    w = _world(n=800, countries=("USA",))
    _open(w, countries=None)
    sc = Scenario(
        prompt="p", events=[NewsEvent(headline="h", valence=0.1, bears_on=[Bearing(question=QID, direction=-0.5)])]
    )
    rows = preview_bearings(w, sc)
    assert rows == [
        {
            "headline": "h",
            "question_id": QID,
            "question": "US x Iran ceasefire continues through October 31?",
            "direction": -0.5,
            "start_day_offset": 0,
            "known": True,
        }
    ]
