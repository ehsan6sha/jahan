"""Event odds without news: time passing (question kinds) and traders who talk (sim/questions.py)."""

import numpy as np
import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.questions import classify_kind, load_question_seeds
from jahan.scenario.inject import InjectError, apply_scenario
from jahan.scenario.spec import QuestionEvent, Scenario
from jahan.sim import questions
from jahan.sim.params import SimParams
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

QID = "us-x-iran-ceasefire-time"


def _world(economy=True, n=3000, **params):
    cfg = RunConfig(
        n_agents=n,
        countries=["USA", "IRN"],
        seed=5,
        start_date="2026-10-13",
        economy=economy,
        params=SimParams(**params),
    )
    return generate_world(cfg)


def _open(w, kind="holds_through", price=0.6, resolves_by="2026-12-31", kind_by="rule"):
    q = QuestionEvent(
        id=QID,
        question="US x Iran ceasefire continues through December 31?",
        price=price,
        fetch_date="2026-10-13",
        resolves_by=resolves_by,
        countries=["USA", "IRN"],
        kind=kind,
        kind_by=kind_by,
    )
    apply_scenario(w, Scenario(prompt="watch", events=[q]), EventLog(), 1)


def _odds(w):
    return questions.current_odds(w)[QID]["odds"]


def test_one_quiet_day_formula_and_martingale():
    b = np.array([0.1, 0.5, 0.9])
    left = 30
    r = left / (left + 1)
    by = 1 - (1 - b) ** r
    holds = b**r
    assert np.all(by < b) and np.all(holds > b)
    # under each person's own constant hazard, tomorrow's expected belief equals today's
    p_today = 1 - (1 - b) ** (1 / (left + 1))  # the chance the event happens today
    assert np.allclose(p_today * 1 + (1 - p_today) * by, b)


@pytest.mark.parametrize(("kind", "sign"), [("holds_through", 1), ("by_date", -1), ("other", 0)])
def test_quiet_days_move_odds_by_kind(kind, sign):
    w = _world(odds_talk_p=0.0)
    _open(w, kind=kind)
    start = _odds(w)
    path = []
    for _ in range(40):
        run_in_memory(w, 1)
        path.append(_odds(w))
    if sign == 0:
        assert max(abs(p - start) for p in path) <= 0.02
    else:
        assert (path[-1] - start) * sign > 0.03
        assert all((b - a) * sign >= -0.005 for a, b in zip([start, *path[:-1]], path, strict=True))


def test_the_deadline_never_reaches_certainty():
    w = _world(odds_talk_p=0.0)
    _open(w, kind="by_date", price=0.3, resolves_by="2026-10-20")
    run_in_memory(w, 7)  # day 7 is the resolve date
    q = w.macro["odds"]["questions"][QID]
    slot = int(w.items["slot"][q["item"]])
    b = w.beliefs.m["belief"][: w.pop.n, slot]
    aware = w.beliefs.m["aware"][: w.pop.n, slot] == 1
    assert np.all(b[aware] >= questions.P_MIN) and np.all(b[aware] <= questions.P_MAX)
    run_in_memory(w, 2)  # the resolve date passes; the question closes the next morning
    assert q["status"] == "closed" and 0 < q["final_odds"] < 0.3


def test_old_questions_have_no_time_effect():
    w = _world(odds_talk_p=0.0)
    q = QuestionEvent(
        id=QID, question="US x Iran ceasefire continues through December 31?", prior=0.6, resolves_by="2026-12-31"
    )
    apply_scenario(w, Scenario(prompt="watch", events=[q]), EventLog(), 1)
    assert "kind" not in w.macro["odds"]["questions"][QID]  # the record is exactly as before kinds existed
    assert questions.pass_time(w) == {}


def test_set_changes_the_kind_and_needs_an_open_question():
    w = _world(odds_talk_p=0.0)
    _open(w, kind="other")
    apply_scenario(
        w,
        Scenario(prompt="set", events=[QuestionEvent(id=QID, action="set", kind="by_date", kind_by="user")]),
        EventLog(),
        2,
    )
    assert w.macro["odds"]["questions"][QID]["kind"] == "by_date"
    with pytest.raises(InjectError, match="no watched question"):
        apply_scenario(
            w, Scenario(prompt="x", events=[QuestionEvent(id="no-such-q", action="set", kind="by_date")]), EventLog(), 3
        )
    with pytest.raises(InjectError, match="resolve date"):
        bad = QuestionEvent(id="undated-q", question="Will it rain?", prior=0.5, kind="by_date", kind_by="user")
        apply_scenario(w, Scenario(prompt="x", events=[bad]), EventLog(), 4)


def test_the_wording_rule_matches_the_saved_questions():
    seeds = load_question_seeds(paths.data_dir())
    assert seeds is not None
    for q in seeds.questions:
        assert classify_kind(q.question)[0] == q.kind, q.question


def test_talk_conserves_traders_log_odds_and_keeps_disagreement():
    w = _world(odds_talk_p=0.5, odds_talk_rate=0.2)
    _open(w, kind="other")
    q = w.macro["odds"]["questions"][QID]
    slot = int(w.items["slot"][q["item"]])
    n = w.pop.n
    tm = questions.trader_mask(w, [0, 1]) & (w.beliefs.m["aware"][:n, slot] == 1)

    def logits():
        b = np.clip(w.beliefs.m["belief"][:n, slot].astype(np.float64), questions.P_MIN, questions.P_MAX)
        return np.log(b / (1 - b))

    seeded = logits()[tm].std()
    from jahan.sim.availability import compute_masks
    from jahan.sim.contacts import sample_contacts

    before = logits()[tm].sum()
    contacted = sample_contacts(w, compute_masks(w, w.config.params), w.config.params)
    moved = questions.traders_talk(w, contacted, w.config.params)
    assert QID in moved
    assert logits()[tm].sum() == pytest.approx(before, abs=1e-2)  # pulls are symmetric (float32 storage)
    run_in_memory(w, 90)
    n = w.pop.n
    tm = questions.trader_mask(w, [0, 1]) & (w.beliefs.m["aware"][:n, slot] == 1)
    assert logits()[tm].std() >= 0.6 * seeded


def test_no_talk_without_everyday_activity():
    w = _world(economy=False, odds_talk_p=0.5)
    _open(w, kind="other")
    from jahan.sim.availability import compute_masks
    from jahan.sim.contacts import sample_contacts

    contacted = sample_contacts(w, compute_masks(w, w.config.params), w.config.params)
    assert questions.traders_talk(w, contacted, w.config.params) == {}


def test_parts_are_recorded():
    w = _world()
    _open(w, kind="holds_through")
    rec = run_in_memory(w, 10)
    t = rec.metrics.table().to_pydict()
    names = {m for m in t["metric"] if m.endswith(QID)}
    assert f"odds_time.{QID}" in names and f"odds.{QID}" in names
