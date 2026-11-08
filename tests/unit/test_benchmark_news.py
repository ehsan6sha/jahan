"""B9 and B10 helpers on canned data (no Claude, no network)."""

from __future__ import annotations

import datetime as dt

import pytest

from jahan import paths
from jahan.llm.provider import make_client
from jahan.scenario import benchmark_news as bn
from jahan.scenario.parser import parse_prompt


def test_news_days_follow_the_registered_split():
    days = bn.load_news_days()
    assert len(days) == 90
    by = {s: sum(1 for d in days if d.split == s) for s in ("fit", "seen", "holdout", "outcome")}
    assert by == {"fit": 64, "seen": 11, "holdout": 8, "outcome": 7}
    assert all(d.day.year <= 2025 for d in days if d.split == "fit")
    assert all(d.day >= dt.date(2026, 7, 1) for d in days if d.split == "holdout")
    assert all(d.headlines and d.real for d in days)
    nov5 = next(d for d in days if d.day == dt.date(2026, 11, 5))
    assert nov5.split == "outcome" and "Thursday, November 5, 2026" in bn.news_prompt(nov5)


def test_b11_decision_is_the_default_and_the_earlier_fit_is_kept():
    from jahan.model.investors import load_investors

    oi = load_investors(paths.data_dir()).other_investors
    assert oi.s == 0.0  # benchmarks.yaml B11: s = 0 beat s = 1.43 on the 57 clean days
    assert oi.fitted_v4["s"] == 1.43 and oi.b11_result["clean_days"] == 57


def test_rating_notes_only_add_to_the_world_facts(tmp_path):
    seen = []

    def answer(_system, messages):
        seen.append(messages[0]["content"])
        return '{"scenario": {"prompt": "p", "events": []}}'

    client = make_client(tmp_path, scripted=answer)
    world = bn._parse_world(dt.date(2026, 9, 1))
    parse_prompt(client, world, "same prompt")
    parse_prompt(client, world, "same prompt", purpose_tag="b11", rating_notes="NOTES HERE")
    assert "NOTES HERE" not in seen[0]
    assert seen[1] == seen[0].replace("\n\nScenario JSON schema:", "\nNOTES HERE\n\nScenario JSON schema:", 1)


def test_parsed_news_lands_on_the_trading_day_before_the_close():
    rating = {
        "scenario": {
            "prompt": "p",
            "events": [
                {"type": "news", "headline": "a", "valence": -0.5, "market_valence": -0.6, "start_day_offset": 2},
                {"type": "news", "headline": "b", "valence": 0.1, "release_utc_hour": 22},
                {"type": "macro", "variable": "gold", "steps": [{"pct_per_day": 1.0}]},
            ],
        }
    }
    evs, notes = bn.news_events(rating)
    assert [e.headline for e in evs] == ["a", "b"]
    assert evs[0].start_day_offset == 0 and evs[1].release_utc_hour == bn.LAST_RELEASE_HOUR
    assert notes == {"dropped": 1, "moved_hour": 1, "moved_day": 1}
    assert bn.news_events({"scenario": None})[0] == []


def _row(real, people=0.0, others=-1.0, tech=None):
    d = bn.NewsDay("x", dt.date(2024, 1, 2), "fit", ["news"], ["h"], [real, 0.1, -0.2, 0.0, 0.3], tech)
    return bn.DayBasis(d, people, others, 0.0)


def test_s_is_chosen_on_the_grid_with_ties_to_the_smallest():
    rows = [_row(-3.0), _row(-3.0), _row(0.0, others=0.0)]
    s, losses = bn.fit_s(rows)
    assert s == pytest.approx(3.0) and len(losses) == len(bn.S_GRID)
    lo, hi = bn.bootstrap_s(rows)
    assert lo <= 3.0 <= hi
    flat = [_row(0.0, others=0.0)]
    assert bn.fit_s(flat)[0] == 0.0  # every s ties: the smallest


def test_scores_count_direction_and_size_on_big_days():
    rows = [_row(-3.0), _row(2.5, others=1.0), _row(-0.4, others=0.0, tech=[-0.5])]
    sc = bn.b9_scores(rows, 2.0, 1.3)
    assert sc["days"] == 3 and sc["big_days"] == 2
    assert sc["big_direction_right"] == 2 and sc["big_within_factor_2"] == 2
    assert sc["day0"] < sc["day0_s0"] and "tech_day0" in sc


def test_retest_agreement():
    def r(mv):
        return {
            "scenario": {
                "prompt": "",
                "events": [{"type": "news", "headline": "h", "valence": 0, "market_valence": mv}],
            }
        }

    ratings = {str(i): {"first": r(x), "retest": r(x + 0.05)} for i, x in enumerate([-0.8, -0.2, 0.3, 0.6])}
    out = bn.ratings_summary(ratings)
    assert out["days_rated_both_times"] == 4 and out["market_valence_correlation"] > 0.99
    assert out["same_sign_share"] == 1.0
