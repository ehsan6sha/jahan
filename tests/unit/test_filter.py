import numpy as np
import pytest

from jahan.query.scope import FilterError, evaluate
from jahan.query.views import agent_card


def test_filters(small_world_factory):
    w = small_world_factory(500)
    age = w.ages()
    assert np.array_equal(evaluate(w, "age > 30"), age > 30)
    m = evaluate(w, "country == 'USA' and age >= 18 and info_tier >= 2")
    assert np.array_equal(m, (age >= 18) & (w.pop["info_tier"] >= 2))
    assert evaluate(w, "sex in ('F',)").sum() == (w.pop["sex"] == 0).sum()
    assert evaluate(w, "not urban").sum() == (w.pop["region"] == 0).sum()


@pytest.mark.parametrize("expr", ["__import__('os')", "age.__class__", "open('x')", "nosuchfield > 1", "lambda: 1"])
def test_unsafe_or_unknown_rejected(small_world_factory, expr):
    w = small_world_factory(200)
    with pytest.raises(FilterError):
        evaluate(w, expr)


def test_agent_card_renders(small_world_factory):
    w = small_world_factory(300)
    text = agent_card(w, 10)
    assert "agent 10" in text and "Circle" in text and "Today (local time)" in text
