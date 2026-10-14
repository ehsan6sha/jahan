"""Days past today's real date: detection, the parser hint, and the non-interactive note."""

import datetime as dt
import json

from jahan.cli import future_window
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, ScriptedProvider
from jahan.llm.recordings import Recordings
from jahan.scenario.parser import parse_prompt
from tests.conftest import cli

D = dt.date


def test_future_window():
    today = D(2026, 10, 13)
    assert future_window(D(2026, 10, 1), 10, today) is None  # ends on Oct 10
    assert future_window(D(2026, 10, 1), 13, today) is None  # ends today
    assert future_window(D(2026, 10, 1), 14, today) == (D(2026, 10, 14), D(2026, 10, 14))
    assert future_window(D(2026, 10, 13), 30, today) == (D(2026, 10, 14), D(2026, 11, 11))
    assert future_window(D(2026, 12, 1), 5, today) == (D(2026, 12, 1), D(2026, 12, 5))  # already in the future
    assert future_window(D(2026, 12, 1), 0, today) is None


def test_feed_parser_is_told_where_real_news_ends(tmp_path):
    w = generate_world(RunConfig(n_agents=300, countries=["USA"], seed=1, start_date="2026-10-13"))
    seen: list[str] = []
    answer = {
        "scenario": {
            "prompt": "gold up",
            "events": [
                {"type": "macro", "variable": "gold", "steps": [{"start_day_offset": 3, "days": 7, "pct_per_day": 1.0}]}
            ],
        }
    }

    def responder(system, messages):
        seen.append(messages[0]["content"])
        return json.dumps(answer)

    c = LLMClient(ScriptedProvider(responder), Recordings(tmp_path / "rec"), Budget())
    res = parse_prompt(c, w, "gold +1%/day for a week", feed=True, first_future_offset=3)
    assert res.scenario is not None
    assert "first simulated day without real news is 2026-10-16 (start_day_offset 3)" in seen[0]
    parse_prompt(c, w, "gold +1%/day for two weeks", feed=True)
    assert "without real news" not in seen[1]


def test_run_past_today_notes_missing_news_without_blocking(tmp_path):
    run = tmp_path / "run"
    today = dt.datetime.now(dt.UTC).date()
    cli("gen", "--run", str(run), "--agents", "400", "--countries", "USA", "--start-date", today.isoformat())
    out = cli("run", "--run", str(run), "--days", "3")
    assert f"{today + dt.timedelta(days=1)} to {today + dt.timedelta(days=2)} are after today's real date" in out
    assert "Simulated to day 3" in out
