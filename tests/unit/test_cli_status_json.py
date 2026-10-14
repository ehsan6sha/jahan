"""Machine-readable CLI output used by `jahan ui`: status, agents and circle as JSON, and prompt previews."""

import json

import pytest

from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, LLMConfig, ScriptedProvider
from jahan.llm.recordings import Recordings
from jahan.scenario.parser import parse_prompt
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot
from tests.conftest import ROOT, cli

GOLD = {
    "scenario": {
        "prompt": "gold up",
        "events": [
            {"type": "macro", "variable": "gold", "steps": [{"start_day_offset": 0, "days": 3, "pct_per_day": 1}]}
        ],
        "horizon_days": 30,
    }
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    run = tmp_path_factory.mktemp("ui") / "w"
    cli(
        "gen",
        "--run",
        str(run),
        "--agents",
        "500",
        "--countries",
        "USA,IRN",
        "--seed",
        "5",
        "--start-date",
        "2026-10-14",
    )
    return run


def _json(out: str) -> dict:
    return json.loads(out.strip().splitlines()[-1])


def test_status_json_describes_the_world(world):
    st = _json(cli("status", "--run", str(world), "--json"))
    assert st["day"] == 0 and st["date"] == "2026-10-14"
    assert [c["iso3"] for c in st["countries"]] == ["USA", "IRN"]
    assert sum(c["agents"] for c in st["countries"]) == st["agents_alive"]
    assert st["prompts"] == [] and st["session"]["current"] is None
    assert st["llm"]["ready"] is False  # tests never reach a model
    assert "employed" in st["vocab"]["status"] and len(st["vocab"]["info_tiers"]) == 4
    assert any(f["key"] == "USA.head_of_state" for f in st["public_figures"])
    assert "Prompts: 0" in cli("status", "--run", str(world))


def test_agents_json_and_filter_errors(world):
    res = _json(
        cli("agents", "--run", str(world), "--filter", "country == 'IRN' and age >= 30", "--limit", "3", "--json")
    )
    assert res["matches"] > 3 and len(res["agents"]) == 3
    a = res["agents"][0]
    assert a["country"] == "IRN" and a["age"] >= 30 and {"id", "name", "occupation", "info_tier"} <= set(a)
    with pytest.raises(AssertionError, match="unknown field"):
        cli("agents", "--run", str(world), "--filter", "shoe_size > 3")


def test_scenario_preview_injects_nothing(world):
    scen = str(ROOT / "examples" / "scenarios" / "gold_spike.yaml")
    out = cli("inject", "--run", str(world), "--scenario", scen, "--preview")
    assert "Preview only: nothing was injected" in out and "gold" in out
    assert RunDir(world).ledger() == []


def test_free_text_preview_then_confirm_reuses_the_recorded_parse(tmp_path):
    """The confirm click must inject exactly what the preview showed, without a second model call."""
    run = tmp_path / "p"
    cli("gen", "--run", str(run), "--agents", "300", "--countries", "USA", "--seed", "2", "--start-date", "2026-10-14")
    calls = []

    def responder(system, messages):
        calls.append(1)
        return json.dumps(GOLD)

    # Record the model's answer exactly as the CLI will ask for it; the CLI then runs replay-only.
    client = LLMClient(ScriptedProvider(responder), Recordings(run / "recordings"), Budget(), LLMConfig.from_env())
    assert parse_prompt(client, load_snapshot(RunDir(run).head), "gold goes up").scenario is not None
    out = cli("inject", "--run", str(run), "--prompt", "gold goes up", "--provider", "replay", "--preview")
    assert "Preview only" in out and RunDir(run).ledger() == []
    out = cli("inject", "--run", str(run), "--prompt", "gold goes up", "--provider", "replay", "--yes")
    assert "Prompt 1 injected" in out and len(calls) == 1


def test_circle_json_lists_listeners_by_trust(world):
    out = cli("agents", "--run", str(world), "--filter", "country == 'USA' and age > 30 and not notable", "--json")
    aid = _json(out)["agents"][0]["id"]
    cli("possess", "--run", str(world), "--agent", str(aid))
    res = _json(cli("circle", "--run", str(world), "--json"))
    trust = [m["trust"] for m in res["listeners"]]
    assert res["agent"] == aid and trust and trust == sorted(trust, reverse=True)
    assert {"id", "name", "tie", "trust"} <= set(res["listeners"][0])
    st = _json(cli("status", "--run", str(world), "--json"))
    me = st["session"]["agents"][0]
    assert me["id"] == aid and me["controlled_now"] and me["audiences"][0] == "circle"
