"""`jahan watch` with questions: seeds and custom questions as prompts, previews, undo, and the Watch rows."""

import json

import pytest

from tests.conftest import ROOT, cli

EXAMPLE = ROOT / "examples" / "scenarios" / "iran_hormuz_odds.yaml"
QID = "us-strikes-iran-before-december"
SEED = "us-x-iran-ceasefire-continues-through-october-31-20260917"


def _json(out: str):
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    run = tmp_path_factory.mktemp("odds") / "w"
    cli(
        "gen",
        "--run",
        str(run),
        "--agents",
        "1500",
        "--countries",
        "USA,IRN",
        "--seed",
        "6",
        "--start-date",
        "2026-10-15",
    )
    return run


def test_seed_question_custom_question_and_undo(world):
    run = str(world)
    cat = _json(cli("watch", "catalog", "--run", run, "--json"))
    assert any(s["id"] == SEED and s["offline"] for s in cat["seeds"])
    out = cli("watch", "add", "--run", run, f"odds:seed:{SEED}")
    assert "simulated traders" in out and "market 79.5% on 2026-10-15" in out
    status = _json(cli("status", "--run", run, "--json"))
    assert status["questions"] == [SEED] and status["prompts"][-1]["source"] == "watch"
    preview = _json(cli("inject", "--run", run, "--scenario", str(EXAMPLE), "--preview", "--json"))
    assert preview["odds_effects"][0]["question_id"] == QID and preview["odds_effects"][0]["known"] is True
    cli("inject", "--run", run, "--scenario", str(EXAMPLE), "--yes")
    cli("watch", "add", "--run", run, f"odds:{QID}")
    cli("run", "--run", run, "--days", "12", "--no-autonomy")
    data = _json(cli("watch", "show", "--run", run, "--json"))
    by = {s["id"]: s for s in data["series"]}
    seed = by[f"odds:{SEED}"]
    assert seed["unit"] == "probability" and seed["since_value"] == pytest.approx(0.795, abs=1e-6)
    assert seed["question"]["real"]["price"] == 0.795 and seed["question"]["status"] == "open"
    mine = by[f"odds:{QID}"]
    assert [r["day"] for r in mine["history"]][0] == 0 and len(mine["history"]) == 12
    assert mine["now"] > mine["since_value"]  # the tanker story made a strike more likely to those who heard it
    assert mine["question"]["movers"] and mine["question"]["movers"][0]["by"] == "scripted"
    text = cli("watch", "show", "--run", run)
    assert "Will the US strike Iran again before December?" in text
    # undo the example: the custom question is gone, the seed stays
    prompt = max(p["prompt_id"] for p in _json(cli("status", "--run", run, "--json"))["prompts"])
    cli("reset", "--run", run, "--before-prompt", str(prompt))
    assert _json(cli("status", "--run", run, "--json"))["questions"] == [SEED]
    # stop watching the seed: a prompt that closes it
    cli("watch", "remove", "--run", run, f"odds:{SEED}")
    status = _json(cli("status", "--run", run, "--json"))
    assert status["prompts"][-1]["prompt"].startswith("Stop watching")


def test_refusals(world, tmp_path):
    run = str(world)
    with pytest.raises(AssertionError, match="needs --question"):
        cli("watch", "add", "--run", run, "odds:custom:my-question")
    with pytest.raises(AssertionError, match="needs the network"):  # an unknown id is read as a Polymarket market
        cli("watch", "add", "--run", run, "odds:something-else")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "prompt: x\nevents:\n- type: news\n  headline: h\n  valence: 0.2\n"
        "  bears_on: [{question: no-such-q, direction: 1}]\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="not a watched question"):
        cli("inject", "--run", run, "--scenario", str(bad), "--yes")


def test_seed_far_from_the_world_date_needs_force(tmp_path):
    run = tmp_path / "old"
    cli(
        "gen",
        "--run",
        str(run),
        "--agents",
        "400",
        "--countries",
        "USA",
        "--seed",
        "2",
        "--start-date",
        "2026-09-01",
        "--allow-later-roster",
    )
    with pytest.raises(AssertionError, match="add --force"):
        cli("watch", "add", "--run", str(run), f"odds:seed:{SEED}")
    out = cli("watch", "add", "--run", str(run), f"odds:seed:{SEED}", "--force", "--countries", "USA")
    assert "Watching" in out
