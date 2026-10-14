"""People acting through the CLI: possession, messages, posts, play, reset + replay, session experiments."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from jahan.actors.engine import circle_edges
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot
from tests.conftest import ROOT, cli, cli_env


def _gen(run: Path) -> None:
    cli("gen", "--run", str(run), "--agents", "700", "--seed", "77", "--start-date", "2026-10-13", "--force")


def _hash(run: Path) -> str:
    out = cli("events-hash", "--run", str(run)).strip().splitlines()[-1]
    assert re.fullmatch(r"[0-9a-f]{64}", out)
    return out


def _flat(e: Exception) -> str:
    """Error text without the CLI's box drawing and line wrapping."""
    return " ".join(re.sub(r"[│┌┐└┘─]", " ", str(e)).split())


def _actor_and_listener(run: Path) -> tuple[int, int]:
    w = load_snapshot(RunDir(run).head)
    for a in np.flatnonzero(w.pop["alive"] & ~w.pop["is_notable"] & (w.ages() > 25)):
        rows = circle_edges(w, int(a))
        if rows.size >= 3:
            return int(a), int(w.edges["src"][rows[0]])
    raise AssertionError("no actor")


def test_possess_say_post_brief_reset_and_replay_is_exact(tmp_path):
    run = tmp_path / "r"
    _gen(run)
    cli("run", "--run", str(run), "--days", "3")
    a, listener = _actor_and_listener(run)
    out = cli("possess", "--run", str(run), "--agent", str(a))
    assert "Session 1: you control" in out
    cli(
        "say",
        "--run",
        str(run),
        "--to",
        str(listener),
        "--topic",
        "health",
        "--stance",
        "0.5",
        "The clinic downtown is excellent",
    )
    out = cli(
        "post",
        "--run",
        str(run),
        "--audience",
        "national",
        "--topic",
        "economy",
        "--stance",
        "-0.5",
        "Prices are rising fast",
    )
    assert "your reach allows" in out  # an ordinary person cannot post nationally
    out = cli("run", "--run", str(run), "--days", "2")
    assert "on autopilot during this run" in out  # while the world runs without you, the engine drives the agent
    assert load_snapshot(RunDir(run).head).pop["controlled_by"][a] == -1
    out = cli("brief", "--run", str(run))
    assert "Your messages" in out or "Nothing new" in out
    led = json.loads((run / "prompts.json").read_text(encoding="utf-8"))
    assert [e.get("session") for e in led] == [1, 1, 1, 1]  # possess, say, post, autopilot for the run
    cli("say", "--run", str(run), "--to", str(listener), "--topic", "health", "--stance", "0.5", "Did you go?")
    assert load_snapshot(RunDir(run).head).pop["controlled_by"][a] == 1  # acting takes the agent back
    cli("run", "--run", str(run), "--days", "3")
    original = _hash(run)
    end_day = load_snapshot(RunDir(run).head).day

    out = cli("reset", "--run", str(run), "--before-prompt", "1")
    assert "6 session turns undone" in out
    assert load_snapshot(RunDir(run).head).pop["controlled_by"][a] == -1
    out = cli("session", "replay", "--run", str(run), "--session", "1", "--to-day", str(end_day))
    assert "off (as when played)" in out and "Replayed turn 6" in out
    assert _hash(run) == original  # the replayed session reproduces the identical event log

    # a reset in the middle of a day (prompt 1 stays, prompt 2 on the same day goes) keeps prompt 1's log rows
    cli("reset", "--run", str(run), "--before-prompt", "2")
    cli("session", "replay", "--run", str(run), "--session", "1", "--to-day", str(end_day))
    assert _hash(run) == original

    # a replay refuses a turn whose recorded actions file was edited
    cli("reset", "--run", str(run), "--before-prompt", "2")
    turn = next(
        t
        for t in json.loads((run / "sessions.json").read_text(encoding="utf-8"))["sessions"]["1"]["turns"]
        if t["undone"] and not t.get("replayed") and t.get("prompt_id") is not None
    )
    f = run / turn["file"]
    f.write_text(f.read_text(encoding="utf-8").replace("excellent", "terrible"), encoding="utf-8")
    try:
        cli("session", "replay", "--run", str(run), "--session", "1", "--no-autonomy")
        raise AssertionError("expected a refusal")
    except AssertionError as e:
        assert "was edited after it was recorded" in _flat(e)


def test_rejected_actions_are_logged_and_explained(tmp_path):
    run = tmp_path / "r"
    _gen(run)
    a, _ = _actor_and_listener(run)
    cli("possess", "--run", str(run), "--agent", str(a))
    w = load_snapshot(RunDir(run).head)
    stranger = int(np.flatnonzero(~np.isin(w.pop["id"], w.edges["src"][w.edges["dst"] == a]))[5])
    try:
        cli("say", "--run", str(run), "--to", str(stranger), "--topic", "world", "--stance", "0", "hi")
        raise AssertionError("expected a refusal")
    except AssertionError as e:
        assert "circle" in _flat(e)
    try:
        cli("act", "--run", str(run), "tariff", "rate_pct=20")
        raise AssertionError("expected a refusal")
    except AssertionError as e:
        assert "only public figures" in _flat(e)
    cli("run", "--run", str(run), "--days", "1")
    t = __import__("pyarrow.parquet", fromlist=["x"]).read_table(next((run / "events").glob("*.parquet"))).to_pydict()
    assert sum(1 for k in t["kind"] if k == 21) == 2  # ACTION_REJECTED, both in the day's log


def test_president_policy_through_the_cli(tmp_path):
    run = tmp_path / "r"
    _gen(run)
    out = cli("possess", "--run", str(run), "--agent", "role:USA.head_of_state")
    assert "tariff" in out
    try:
        cli("act", "--run", str(run), "set_interest_rate", "change_bp=50")
        raise AssertionError("expected a refusal")
    except AssertionError as e:
        assert "cannot set_interest_rate" in _flat(e)
    out = cli(
        "post",
        "--run",
        str(run),
        "--audience",
        "national",
        "--topic",
        "economy",
        "--stance",
        "0.5",
        "Our economy is the strongest it has ever been",
    )
    assert "audience national" in out
    cli("run", "--run", str(run), "--days", "2")
    w = load_snapshot(RunDir(run).head)
    labels = [v for v in w.macro["item_labels"].values() if "strongest" in v]
    assert labels and labels[0].startswith("[simulated] Donald Trump")


def test_scripted_play_session_and_session_experiment(tmp_path):
    run = tmp_path / "r"
    _gen(run)
    cli(
        "inject",
        "--run",
        str(run),
        "--scenario",
        str(ROOT / "examples" / "scenarios" / "launch_product_x.yaml"),
        "--yes",
    )
    a, listener = _actor_and_listener(run)
    cli("possess", "--run", str(run), "--agent", str(a))
    script = "\n".join(
        [
            "help",
            f"say {listener} The Smart Ring X is wonderful topic=tech stance=1 about='Smart Ring X'",
            "post circle Everyone should get a Smart Ring X topic=tech stance=0.5",
            "say 999999 nobody hears this topic=world stance=0",
            "next",
            "circle",
            "decide 'Smart Ring X' adopt",
            "next",
            "auto 2",
        ]
    )
    out = cli("play", "--run", str(run), "--days", "4", "--provider", "none", "--no-autonomy", stdin=script)
    assert out.count("queued") >= 3 and "no agent 999999" in out
    assert "World is at day 4" in out
    turns = json.loads((run / "sessions.json").read_text(encoding="utf-8"))["sessions"]["1"]["turns"]
    assert [t.get("autopilot") is not None for t in turns][-2:] == [True, True]
    events = [
        __import__("pyarrow.parquet", fromlist=["x"]).read_table(p).to_pydict()
        for p in sorted((run / "events").glob("*.parquet"))
    ]
    kinds = [k for t in events for k in t["kind"]]
    assert kinds.count(23) == 2  # AUTOPILOT_TURN for the two `auto` days
    assert kinds.count(18) == 1  # RELEASED once: on autopilot days the engine drives the agent, it is not frozen
    assert load_snapshot(RunDir(run).head).pop["controlled_by"][a] == -1
    assert 19 in kinds  # HUMAN_ACTION

    cli("experiment", "--run", str(run), "--session", "1", "--seeds", "2", "--workers", "1", "--horizon", "8")
    out = cli("report", "--run", str(run), "--session", "1")
    report = Path(out.strip().split("Report: ")[-1]).read_text(encoding="utf-8")
    assert "# Session report: session 1" in report and "message from" in report
    assert "not re-run inside the branches" in report


def test_example_president_session_script(tmp_path):
    """The shipped example script works as documented (even with a byte-order mark from a Windows pipe)."""
    run = tmp_path / "r"
    cli("gen", "--run", str(run), "--agents", "900", "--countries", "USA,CHN", "--start-date", "2026-10-13")
    cli("possess", "--run", str(run), "--agent", "role:USA.head_of_state")
    script = "﻿" + (ROOT / "examples" / "sessions" / "president_trade.txt").read_text(encoding="utf-8")
    out = cli("play", "--run", str(run), "--days", "7", "--provider", "none", "--no-autonomy", stdin=script)
    assert "unknown command" not in out and out.count("queued") == 3
    assert "[simulated] Donald Trump: New tariffs take effect today" in out
    w = load_snapshot(RunDir(run).head)
    assert w.day == 7 and w.macro.get("sentiment_shift") is not None


def test_session_experiment_seed0_treated_replays_the_main_timeline(tmp_path):
    import pyarrow.parquet as pq

    from jahan.store.eventlog import EVENT_SCHEMA, read_all

    run = tmp_path / "r"
    cli("gen", "--run", str(run), "--agents", "900", "--seed", "5", "--start-date", "2026-10-13")
    cli(
        "inject",
        "--run",
        str(run),
        "--scenario",
        str(ROOT / "examples" / "scenarios" / "launch_product_x.yaml"),
        "--yes",
    )
    cli("run", "--run", str(run), "--days", "2", "--no-autonomy")
    cli("possess", "--run", str(run), "--agent", "role:USA.head_of_state")
    script = (ROOT / "examples" / "sessions" / "president_trade.txt").read_text(encoding="utf-8")
    cli("play", "--run", str(run), "--days", "7", "--provider", "none", "--no-autonomy", stdin=script)
    cli("experiment", "--run", str(run), "--session", "1", "--seeds", "1", "--workers", "1", "--horizon", "7")
    cols = ["day", "kind", "agent", "other", "item", "value", "country"]

    def rows(d: dict) -> list[tuple]:
        # rejections and autopilot markers are logged outside prompts, so branches do not contain them
        return [
            tuple(d[c][i] for c in cols)
            for i in range(len(d["day"]))
            if 2 <= d["day"][i] < 9 and d["kind"][i] not in (21, 23)
        ]

    main = rows(read_all(run / "events", "events", EVENT_SCHEMA).to_pydict())
    branch = rows(
        pq.read_table(run / "experiments" / "session_0001" / "seed_00" / "treated" / "events.parquet").to_pydict()
    )
    assert main and main == branch


def test_play_times_out_to_autopilot(tmp_path):
    run = tmp_path / "r"
    _gen(run)
    a, _ = _actor_and_listener(run)
    cli("possess", "--run", str(run), "--agent", str(a))
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jahan.cli",
            "play",
            "--run",
            str(run),
            "--days",
            "2",
            "--timeout",
            "1",
            "--provider",
            "none",
            "--no-autonomy",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=cli_env(),
        cwd=ROOT,
    )
    try:
        p.wait(timeout=300)  # stdin stays open (no input, no EOF), so each turn times out
        out, err = p.stdout.read(), p.stderr.read()  # type: ignore[union-attr]
    finally:
        p.kill()
        p.stdin.close()  # type: ignore[union-attr]
    assert p.returncode == 0, err
    assert "World is at day 2" in out
    turns = json.loads((run / "sessions.json").read_text(encoding="utf-8"))["sessions"]["1"]["turns"]
    assert sum(1 for t in turns if t.get("autopilot")) == 2
