"""Bit-identical replays across processes, and exact time travel."""

import re
from pathlib import Path

from tests.conftest import SCENARIO, cli

ROOT = Path(__file__).resolve().parents[2]


def _hash(run: Path) -> str:
    out = cli("events-hash", "--run", str(run)).strip().splitlines()[-1]
    assert re.fullmatch(r"[0-9a-f]{64}", out)
    return out


def _gen(run: Path, *extra: str) -> None:
    cli("gen", "--run", str(run), "--agents", "700", "--seed", "77", "--start-date", "2026-10-13", "--force", *extra)


def test_same_seed_same_log_across_processes(tmp_path):
    hashes = []
    for i, hs in enumerate(("1", "12345")):
        run = tmp_path / f"r{i}"
        _gen(run)
        cli("run", "--run", str(run), "--days", "35", env_extra={"PYTHONHASHSEED": hs})
        cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes", env_extra={"PYTHONHASHSEED": hs})
        cli("run", "--run", str(run), "--days", "25", env_extra={"PYTHONHASHSEED": hs})
        hashes.append(_hash(run))
    assert hashes[0] == hashes[1]


def test_chunking_does_not_change_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _gen(a)
    _gen(b)
    cli("run", "--run", str(a), "--days", "50")
    for _ in range(5):
        cli("run", "--run", str(b), "--days", "10")
    assert _hash(a) == _hash(b)


def test_reset_restores_the_no_prompt_timeline(tmp_path):
    with_prompt, plain = tmp_path / "p", tmp_path / "q"
    _gen(with_prompt)
    _gen(plain)
    cli("run", "--run", str(with_prompt), "--days", "30")
    cli("inject", "--run", str(with_prompt), "--scenario", str(SCENARIO), "--yes")
    cli("run", "--run", str(with_prompt), "--days", "20")
    cli("run", "--run", str(plain), "--days", "50")
    assert _hash(with_prompt) != _hash(plain)  # the prompt really changed the timeline
    cli("reset", "--run", str(with_prompt), "--before-prompt", "1")
    cli("run", "--run", str(with_prompt), "--days", "20")
    assert _hash(with_prompt) == _hash(plain)


def test_two_prompts_on_one_day_keep_both_event_sets(tmp_path):
    run = tmp_path / "two"
    _gen(run)
    cli("run", "--run", str(run), "--days", "5")
    cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes")
    cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes")
    cli("run", "--run", str(run), "--days", "2")
    import pyarrow.parquet as pq

    from jahan.model.enums import EventKind

    kinds = []
    for p in sorted((run / "events").glob("events_*.parquet")):
        kinds += pq.read_table(p).column("kind").to_pylist()
    assert kinds.count(EventKind.SCENARIO_INJECTED) == 2


def test_core_never_imports_llm_or_ingest():
    core = ["sim", "gen", "model", "store"]
    for pkg in core:
        for f in (ROOT / "src" / "jahan" / pkg).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            assert "jahan.llm" not in text and "jahan.ingest" not in text, f
