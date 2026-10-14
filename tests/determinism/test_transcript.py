"""A traced re-run of one day reproduces that day of the original timeline exactly."""

import pyarrow.parquet as pq

from jahan.query.transcript import render, trace_day
from jahan.store.eventlog import content_hash
from jahan.store.rundir import RunDir
from tests.conftest import SCENARIO, cli


def test_trace_reproduces_the_day_and_renders(tmp_path):
    run = tmp_path / "t"
    cli("gen", "--run", str(run), "--agents", "800", "--seed", "21", "--start-date", "2026-10-13")
    cli("run", "--run", str(run), "--days", "32")
    cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes")
    cli("run", "--run", str(run), "--days", "10")
    rd = RunDir(run)
    day = 36  # after the periodic snapshot at day 30 and the prompt at day 32
    tables = [pq.read_table(p) for p in sorted((run / "events").glob("events_*.parquet"))]
    original = [t.filter(t["day"].to_numpy() == day) for t in tables]
    original = [t for t in original if t.num_rows]
    assert original
    # pick an agent who was aware of the product by then (likely to have conversations)
    _, rows, day_log = trace_day(rd, 5, day)
    assert content_hash(day_log.table()) == content_hash(original[0])
    world, rows, _ = trace_day(rd, 5, day)
    text = render(world, 5, rows)
    assert text.strip()
