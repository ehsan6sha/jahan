"""Run a world forward inside a run directory, with chunked logs and periodic snapshots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jahan.model.world import World
from jahan.sim.tick import run_day
from jahan.store.eventlog import EventLog, MetricsLog, write_chunk
from jahan.store.rundir import RunDir
from jahan.store.snapshot import save_snapshot


class Recorder:
    """Buffers one contiguous chunk of events and metrics."""

    def __init__(self, start_day: int) -> None:
        self.events = EventLog()
        self.metrics = MetricsLog()
        self.start_day = start_day

    def flush(self, events_dir: Path, metrics_dir: Path, end_day_exclusive: int) -> None:
        last = end_day_exclusive - 1
        if last >= self.start_day:
            write_chunk(self.events.table(), events_dir, "events", self.start_day, last)
            write_chunk(self.metrics.table(), metrics_dir, "metrics", self.start_day, last)
        self.events.clear()
        self.metrics.clear()
        self.start_day = end_day_exclusive


def run(
    world: World,
    rundir: RunDir,
    days: int,
    progress: Callable[[int, dict[str, int]], None] | None = None,
    recorder: Recorder | None = None,
    before_day: Callable[[World, EventLog], None] | None = None,
) -> Recorder:
    """Advance ``days`` days.  Flushes a chunk + periodic snapshot every ``snapshot_every`` days,
    flushes the remainder at the end, and writes the head snapshot.

    ``before_day(world, log)`` runs at the start of each day, before the tick (public-figure autonomy).
    Periodic snapshots are taken before it, so re-running from a snapshot calls it again for that day.
    """
    rec = recorder or Recorder(world.day)
    every = world.config.snapshot_every
    for _ in range(days):
        if before_day is not None:
            before_day(world, rec.events)
        stats = run_day(world, rec.events, rec.metrics)
        if progress:
            progress(world.day, stats)
        if every > 0 and world.day % every == 0:
            rec.flush(rundir.events, rundir.metrics, world.day)
            save_snapshot(world, rundir.periodic(world.day), label=f"day {world.day}")
    rec.flush(rundir.events, rundir.metrics, world.day)
    save_snapshot(world, rundir.head, label="head")
    return rec


def run_in_memory(world: World, days: int) -> Recorder:
    """Advance without touching disk (experiments, tests)."""
    rec = Recorder(world.day)
    for _ in range(days):
        run_day(world, rec.events, rec.metrics)
    return rec
