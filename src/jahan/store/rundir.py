"""Run directory layout and the prompt ledger.

runs/<name>/
  meta.json                 created, reset history
  prompts.json              ledger of injected prompts
  scenarios/<id>.yaml       scenario spec as injected
  snapshots/head.npz|json   current state
  snapshots/day_00030.*     periodic snapshots
  snapshots/pre_prompt_0001_day_00060.*
  events/events_<first>_<last>.parquet
  metrics/metrics_<first>_<last>.parquet
  experiments/<id>/seed_<s>/{baseline,treated}/...
  reports/<id>/report.md
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


class RunDir:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # paths
    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def events(self) -> Path:
        return self.root / "events"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def head(self) -> Path:
        return self.snapshots / "head.npz"

    @property
    def pending_events(self) -> Path:
        """Legacy single file of parked events (runs created before per-prompt parking)."""
        return self.root / "pending_events.parquet"

    @property
    def pending_dir(self) -> Path:
        """Events for the current day emitted outside `run` (prompts, rejections, autopilot markers).

        One file per parking, ``<seq>_<tag>.parquet``: tag is the prompt id for a prompt's own events, or the
        last prompt id at the time for loose events, so `reset --before-prompt N` drops exactly tags >= N.
        The next `run` merges them in sequence order into its first chunk.
        """
        return self.root / "pending"

    def parked_archive(self, day: int) -> Path:
        """Permanent copy of every parked file of ``day``.  A reset to a prompt on that day restores the
        files of earlier prompts, whose rows would otherwise vanish with the truncated day-``day`` chunk."""
        return self.root / "parked" / f"day_{day:05d}"

    def pending_files(self) -> list[Path]:
        files = sorted(self.pending_dir.glob("*.parquet")) if self.pending_dir.exists() else []
        return ([self.pending_events] if self.pending_events.exists() else []) + files

    def periodic(self, day: int) -> Path:
        return self.snapshots / f"day_{day:05d}.npz"

    def pre_prompt(self, prompt_id: int, day: int) -> Path:
        return self.snapshots / f"pre_prompt_{prompt_id:04d}_day_{day:05d}.npz"

    def scenario_file(self, prompt_id: int) -> Path:
        return self.root / "scenarios" / f"{prompt_id:04d}.yaml"

    def experiment_dir(self, prompt_id: int) -> Path:
        return self.root / "experiments" / f"{prompt_id:04d}"

    def report_dir(self, prompt_id: int) -> Path:
        return self.root / "reports" / f"{prompt_id:04d}"

    def exists(self) -> bool:
        return self.head.exists()

    # json helpers
    def _read(self, name: str, default: Any) -> Any:
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name: str, obj: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")

    def ledger(self) -> list[dict[str, Any]]:
        return list(self._read("prompts.json", []))

    def save_ledger(self, entries: list[dict[str, Any]]) -> None:
        self._write("prompts.json", entries)

    def meta(self) -> dict[str, Any]:
        return dict(self._read("meta.json", {}))

    def save_meta(self, meta: dict[str, Any]) -> None:
        self._write("meta.json", meta)

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / "meta.json").exists():
            self.save_meta({"created": dt.datetime.now(dt.UTC).isoformat(), "reset_history": []})
        if not (self.root / "prompts.json").exists():
            self.save_ledger([])

    def next_prompt_id(self) -> int:
        led = self.ledger()
        return (max(e["prompt_id"] for e in led) + 1) if led else 1

    def prompt(self, prompt_id: int) -> dict[str, Any]:
        for e in self.ledger():
            if e["prompt_id"] == prompt_id:
                return e
        raise KeyError(f"no prompt {prompt_id} in ledger")
