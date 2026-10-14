"""Event and metrics logs, written as Parquet chunks per contiguous day range.

Chunk boundaries align with snapshots and prompts, so truncating a timeline on
reset is file-level.  ``content_hash`` is computed over the logical content in
day order, independent of how days were split into chunks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import ArrayLike, NDArray

EVENT_SCHEMA = pa.schema(
    [
        ("day", pa.int32()),
        ("seq", pa.int32()),
        ("kind", pa.int16()),
        ("agent", pa.int64()),
        ("other", pa.int64()),
        ("item", pa.int32()),
        ("value", pa.float32()),
        ("country", pa.int16()),
        ("detail", pa.string()),
    ]
)

METRIC_SCHEMA = pa.schema(
    [
        ("day", pa.int32()),
        ("country", pa.int16()),
        ("item", pa.int32()),
        ("metric", pa.string()),
        ("value", pa.float64()),
    ]
)


def _arr(x: ArrayLike, n: int, dtype: Any) -> NDArray[Any]:
    a = np.asarray(x, dtype=dtype)
    return np.broadcast_to(a, (n,)).copy() if a.ndim == 0 else a.astype(dtype)


class EventLog:
    def __init__(self) -> None:
        self._parts: list[dict[str, NDArray[Any]]] = []
        self._details: list[list[str]] = []

    def add(
        self,
        day: int,
        kind: int,
        agent: ArrayLike,
        other: ArrayLike = -1,
        item: ArrayLike = -1,
        value: ArrayLike = 0.0,
        country: ArrayLike = -1,
        detail: list[str] | str | None = None,
    ) -> None:
        agent_a = np.atleast_1d(np.asarray(agent, dtype=np.int64))
        n = agent_a.shape[0]
        if n == 0:
            return
        self._parts.append(
            {
                "day": np.full(n, day, np.int32),
                "kind": np.full(n, kind, np.int16),
                "agent": agent_a,
                "other": _arr(other, n, np.int64),
                "item": _arr(item, n, np.int32),
                "value": _arr(value, n, np.float32),
                "country": _arr(country, n, np.int16),
            }
        )
        if detail is None:
            self._details.append([""] * n)
        elif isinstance(detail, str):
            self._details.append([detail] * n)
        else:
            self._details.append(list(detail))

    def extend_table(self, table: pa.Table) -> None:
        """Append rows from a previously written event table (keeps their order)."""
        if table.num_rows == 0:
            return
        d = table.to_pydict()
        self._parts.append(
            {
                "day": np.asarray(d["day"], np.int32),
                "kind": np.asarray(d["kind"], np.int16),
                "agent": np.asarray(d["agent"], np.int64),
                "other": np.asarray(d["other"], np.int64),
                "item": np.asarray(d["item"], np.int32),
                "value": np.asarray(d["value"], np.float32),
                "country": np.asarray(d["country"], np.int16),
            }
        )
        self._details.append(list(d["detail"]))

    def __len__(self) -> int:
        return sum(p["agent"].shape[0] for p in self._parts)

    def table(self) -> pa.Table:
        if not self._parts:
            return EVENT_SCHEMA.empty_table()
        cols = {k: np.concatenate([p[k] for p in self._parts]) for k in self._parts[0]}
        details = [d for ds in self._details for d in ds]
        # seq: order of insertion within each day (stable)
        order = np.argsort(cols["day"], kind="stable")
        cols = {k: v[order] for k, v in cols.items()}
        details = [details[i] for i in order]
        day = cols["day"]
        starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
        seq = np.arange(day.shape[0]) - np.repeat(starts, np.diff(np.r_[starts, day.shape[0]]))
        return pa.table(
            {
                "day": cols["day"],
                "seq": seq.astype(np.int32),
                "kind": cols["kind"],
                "agent": cols["agent"],
                "other": cols["other"],
                "item": cols["item"],
                "value": cols["value"],
                "country": cols["country"],
                "detail": pa.array(details, pa.string()),
            },
            schema=EVENT_SCHEMA,
        )

    def clear(self) -> None:
        self._parts.clear()
        self._details.clear()


class MetricsLog:
    def __init__(self) -> None:
        self.rows: dict[str, list[Any]] = {"day": [], "country": [], "item": [], "metric": [], "value": []}

    def add(self, day: int, country: int, item: int, metric: str, value: float) -> None:
        self.rows["day"].append(day)
        self.rows["country"].append(country)
        self.rows["item"].append(item)
        self.rows["metric"].append(metric)
        self.rows["value"].append(float(value))

    def table(self) -> pa.Table:
        return pa.table(self.rows, schema=METRIC_SCHEMA)

    def clear(self) -> None:
        for v in self.rows.values():
            v.clear()


def write_chunk(table: pa.Table, directory: Path, prefix: str, first_day: int, last_day: int) -> Path | None:
    if last_day < first_day:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{first_day:05d}_{last_day:05d}.parquet"
    pq.write_table(table, path)
    return path


def chunk_range(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    return int(parts[-2]), int(parts[-1])


def list_chunks(directory: Path, prefix: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(f"{prefix}_*.parquet"), key=chunk_range)


def read_all(directory: Path, prefix: str, schema: pa.Schema) -> pa.Table:
    chunks = list_chunks(directory, prefix)
    if not chunks:
        return schema.empty_table()
    return pa.concat_tables([pq.read_table(p) for p in chunks])


def truncate_from(directory: Path, prefix: str, day: int) -> list[Path]:
    removed = []
    for p in list_chunks(directory, prefix):
        first, last = chunk_range(p)
        if first >= day:
            p.unlink()
            removed.append(p)
        elif last >= day:
            raise RuntimeError(f"chunk {p.name} straddles reset day {day}; chunk boundaries are broken")
    return removed


def content_hash(table: pa.Table) -> str:
    """sha256 over columns in (day, seq) order, independent of chunking and file metadata."""
    h = hashlib.sha256()
    if table.num_rows:
        idx = pc.sort_indices(table, sort_keys=[("day", "ascending"), ("seq", "ascending")])
        table = table.take(idx)
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        h.update(name.encode())
        if pa.types.is_string(col.type):
            for s in col.to_pylist():
                h.update(s.encode())
                h.update(b"\0")
        else:
            h.update(np.ascontiguousarray(col.to_numpy(zero_copy_only=False)).tobytes())
    return h.hexdigest()
