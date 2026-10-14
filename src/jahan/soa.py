"""Growable struct-of-arrays table.

All simulation state lives in numpy column arrays.  Rows ``[0, n)`` are valid;
capacity beyond ``n`` is preallocated headroom so births and created agents can
be appended without reallocating mid-run.  Rows are never deleted during a run
(deaths set an ``alive`` flag); ``compact`` exists for offline maintenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Col:
    dtype: str
    shape: tuple[int, ...] = ()
    default: float | int | bool | bytes = 0


ColumnSpec = Mapping[str, Col]


class SoATable:
    def __init__(self, spec: ColumnSpec, capacity: int = 16) -> None:
        self.spec: dict[str, Col] = dict(spec)
        self.n = 0
        self._cap = max(1, int(capacity))
        self._data: dict[str, NDArray[Any]] = {name: self._alloc(col, self._cap) for name, col in self.spec.items()}

    @staticmethod
    def _alloc(col: Col, cap: int) -> NDArray[Any]:
        return np.full((cap, *col.shape), col.default, dtype=col.dtype)

    @property
    def capacity(self) -> int:
        return self._cap

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, name: str) -> NDArray[Any]:
        """View of the valid rows of a column (writes go through)."""
        return self._data[name][: self.n]

    def __setitem__(self, name: str, value: Any) -> None:
        self._data[name][: self.n] = value

    def columns(self) -> list[str]:
        return list(self.spec)

    def reserve(self, capacity: int) -> None:
        if capacity <= self._cap:
            return
        for name, col in self.spec.items():
            new = self._alloc(col, capacity)
            new[: self.n] = self._data[name][: self.n]
            self._data[name] = new
        self._cap = capacity

    def append(self, count: int, **values: Any) -> slice:
        """Append ``count`` rows; unspecified columns take their default."""
        if count <= 0:
            return slice(self.n, self.n)
        need = self.n + count
        if need > self._cap:
            self.reserve(max(need, self._cap * 2))
        sl = slice(self.n, need)
        unknown = set(values) - set(self.spec)
        if unknown:
            raise KeyError(f"unknown columns: {sorted(unknown)}")
        for name, col in self.spec.items():
            if name in values:
                self._data[name][sl] = values[name]
            else:
                self._data[name][sl] = col.default
        self.n = need
        return sl

    def compact(self, keep: NDArray[np.bool_]) -> None:
        if keep.shape != (self.n,):
            raise ValueError("mask must cover exactly the valid rows")
        m = int(keep.sum())
        for name in self.spec:
            arr = self._data[name]
            arr[:m] = arr[: self.n][keep]
        self.n = m

    def to_arrays(self, prefix: str) -> dict[str, NDArray[Any]]:
        return {f"{prefix}{name}": np.ascontiguousarray(self._data[name][: self.n]) for name in self.spec}

    @classmethod
    def from_arrays(
        cls, spec: ColumnSpec, arrays: Mapping[str, NDArray[Any]], prefix: str, headroom: float = 1.5
    ) -> SoATable:
        names = list(spec)
        n = int(arrays[f"{prefix}{names[0]}"].shape[0]) if names else 0
        t = cls(spec, capacity=max(16, int(n * headroom)))
        t.n = n
        for name in names:
            key = f"{prefix}{name}"
            if key in arrays:
                t._data[name][:n] = arrays[key]
        return t
