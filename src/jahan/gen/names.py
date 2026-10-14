"""Per-country weighted name tables.  Names are deliberately not unique.

A names file has national default lists and optional groups selected by ethnicity and/or
religion (first matching group wins), plus naming customs::

    female: {Name: weight, ...}
    male: {...}
    surnames: {...}
    groups:
      - {ethnicities: [hausa, fulani], religions: [], female: {...}, male: {...}, surnames: {...}}
    family_name_first: false   # display order (e.g. China: family name first)
    mononym_share: 0.0         # share of people without a family name (e.g. Indonesia)
    surname_adoption: 0.7      # probability a wife takes her husband's family name on marriage
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass
class NameGroup:
    ethnicities: frozenset[str]
    religions: frozenset[str]
    f_idx: NDArray[np.int32]
    f_w: NDArray[np.float64]
    m_idx: NDArray[np.int32]
    m_w: NDArray[np.float64]
    s_idx: NDArray[np.int32]
    s_w: NDArray[np.float64]
    no_surnames: bool = False  # group explicitly has no family names (e.g. Balinese birth-order names)

    def matches(self, eth: str, rel: str) -> bool:
        return (not self.ethnicities or eth in self.ethnicities) and (not self.religions or rel in self.religions)


@dataclass
class NameTable:
    first: list[str]
    surnames: list[str]
    groups: list[NameGroup]  # the last group is the national default (matches everyone)
    family_name_first: bool = False
    mononym_share: float = 0.0
    surname_adoption: float = 0.7
    _cache: dict[tuple[str, str], int] = field(default_factory=dict)

    def first_name(self, idx: int) -> str:
        return self.first[idx] if 0 <= idx < len(self.first) else "?"

    def surname(self, idx: int) -> str:
        return self.surnames[idx] if 0 <= idx < len(self.surnames) else ""

    def display(self, first_idx: int, last_idx: int) -> str:
        f, s = self.first_name(first_idx), self.surname(last_idx)
        if not s:
            return f
        return f"{s} {f}" if self.family_name_first else f"{f} {s}"

    def group_of(self, eth: str, rel: str) -> int:
        key = (eth, rel)
        if key not in self._cache:
            self._cache[key] = next(i for i, g in enumerate(self.groups) if g.matches(eth, rel))
        return self._cache[key]

    def group_index(self, eth_codes: Sequence[str], rel_codes: Sequence[str]) -> NDArray[np.int64]:
        return np.array([self.group_of(e, r) for e, r in zip(eth_codes, rel_codes, strict=True)], dtype=np.int64)


def _norm(w: list[float]) -> NDArray[np.float64]:
    a = np.asarray(w, dtype=np.float64)
    if a.size == 0 or a.sum() <= 0:
        raise ValueError("empty name weights")
    return a / a.sum()


def load_names(data_dir: Path, rel_path: str) -> NameTable:
    with (data_dir / rel_path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    first: list[str] = []
    surnames: list[str] = []

    def add(names: dict, pool: list[str]) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        idx = []
        for n in names:
            if n not in pool:
                pool.append(n)
            idx.append(pool.index(n))
        return np.asarray(idx, dtype=np.int32), _norm(list(names.values()))

    def group(spec: dict, eths: list[str], rels: list[str]) -> NameGroup:
        f_idx, f_w = add(spec["female"], first)
        m_idx, m_w = add(spec["male"], first)
        s_idx, s_w = add(spec["surnames"], surnames) if spec.get("surnames") else (np.zeros(0, np.int32), np.zeros(0))
        no_surnames = "surnames" in spec and not spec["surnames"]
        return NameGroup(frozenset(eths), frozenset(rels), f_idx, f_w, m_idx, m_w, s_idx, s_w, no_surnames)

    groups = [group(g, g.get("ethnicities", []), g.get("religions", [])) for g in raw.get("groups", [])]
    groups.append(group(raw, [], []))
    return NameTable(
        first=first,
        surnames=surnames,
        groups=groups,
        family_name_first=bool(raw.get("family_name_first", False)),
        mononym_share=float(raw.get("mononym_share", 0.0)),
        surname_adoption=float(raw.get("surname_adoption", 0.7)),
    )


def sample_first_names(
    table: NameTable, sex: NDArray[np.int8], rng: np.random.Generator, groups: NDArray[np.int64] | None = None
) -> NDArray[np.int32]:
    n = sex.shape[0]
    groups = np.full(n, len(table.groups) - 1, np.int64) if groups is None else groups
    out = np.empty(n, dtype=np.int32)
    for gi in np.unique(groups):
        g = table.groups[int(gi)]
        for s, idx, w in ((0, g.f_idx, g.f_w), (1, g.m_idx, g.m_w)):
            m = (groups == gi) & (sex == s)
            if m.any():
                out[m] = idx[rng.choice(idx.shape[0], size=int(m.sum()), p=w)]
    return out


def sample_surnames(
    table: NameTable, n: int, rng: np.random.Generator, groups: NDArray[np.int64] | None = None
) -> NDArray[np.int32]:
    groups = np.full(n, len(table.groups) - 1, np.int64) if groups is None else groups
    out = np.full(n, -1, dtype=np.int32)
    for gi in np.unique(groups):
        g = table.groups[int(gi)]
        if g.no_surnames:
            continue
        if g.s_idx.size == 0:
            g = table.groups[-1]
        m = groups == gi
        if m.any() and g.s_idx.size:
            out[m] = g.s_idx[rng.choice(g.s_idx.shape[0], size=int(m.sum()), p=g.s_w)]
    if table.mononym_share > 0:
        out[rng.random(n) < table.mononym_share] = -1
    return out
