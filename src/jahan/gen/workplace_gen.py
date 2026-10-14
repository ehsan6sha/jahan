"""Workplaces (work teams) and schools (classes).

A simulated "workplace" is the group of colleagues an agent interacts with at
work, not the whole firm: each agent is a sample of many real people, so firm
size distributions do not apply directly.  Team sizes are lognormal (HEURISTIC).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EmpStatus

Arr = NDArray[Any]

TEAM_MEDIAN = 8.0
TEAM_SIGMA = 0.6
CLASS_SIZE = 25


def _group_key(region: Arr, tz: Arr) -> NDArray[np.int64]:
    return region.astype(np.int64) * 100000 + (tz.astype(np.int64) + 50000)


def assign_workplaces(
    people: dict[str, Arr], rng: np.random.Generator
) -> tuple[NDArray[np.int64], list[tuple[int, int, int]]]:
    """Return local workplace index per person (-1 none) and rows (kind, region, size)."""
    n = people["age"].shape[0]
    wp = np.full(n, -1, dtype=np.int64)
    rows: list[tuple[int, int, int]] = []
    key = _group_key(people["region"], people["tz_offset_min"])
    age = people["age"]
    emp = people["emp_status"]

    employed = emp == int(EmpStatus.EMPLOYED)
    for k in np.unique(key[employed]):
        members = np.flatnonzero(employed & (key == k))
        members = members[rng.permutation(members.shape[0])]
        i = 0
        while i < members.shape[0]:
            size = int(np.clip(round(np.exp(np.log(TEAM_MEDIAN) + TEAM_SIGMA * rng.standard_normal())), 2, 40))
            grp = members[i : i + size]
            wp[grp] = len(rows)
            rows.append((0, int(people["region"][grp[0]]), int(grp.shape[0])))
            i += size

    student = (emp == int(EmpStatus.STUDENT)) | ((emp == int(EmpStatus.CHILD)) & (age >= 5))
    band = np.select([age < 11, age < 14, age < 18], [0, 1, 2], 3)
    skey = key * 10 + band
    for k in np.unique(skey[student]):
        members = np.flatnonzero(student & (skey == k))
        members = members[rng.permutation(members.shape[0])]
        for i in range(0, members.shape[0], CLASS_SIZE):
            grp = members[i : i + CLASS_SIZE]
            wp[grp] = len(rows)
            rows.append((1, int(people["region"][grp[0]]), int(grp.shape[0])))
    return wp, rows
