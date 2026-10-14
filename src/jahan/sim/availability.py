"""Hourly availability masks (24-bit) per agent per day.

Masks are built in local time, then rotated into UTC bins.  An offset of
+5:30 maps local hour L to the UTC bin containing L-5.5, i.e. bin L-6; in
general ``utc_bin = (L - ceil(offset_min / 60)) mod 24``.  Half- and
quarter-hour zones are therefore approximated to the hour (documented).
Day of week is taken from the UTC world day for everyone (M1 simplification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EmpStatus, WorkShift
from jahan.model.world import World
from jahan.sim.params import SimParams

FULL = np.uint32((1 << 24) - 1)
U32 = NDArray[np.uint32]


def window_mask(start: NDArray[np.int64], end: NDArray[np.int64]) -> U32:
    """Bits for hours in [start, end) modulo 24 (start == end -> empty)."""
    start = np.mod(start, 24)
    length = np.mod(end - start, 24)
    out = np.zeros(start.shape[0], dtype=np.uint32)
    for h in range(24):
        inside = np.mod(h - start, 24) < length
        out |= inside.astype(np.uint32) << np.uint32(h)
    return out


def utc_shift(tz_offset_min: NDArray[np.int16]) -> NDArray[np.int64]:
    off = tz_offset_min.astype(np.int64)
    return np.mod(-((-off) // 60), 24)  # ceil(off / 60) mod 24


def local_to_utc(mask: U32, tz_offset_min: NDArray[np.int16]) -> U32:
    s = utc_shift(tz_offset_min).astype(np.uint32)
    m = mask.astype(np.uint32)
    rot = (m >> s) | (m << (np.uint32(24) - s))
    return (rot & FULL).astype(np.uint32)


def utc_hour_to_local(utc_hour: int, tz_offset_min: NDArray[np.int16]) -> NDArray[np.int64]:
    return np.mod(utc_hour + utc_shift(tz_offset_min), 24)


@dataclass
class Masks:
    awake_local: U32
    work_local: U32
    free_local: U32
    awake_utc: U32
    work_utc: U32
    free_utc: U32


def compute_masks(world: World, params: SimParams) -> Masks:
    pop = world.pop
    n = pop.n
    age = world.ages()
    shift = pop["work_shift"]
    emp = pop["emp_status"]
    c = np.clip(np.round(pop["chronotype"]), -2, 2).astype(np.int64)

    s_start = np.full(n, params.sleep_start_local, np.int64) + c
    s_end = np.full(n, params.sleep_end_local, np.int64) + c
    night = shift == WorkShift.NIGHT
    evening = shift == WorkShift.EVENING
    s_start[night], s_end[night] = 8, 16
    s_start[evening], s_end[evening] = 2, 10
    small = age < 5
    s_start[small], s_end[small] = 20, 7
    sleep = window_mask(s_start, s_end)
    awake = (~sleep) & FULL

    weekday = world.dow() < 5
    work = np.zeros(n, dtype=np.uint32)
    if weekday:
        employed = emp == EmpStatus.EMPLOYED
        w_start = np.where(night, 23, np.where(evening, 15, 9)).astype(np.int64)
        w_end = np.where(night, 7, np.where(evening, 23, 17)).astype(np.int64)
        wm = window_mask(w_start, w_end)
        work = np.where(employed, wm, work)
        school = ((emp == EmpStatus.STUDENT) | (emp == EmpStatus.CHILD)) & (age >= 5)
        sm = window_mask(np.full(n, 8, np.int64), np.full(n, 15, np.int64))
        work = np.where(school & ~employed, sm, work).astype(np.uint32)
    work[pop["trip_end_day"] >= 0] = 0  # travellers are off work and school
    work &= awake
    free = awake & (~work) & FULL
    tz = pop["tz_offset_min"]
    return Masks(
        awake_local=awake,
        work_local=work,
        free_local=free,
        awake_utc=local_to_utc(awake, tz),
        work_utc=local_to_utc(work, tz),
        free_utc=local_to_utc(free, tz),
    )


def popcount(x: NDArray[np.uint32]) -> NDArray[np.uint8]:
    return np.bitwise_count(x)
