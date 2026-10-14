import numpy as np

from jahan.model.enums import EmpStatus
from jahan.sim.availability import compute_masks, local_to_utc, utc_hour_to_local, window_mask


def bits(m: int) -> list[int]:
    return [h for h in range(24) if (int(m) >> h) & 1]


def test_window_mask_wraps_midnight():
    m = window_mask(np.array([23, 9, 5]), np.array([7, 17, 5]))
    assert bits(m[0]) == [0, 1, 2, 3, 4, 5, 6, 23]
    assert bits(m[1]) == list(range(9, 17))
    assert bits(m[2]) == []


def test_local_to_utc_offsets():
    one = window_mask(np.array([18]), np.array([19]))  # local 18:00
    assert bits(local_to_utc(one, np.array([-300], np.int16))[0]) == [23]  # US Eastern: 18:00 -> 23:00 UTC
    ten = window_mask(np.array([10]), np.array([11]))  # local 10:00
    assert bits(local_to_utc(ten, np.array([330], np.int16))[0]) == [4]  # India: 10:00 -> 04:30 UTC, bin 4
    assert bits(local_to_utc(ten, np.array([210], np.int16))[0]) == [6]  # Iran: 10:00 -> 06:30 UTC, bin 6
    assert bits(local_to_utc(ten, np.array([345], np.int16))[0]) == [4]  # Nepal: 10:00 -> 04:15 UTC, bin 4


def test_utc_hour_to_local():
    out = utc_hour_to_local(2, np.array([-300, 0, 480, 330], np.int16))
    assert out.tolist() == [21, 2, 10, 8]


def test_masks_weekday_vs_weekend(small_world_factory):
    w = small_world_factory(400)
    emp = w.pop["emp_status"] == EmpStatus.EMPLOYED
    w.day = 0  # 2026-10-13 is a Tuesday
    assert w.dow() == 1
    m = compute_masks(w, w.config.params)
    assert (m.work_local[emp] != 0).all()
    assert ((m.awake_local & m.work_local) == m.work_local).all()
    assert ((m.free_local & m.work_local) == 0).all()
    w.day = 4  # Saturday
    m2 = compute_masks(w, w.config.params)
    assert (m2.work_local == 0).all()
    assert (np.bitwise_count(m.awake_local) >= 14).mean() > 0.9  # everyone but toddlers awake >= 14 hours
