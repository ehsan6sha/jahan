import numpy as np
import pytest

from jahan.soa import Col, SoATable

SPEC = {"id": Col("i8", default=-1), "x": Col("f4", default=0.5), "v": Col("i2", (3,), default=-1)}


def test_append_defaults_and_values():
    t = SoATable(SPEC, capacity=2)
    t.append(3, id=np.arange(3), x=[1.0, 2.0, 3.0])
    assert t.n == 3 and t.capacity >= 3
    assert t["id"].tolist() == [0, 1, 2]
    assert t["v"].tolist() == [[-1, -1, -1]] * 3
    t.append(2, id=[10, 11])
    assert t["x"][3:].tolist() == [0.5, 0.5]


def test_unknown_column_rejected():
    t = SoATable(SPEC)
    with pytest.raises(KeyError):
        t.append(1, nope=1)


def test_reserve_preserves_data():
    t = SoATable(SPEC, capacity=4)
    t.append(4, id=np.arange(4))
    t.reserve(100)
    assert t.capacity == 100 and t["id"].tolist() == [0, 1, 2, 3]


def test_views_write_through():
    t = SoATable(SPEC)
    t.append(3, id=np.arange(3))
    t["x"][1] = 9.0
    assert t["x"][1] == 9.0


def test_compact_and_roundtrip():
    t = SoATable(SPEC)
    t.append(5, id=np.arange(5), x=np.arange(5, dtype=np.float32))
    t.compact(np.array([True, False, True, False, True]))
    assert t["id"].tolist() == [0, 2, 4]
    arrays = t.to_arrays("t.")
    t2 = SoATable.from_arrays(SPEC, arrays, "t.")
    assert t2.n == 3 and t2["x"].tolist() == [0.0, 2.0, 4.0] and t2.capacity >= 3
