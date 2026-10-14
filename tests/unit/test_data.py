import pytest

from jahan import paths
from jahan.gen.allocate import allocate
from jahan.gen.registry import load_registry, validate_data
from jahan.model.country import DataError, load_country, normalise


def test_normalise_percent_and_fraction():
    assert normalise({"a": 60, "b": 40}, "t") == {"a": 0.6, "b": 0.4}
    assert normalise({"a": 0.6, "b": 0.4}, "t") == {"a": 0.6, "b": 0.4}
    assert abs(sum(normalise({"a": 50.5, "b": 50.0}, "t").values()) - 1) < 1e-12


def test_normalise_rejects_bad_sums():
    with pytest.raises(DataError):
        normalise({"a": 60, "b": 60}, "t")
    with pytest.raises(DataError):
        normalise({"a": 0.3, "b": 0.3}, "t")


def test_usa_loads_and_registry_is_consistent():
    spec = load_country(paths.data_dir(), "USA")
    assert spec.iso3 == "USA"
    reg = load_registry(paths.data_dir())
    for sid in spec.source_ids():
        assert sid in reg
    assert validate_data(paths.data_dir(), ["USA"]) == []


def test_strict_mode_flags_provisional_data():
    problems = validate_data(paths.data_dir(), ["USA"], strict=True)
    assert problems and all("strict mode" in p for p in problems)


def test_allocation():
    assert sum(allocate(1000, [1, 4, 9], "sqrt")) == 1000
    assert allocate(10, [1, 1], "equal") == [5, 5]
    assert allocate(1000, [100, 300], "proportional") == [250, 750]
