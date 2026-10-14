import numpy as np
import pytest

from jahan.rng import DOMAIN_GEN, GOLDEN, base_key, derive_seed, hash64, mix_int, mix_np, normal, u01


def test_mix_matches_reference_splitmix64():
    # splitmix64 seeded with 0: first output is 0xE220A8397B1DCDAF (reference implementation by Vigna)
    assert mix_int(0 + GOLDEN) == 0xE220A8397B1DCDAF


def test_vector_mix_matches_scalar():
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2**63, size=200, dtype=np.int64).astype(np.uint64)
    assert [int(v) for v in mix_np(xs)] == [mix_int(int(v)) for v in xs]


def test_pinned_vectors():
    """Canonical encoding is part of the reproducibility contract: these must never change."""
    assert [int(x) for x in hash64(42, 1, 0, np.array([0, 1, 2, 123456789]))] == [
        18096000355998333061,
        10845148009202175074,
        15180472902410712755,
        6870632663512485783,
    ]
    assert [int(x) for x in hash64(2**63 + 5, 65535, 2**32 - 1, np.array([2**63], dtype=np.uint64), k=2**32 - 1)] == [
        6457888804371476071
    ]
    assert base_key(1, 2, 3, 4) == 16406182773182849785
    assert [int(x) for x in hash64(7, 3, 10, np.array([5]), domain=DOMAIN_GEN)] == [8618223324108254855]


def test_order_independence():
    keys = np.arange(1000)
    perm = np.random.default_rng(1).permutation(1000)
    a = u01(9, 2, 5, keys)
    b = u01(9, 2, 5, keys[perm])
    assert np.array_equal(a[perm], b)


def test_streams_days_k_differ():
    keys = np.arange(5000)
    base = u01(1, 1, 1, keys)
    for other in (u01(1, 2, 1, keys), u01(1, 1, 2, keys), u01(1, 1, 1, keys, k=1), u01(2, 1, 1, keys)):
        assert not np.array_equal(base, other)
        assert abs(np.corrcoef(base, other)[0, 1]) < 0.05


def test_uniformity():
    x = u01(123, 4, 7, np.arange(200_000))
    assert 0 <= x.min() and x.max() < 1
    counts = np.histogram(x, bins=20, range=(0, 1))[0]
    expected = x.size / 20
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 43.8  # 99.9th percentile of chi2 with 19 dof


def test_normal_moments():
    z = normal(5, 6, 0, np.arange(100_000))
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1) < 0.02


def test_negative_keys_rejected():
    with pytest.raises(ValueError):
        u01(1, 1, 1, np.array([-1]))


def test_derive_seed_distinct():
    seeds = {derive_seed(10, s) for s in range(100)}
    assert len(seeds) == 100
