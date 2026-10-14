import numpy as np

from jahan import paths
from jahan.gen.household_gen import MAX_TARGET_DISTANCE, _match_husbands, _remaining_mix
from jahan.model.country import load_country


def test_match_husbands_unique_and_close():
    spec = load_country(paths.data_dir(), "USA")
    rng = np.random.default_rng(0)
    n = 400
    age = rng.uniform(18, 80, n)
    sex = (np.arange(n) % 2).astype(np.int8)
    women = np.flatnonzero(sex == 0)[:120]
    men = np.flatnonzero(sex == 1)
    used = np.zeros(n, dtype=bool)
    wives, husbands, unmatched = _match_husbands(women, men, used, age, spec, rng)
    assert len(set(husbands)) == len(husbands)  # each man at most once
    assert set(wives) | set(unmatched) == set(women.tolist())
    assert used[husbands].all() and used.sum() == len(husbands)
    gap = age[husbands] - age[wives]
    # husbands are within the matching window of the sampled target (gap mean +- several sd)
    assert np.all(np.abs(gap - spec.household.spouse_age_gap_mean) < MAX_TARGET_DISTANCE + 5 * 4.0)
    # a second call cannot reuse matched men
    w2, h2, _ = _match_husbands(women, men, used, age, spec, rng)
    assert not set(h2) & set(husbands)


def test_remaining_mix_bounds():
    for needed, remaining in [(0, 100), (10, 100), (50, 100), (100, 100), (200, 100), (-5, 100)]:
        a, j = _remaining_mix(needed, remaining)
        assert 0 <= a <= 1 and 0 <= j <= 0.97 and a + j <= 1
    assert _remaining_mix(100, 100) == (1.0, 0.0)  # everyone alone when one household per person is needed
