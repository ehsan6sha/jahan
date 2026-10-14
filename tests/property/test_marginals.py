"""Generated populations reproduce the registry-backed distributions within sampling tolerance.

These tests assert fidelity to the data files, not to reality: realism depends on the
confidence of each registry entry.  Tolerance per cell is 4 standard errors + 1 pp.
"""

import numpy as np
import pytest

from jahan.config import RunConfig
from jahan.gen.population_gen import EMP_ORDER, internet_probability, match_rows
from jahan.gen.world_gen import generate_world
from jahan.model.enums import Marital

N = 10_000
COUNTRIES = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]


def tol(p: float, n: int) -> float:
    return 4 * np.sqrt(max(p * (1 - p), 1e-4) / max(n, 1)) + 0.01


@pytest.fixture(scope="module", params=[(c, s) for c in COUNTRIES for s in (11, 22)], ids=lambda p: f"{p[0]}-{p[1]}")
def world(request):
    iso, seed = request.param
    return generate_world(RunConfig(n_agents=N, countries=[iso], seed=seed, start_date="2026-10-13", notables=False))


def test_age_bands(world):
    spec = world.countries[0]
    age = world.ages()
    edges = np.asarray(spec.age_sex.bin_edges, dtype=float)
    observed = np.histogram(age, bins=edges)[0] / age.size
    expected = np.asarray(spec.age_sex.total_pct)
    assert np.all(np.abs(observed - expected) < [tol(p, N) for p in expected])


def test_sex(world):
    spec = world.countries[0]
    expected_male = float(np.dot(spec.age_sex.total_pct, spec.age_sex.male_share))
    assert abs(world.pop["sex"].mean() - expected_male) < tol(expected_male, N)


def test_education_rows(world):
    spec = world.countries[0]
    age = world.ages()
    rows = spec.education.rows
    row_of = match_rows(rows, age, world.pop["sex"])
    edu = world.pop["education"]
    for i, r in enumerate(rows):
        m = row_of == i
        n = int(m.sum())
        if n < 300:
            continue
        for level, share in r.shares.items():
            obs = (edu[m] == int(level)).mean()
            assert abs(obs - share) < tol(share, n), (r.age_min, r.sex, level, obs, share)


def test_employment_rows(world):
    spec = world.countries[0]
    age = world.ages()
    rows = spec.employment.rows
    row_of = match_rows(rows, age, world.pop["sex"])
    emp = world.pop["emp_status"]
    for i, r in enumerate(rows):
        m = row_of == i
        n = int(m.sum())
        if n < 300:
            continue
        for status, share in r.shares.items():
            obs = (emp[m] == EMP_ORDER.index(status)).mean()
            assert abs(obs - share) < tol(share, n), (r.age_min, r.sex, status, obs, share)


def test_first_language(world):
    spec = world.countries[0]
    l1 = spec.languages.l1
    code = spec.languages_official[0]
    expected = sum(
        share * l1.by_ethnicity.get(e, l1.default).get(code, 0.0) for e, share in spec.ethnicity.shares.items()
    )
    observed = (world.pop["lang"][:, 0] == world.lang_id(code)).mean()
    assert abs(observed - expected) < 0.03


def test_households(world):
    spec = world.countries[0]
    hh = world.pop["household_id"]
    sizes = np.bincount(hh[hh >= 0])  # public figures live outside households
    sizes = sizes[sizes > 0]
    assert abs(sizes.mean() - spec.household.mean_size()) < 0.3
    age = world.ages()
    adult = age >= 18
    married = world.pop["marital"] == Marital.MARRIED
    women = world.pop["sex"] == 0
    # the table drives women's marriage; up to age 45 the supply of men is ample, so it must match
    for r in spec.household.married_share_by_age:
        if r.age_min >= 45:
            continue
        m = women & (age >= r.age_min) & (age < r.age_max)
        if m.sum() >= 300:
            assert abs(married[m].mean() - r.share) < tol(r.share, int(m.sum())) + 0.01, (r.age_min, married[m].mean())
    # overall: without polygyny, large spousal age gaps leave older women unmatched (documented limitation)
    expected = (
        sum(
            r.share * ((age >= r.age_min) & (age < r.age_max) & adult).sum()
            for r in spec.household.married_share_by_age
        )
        / adult.sum()
    )
    allowance = 0.03 + 0.01 * max(0.0, spec.household.spouse_age_gap_mean - 3.0)
    assert abs(married[adult].mean() - expected) < allowance


def test_internet_matches_world_bank_total(world):
    spec = world.countries[0]
    observed = world.pop["internet"].mean()
    expected = float(internet_probability(spec, world.ages()).mean())
    assert abs(observed - expected) < tol(expected, N)
    if spec.internet_target is not None:
        assert abs(expected - spec.internet_target) < 0.01


def test_spouses_and_parents_consistent(world):
    pop = world.pop
    sp = pop["spouse_id"]
    married = sp >= 0
    assert np.array_equal(sp[sp[married]], np.flatnonzero(married))
    assert (pop["household_id"][married] == pop["household_id"][sp[married]]).all()
    assert (pop["sex"][married] != pop["sex"][sp[married]]).all()
    kids = pop["mother_id"] >= 0
    mom = pop["mother_id"][kids]
    age = world.ages()
    gap = age[mom] - age[kids]
    mab = world.countries[0].household.mother_age_at_birth
    assert (gap >= mab.min - 1e-6).all() and (gap <= mab.max + 1e-6).all()
    assert (pop["sex"][mom] == 0).all()
