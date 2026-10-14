"""Assemble sampled individuals into households.

1. Women get a married flag from married_share_by_age; each finds the unmarried man
   whose age is closest to her age plus a sampled spousal gap, first within her
   ethnicity, then across.  Each couple forms a household.
2. Children (<18) are assigned to a mother whose age at the child's birth is
   plausible (weighted by a normal density around the mean age at birth, married
   mothers preferred).  Children inherit the mother's household, region, time
   zone, ethnicity, religion and first language.
3. Unmarried young adults stay with their mother with probability
   adult_child_at_home_by_age.
4. Remaining adults live alone, join a family household, or share with 1-2 other
   adults; the mix is solved so the household count matches population divided by
   the country's mean household size.
5. If households are still too many, young couples join a parent-aged woman's
   household (joint families), which also links the husband to his parents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.gen.population_gen import _age_share
from jahan.model.country import CountrySpec
from jahan.model.enums import Marital, RaisedBy

Arr = NDArray[Any]

MAX_CHILDREN_PER_MOTHER = 7
# relative weight of a mother who already has k children in the household (spreads children across families)
CHILD_SPREAD = np.array([1.0, 0.9, 0.5, 0.25, 0.1, 0.05, 0.02, 0.0])


@dataclass
class HouseholdResult:
    household: NDArray[np.int64]  # local household index per person
    n_households: int
    marital: NDArray[np.int8]
    spouse: NDArray[np.int64]  # local index or -1
    mother: NDArray[np.int64]
    father: NDArray[np.int64]


MAX_TARGET_DISTANCE = 10.0  # years between a wife's preferred husband age and the man found (first pass)
RELAXED_TARGET_DISTANCE = 25.0  # second pass across ethnic groups for women still unmatched


def _match_husbands(
    ws: NDArray[np.int64],
    men: NDArray[np.int64],
    used: NDArray[np.bool_],
    age: NDArray[np.float64],
    spec: CountrySpec,
    rng: np.random.Generator,
    max_distance: float = MAX_TARGET_DISTANCE,
) -> tuple[list[int], list[int], list[int]]:
    """For each woman, take the free man whose age is closest to her age plus a sampled spousal gap.

    Women are processed in random order; a 'next free' pointer structure keeps it O(n log n).
    Returns (wives, husbands, unmatched_women); marks matched men in ``used``.
    """
    free_men = men[~used[men]]
    if free_men.size == 0 or ws.size == 0:
        return [], [], [int(w) for w in ws]
    order = np.argsort(age[free_men], kind="stable")
    m_sorted = free_men[order]
    m_age = age[m_sorted]
    n = m_sorted.size
    nxt = list(range(n + 1))  # nxt[i]: smallest free index >= i (n = none)
    prv = list(range(n + 1))  # position j stands for index j - 1; root 0 means "no free index to the left"

    def find(parent: list[int], i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    wives: list[int] = []
    husbands: list[int] = []
    unmatched: list[int] = []
    gaps = rng.normal(spec.household.spouse_age_gap_mean, spec.household.spouse_age_gap_sd, ws.size)
    for w, gap in zip(rng.permutation(ws), gaps, strict=True):
        t = float(age[w]) + float(gap)
        i = int(np.searchsorted(m_age, t))
        r = find(nxt, min(i, n))
        lft = find(prv, i) - 1 if i > 0 else -1  # prv is shifted by one
        best = -1
        if r < n and (lft < 0 or abs(m_age[r] - t) <= abs(m_age[lft] - t)):
            best = r
        elif lft >= 0:
            best = lft
        if best < 0 or abs(m_age[best] - t) > max_distance:
            unmatched.append(int(w))
            continue
        man = int(m_sorted[best])
        wives.append(int(w))
        husbands.append(man)
        used[man] = True
        nxt[best] = best + 1
        prv[best + 1] = best
    return wives, husbands, unmatched


SHARED_SIZE = 2.3  # mean size of non-family shared households (HEURISTIC)


def _remaining_mix(needed: int, remaining: int) -> tuple[float, float]:
    """(p_alone, p_join) so that the unattached adults create about ``needed`` new households.

    Non-joiners create households at a rate between 1/SHARED_SIZE (all sharing) and 1 (all alone);
    when even all-sharing would create too many, a share of them joins existing family households.
    """
    if remaining <= 0:
        return 0.0, 0.0
    r = needed / remaining  # households to create per unattached adult
    share_rate = 1.0 / SHARED_SIZE
    default_rate = 0.6 + 0.4 * share_rate  # 60% alone / 40% sharing when joining is needed
    if r >= default_rate:
        p_alone = min(1.0, (r - share_rate) / (1.0 - share_rate))
        return p_alone, 0.0
    p_join = float(np.clip(1.0 - max(r, 0.0) / default_rate, 0.0, 0.97))
    return 0.6 * (1.0 - p_join), p_join


def _form_joint_families(
    people: dict[str, Arr],
    hh: NDArray[np.int64],
    spouse: NDArray[np.int64],
    mother: NDArray[np.int64],
    father: NDArray[np.int64],
    excess: int,
    rng: np.random.Generator,
) -> None:
    """Merge young couples' households into a parent-aged woman's household (patrilocal joint family).

    The husband gets that woman (and her husband) as parents, which also creates kin structure.
    """
    age = people["age"]
    sex = people["sex"]
    loc = people["region"].astype(np.int64) * 100000 + people["tz_offset_min"].astype(np.int64)
    eth = people["ethnicity"]
    husbands = np.flatnonzero((sex == 1) & (spouse >= 0) & (age >= 20) & (age < 45) & (mother < 0))
    husbands = husbands[rng.permutation(husbands.size)]
    older_women = np.flatnonzero((sex == 0) & (age >= 38))
    merged_into: dict[int, int] = {}
    for h in husbands:
        if excess <= 0:
            break
        gap = age[older_women] - age[h]
        ok = (gap >= 18) & (gap <= 45) & (loc[older_women] == loc[h]) & (eth[older_women] == eth[h])
        ok &= hh[older_women] != hh[h]
        cands = older_women[ok]
        if cands.size == 0:
            continue
        cands = cands[[merged_into.get(int(hh[c]), 0) < 1 for c in cands]]
        if cands.size == 0:
            continue
        p = int(rng.choice(cands))
        src, dst = int(hh[h]), int(hh[p])
        hh[hh == src] = dst
        merged_into[dst] = merged_into.get(dst, 0) + 1
        mother[h] = p
        if spouse[p] >= 0:
            father[h] = spouse[p]
        excess -= 1


def assemble_households(
    people: dict[str, Arr], spec: CountrySpec, rng: np.random.Generator, surname_adoption: float = 0.7
) -> HouseholdResult:
    age: NDArray[np.float64] = people["age"]
    sex = people["sex"]
    n = age.shape[0]
    hh = np.full(n, -1, dtype=np.int64)
    marital = np.zeros(n, dtype=np.int8)
    spouse = np.full(n, -1, dtype=np.int64)
    mother = np.full(n, -1, dtype=np.int64)
    father = np.full(n, -1, dtype=np.int64)
    next_hh = 0

    adult = age >= 18
    # women are flagged married from the age table; each finds a husband by preferred age gap among
    # unmarried men (same ethnicity first), so married men and women balance exactly
    women_married = adult & (sex == 0) & (rng.random(n) < _age_share(age, spec.household.married_share_by_age))
    eth = people["ethnicity"]
    men = np.flatnonzero(adult & (sex == 1))
    used = np.zeros(n, dtype=bool)
    wives_l: list[int] = []
    hus_l: list[int] = []
    leftover: list[int] = []
    for e in np.unique(eth):
        ws = np.flatnonzero(women_married & (eth == e))
        w_, h_, u_ = _match_husbands(ws, men[eth[men] == e], used, age, spec, rng)
        wives_l += w_
        hus_l += h_
        leftover += u_
    w_, h_, _ = _match_husbands(
        np.asarray(sorted(leftover), dtype=np.int64), men, used, age, spec, rng, RELAXED_TARGET_DISTANCE
    )
    wives_l += w_
    hus_l += h_
    wives = np.asarray(wives_l, dtype=np.int64)
    husbands = np.asarray(hus_l, dtype=np.int64)

    k = wives.shape[0]
    hh_ids = np.arange(next_hh, next_hh + k)
    next_hh += k
    hh[wives] = hh_ids
    hh[husbands] = hh_ids
    marital[wives] = Marital.MARRIED
    marital[husbands] = Marital.MARRIED
    spouse[wives] = husbands
    spouse[husbands] = wives
    # couple shares one region and time zone (randomly the wife's or the husband's)
    take_w = rng.random(k) < 0.5
    for col in ("region", "tz_offset_min"):
        v = np.where(take_w, people[col][wives], people[col][husbands])
        people[col][wives] = v
        people[col][husbands] = v
    # surname: wife adopts husband's with the country's customary probability (names file)
    adopt = rng.random(k) < surname_adoption
    people["last_name_id"][wives[adopt]] = people["last_name_id"][husbands[adopt]]

    # marital status of unmarried adults (HEURISTIC by age)
    um = adult & (marital != Marital.MARRIED)
    u = rng.random(n)
    a = age
    st = np.where(
        a < 30,
        Marital.SINGLE,
        np.where(
            a < 60,
            np.where(u < 0.55, Marital.SINGLE, Marital.DIVORCED),
            np.where(u < 0.45, Marital.WIDOWED, np.where(u < 0.75, Marital.DIVORCED, Marital.SINGLE)),
        ),
    )
    marital[um] = st[um].astype(np.int8)

    mab = spec.household.mother_age_at_birth
    women = np.flatnonzero((sex == 0) & (age >= mab.min))
    w_age = age[women]
    w_married = (marital[women] == Marital.MARRIED).astype(np.float64)
    kids_count = np.zeros(women.shape[0], dtype=np.int64)

    def pick_mother(child_age: float) -> int:
        at_birth = w_age - child_age
        ok = (at_birth >= mab.min) & (at_birth <= mab.max) & (kids_count < MAX_CHILDREN_PER_MOTHER)
        if not ok.any():
            return -1
        wgt = np.exp(-0.5 * ((at_birth - mab.mean) / mab.sd) ** 2) * (1.0 + 2.0 * w_married)
        wgt *= CHILD_SPREAD[np.minimum(kids_count, CHILD_SPREAD.shape[0] - 1)]
        wgt = np.where(ok, wgt, 0.0)
        if wgt.sum() <= 0:
            return -1
        j = int(rng.choice(women.shape[0], p=wgt / wgt.sum()))
        kids_count[j] += 1
        return j

    def join_mother(child: int, j: int) -> None:
        nonlocal next_hh
        m = int(women[j])
        if hh[m] < 0:
            hh[m] = next_hh
            next_hh += 1
        hh[child] = hh[m]
        mother[child] = m
        if marital[m] == Marital.MARRIED:
            father[child] = spouse[m]
        for col in ("region", "tz_offset_min", "ethnicity", "religion"):
            people[col][child] = people[col][m]
        people["lang"][child, 0] = people["lang"][m, 0]
        dup = people["lang"][child, 1:] == people["lang"][child, 0]
        people["lang"][child, 1:][dup] = -1
        f = father[child]
        people["last_name_id"][child] = people["last_name_id"][f if f >= 0 else m]

    children = np.flatnonzero(age < 18)
    children = children[np.argsort(-age[children], kind="stable")]
    orphans = []
    for c in children:
        j = pick_mother(float(age[c]))
        if j < 0:
            orphans.append(int(c))
            continue
        join_mother(int(c), j)
        people["raised_by"][c] = RaisedBy.TWO_PARENT if father[c] >= 0 else RaisedBy.SINGLE

    # adult children still at home
    at_home_p = _age_share(age, spec.household.adult_child_at_home_by_age)
    cand = np.flatnonzero(adult & (hh < 0) & (rng.random(n) < at_home_p))
    for c in cand:
        j = pick_mother(float(age[c]))
        if j >= 0:
            join_mother(int(c), j)

    # children without a plausible mother live with a random adult (raised_by OTHER)
    adults_idx = np.flatnonzero(adult)
    for oc in orphans:
        g = int(rng.choice(adults_idx))
        if hh[g] < 0:
            hh[g] = next_hh
            next_hh += 1
        hh[oc] = hh[g]
        people["raised_by"][oc] = RaisedBy.OTHER
        for col in ("region", "tz_offset_min"):
            people[col][oc] = people[col][g]

    # remaining adults: alone, joining a family household, or sharing with 1-2 others.  The mix is
    # solved so the number of households matches population / target mean size (data-driven).
    rest = np.flatnonzero(hh < 0)
    rest = rest[rng.permutation(rest.shape[0])]
    target_households = max(1, round(n / spec.household.mean_size()))
    existing = int(np.unique(hh[hh >= 0]).size)
    p_alone, p_join = _remaining_mix(target_households - existing, int(rest.shape[0]))
    ur: NDArray[np.float64] = np.asarray(rng.random(int(rest.shape[0])))
    alone = rest[ur < p_alone]
    joiners = rest[(ur >= p_alone) & (ur < p_alone + p_join)]
    sharers = rest[ur >= p_alone + p_join]
    for r in alone:
        hh[r] = next_hh
        next_hh += 1
    loc_key = people["region"].astype(np.int64) * 100000 + people["tz_offset_min"].astype(np.int64)
    family = hh >= 0
    for r in joiners:
        pool = np.flatnonzero(family & (loc_key == loc_key[r]) & (people["ethnicity"] == people["ethnicity"][r]))
        if pool.size == 0:
            hh[r] = next_hh
            next_hh += 1
            continue
        hh[r] = hh[int(rng.choice(pool))]
    key = loc_key[sharers]
    order = np.argsort(key, kind="stable")
    sharers, key = sharers[order], key[order]
    i = 0
    while i < sharers.shape[0]:
        size = 3 if rng.random() < 0.3 else 2
        j = i + 1
        while j < sharers.shape[0] and j < i + size and key[j] == key[i]:
            j += 1
        hh[sharers[i:j]] = next_hh
        next_hh += 1
        i = j

    # joint families: where households are still too many (e.g. married sons living with parents),
    # young couples move into a plausible parent's household until the target count is reached
    excess = int(np.unique(hh).size) - target_households
    if excess > 0:
        _form_joint_families(people, hh, spouse, mother, father, excess, rng)

    # renumber households densely in order of first appearance
    _, inv = np.unique(hh, return_inverse=True)
    return HouseholdResult(inv.astype(np.int64), int(inv.max()) + 1 if n else 0, marital, spouse, mother, father)
