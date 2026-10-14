"""Social graph: household, kin, colleague, classmate, friend and acquaintance ties.

Undirected ties are generated, de-duplicated (strongest kind wins) and stored as
two directed edges.  Edge ``src -> dst`` means "src listens to / trusts dst";
information flows from dst to src along it.  The two directions of a tie get
uids 2k and 2k+1 so ``uid >> 1`` is a shared pair key (used to make contacts
symmetric).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.gen.population_gen import edu_group_of
from jahan.gen.sampler import logit, sigmoid
from jahan.model.enums import EdgeKind
from jahan.sim.params import SimParams

Arr = NDArray[Any]

# lower = stronger, used to resolve duplicate ties
KIND_PRIORITY = {
    EdgeKind.HOUSEHOLD: 0,
    EdgeKind.KIN: 1,
    EdgeKind.FRIEND: 2,
    EdgeKind.COLLEAGUE: 3,
    EdgeKind.CLASSMATE: 4,
    EdgeKind.CROSS_BORDER: 5,
    EdgeKind.ACQUAINTANCE: 6,
}
# HEURISTIC trust priors by tie kind
BASE_TRUST = {
    EdgeKind.HOUSEHOLD: 0.75,
    EdgeKind.KIN: 0.65,
    EdgeKind.FRIEND: 0.62,
    EdgeKind.COLLEAGUE: 0.55,
    EdgeKind.CLASSMATE: 0.55,
    EdgeKind.CROSS_BORDER: 0.60,  # in M1 these are diaspora family ties
    EdgeKind.ACQUAINTANCE: 0.45,
}


def kind_strength(params: SimParams) -> dict[EdgeKind, float]:
    return {
        EdgeKind.HOUSEHOLD: params.strength_household,
        EdgeKind.KIN: params.strength_kin,
        EdgeKind.COLLEAGUE: params.strength_colleague,
        EdgeKind.CLASSMATE: params.strength_classmate,
        EdgeKind.FRIEND: params.strength_friend,
        EdgeKind.ACQUAINTANCE: params.strength_acquaintance,
        EdgeKind.CROSS_BORDER: params.strength_cross_border,
    }


def _share_language(lang: Arr, a: int, b: int) -> bool:
    la = lang[a][lang[a] >= 0]
    lb = lang[b][lang[b] >= 0]
    return bool(np.intersect1d(la, lb).size)


def undirected_ties(
    people: dict[str, Arr],
    household: NDArray[np.int64],
    workplace: NDArray[np.int64],
    wp_kind: NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return local index arrays (a, b, kind) with a < b, de-duplicated."""
    n = household.shape[0]
    A: list[int] = []
    B: list[int] = []
    K: list[int] = []

    def add(a: int, b: int, kind: EdgeKind) -> None:
        if a != b:
            A.append(min(a, b))
            B.append(max(a, b))
            K.append(int(kind))

    # household: complete graph
    order = np.argsort(household, kind="stable")
    hs = household[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    ends = np.r_[starts[1:], n]
    hh_members = [order[s:e] for s, e in zip(starts, ends, strict=True)]
    for mem in hh_members:
        for i in range(mem.shape[0]):
            for j in range(i + 1, mem.shape[0]):
                add(int(mem[i]), int(mem[j]), EdgeKind.HOUSEHOLD)

    region = people["region"].astype(np.int64)
    tz = people["tz_offset_min"].astype(np.int64)
    eth = people["ethnicity"].astype(np.int64)
    age = people["age"]
    lang = people["lang"]

    # kin: link each household to 1-3 others in the same region, time zone and ethnicity (HEURISTIC)
    hh_key = np.array([region[m[0]] * 10**7 + (tz[m[0]] + 5000) * 100 + eth[m[0]] for m in hh_members])
    by_key: dict[int, list[int]] = {}
    for h, k in enumerate(hh_key):
        by_key.setdefault(int(k), []).append(h)
    for h, mem in enumerate(hh_members):
        kin_pool = by_key[int(hh_key[h])]
        if len(kin_pool) < 2:
            continue
        for _ in range(int(rng.integers(1, 4))):
            o = kin_pool[int(rng.integers(0, len(kin_pool)))]
            if o != h:
                add(int(rng.choice(mem)), int(rng.choice(hh_members[o])), EdgeKind.KIN)

    # colleagues / classmates: up to 5 random others in the same group
    has = workplace >= 0
    w_order = np.argsort(np.where(has, workplace, -1), kind="stable")
    w_sorted = workplace[w_order]
    w_starts = np.flatnonzero(np.r_[True, w_sorted[1:] != w_sorted[:-1]])
    w_ends = np.r_[w_starts[1:], n]
    for s, e in zip(w_starts, w_ends, strict=True):
        if w_sorted[s] < 0:
            continue
        mem = w_order[s:e]
        kind = EdgeKind.CLASSMATE if wp_kind[w_sorted[s]] == 1 else EdgeKind.COLLEAGUE
        for i in mem:
            others = mem[mem != i]
            if others.size == 0:
                continue
            for j in rng.choice(others, size=min(5, others.size), replace=False):
                add(int(i), int(j), kind)

    # friends: homophily on region, time zone, age band, education group; language-gated
    edu_grp = edu_group_of(people["education"])
    band = np.clip((age // 10).astype(np.int64), 0, 9)
    loc = region * 10**6 + (tz + 5000)
    bucket = loc * 1000 + band * 10 + edu_grp
    buckets: dict[int, NDArray[np.int64]] = {}
    for k in np.unique(bucket):
        buckets[int(k)] = np.flatnonzero(bucket == k)
    loc_band: dict[int, NDArray[np.int64]] = {}
    lb_key = loc * 100 + band
    for k in np.unique(lb_key):
        loc_band[int(k)] = np.flatnonzero(lb_key == k)
    loc_members: dict[int, NDArray[np.int64]] = {int(k): np.flatnonzero(loc == k) for k in np.unique(loc)}
    extr = people["big5"][:, 2]
    lam = np.clip(3.5 + 1.2 * extr, 0.5, 9.0) / 2.0  # each tie has two ends
    lam = np.where(age < 3, 0.0, lam)
    n_friends = rng.poisson(lam)
    for i in range(n):
        for _ in range(int(n_friends[i])):
            for _attempt in range(3):
                u = rng.random()
                if u < 0.7:
                    pool = buckets[int(bucket[i])]
                elif u < 0.9:
                    adj = int(band[i] + (1 if rng.random() < 0.5 else -1))
                    pool = loc_band.get(int(loc[i] * 100 + adj), buckets[int(bucket[i])])
                else:
                    pool = loc_members[int(loc[i])]
                j = int(pool[int(rng.integers(0, pool.shape[0]))])
                if j != i and age[j] >= 3 and _share_language(lang, i, j):
                    add(i, j, EdgeKind.FRIEND)
                    break

    # acquaintances: 2 random others in the same region and time zone
    for i in range(n):
        pool = loc_members[int(loc[i])]
        for _ in range(2):
            j = int(pool[int(rng.integers(0, pool.shape[0]))])
            add(i, j, EdgeKind.ACQUAINTANCE)

    if not A:
        e = np.empty(0, np.int64)
        return e, e, e
    a = np.asarray(A, dtype=np.int64)
    b = np.asarray(B, dtype=np.int64)
    k = np.asarray(K, dtype=np.int64)
    pri = np.array([KIND_PRIORITY[EdgeKind(x)] for x in range(len(EdgeKind))])[k]
    order = np.lexsort((pri, b, a))
    a, b, k = a[order], b[order], k[order]
    first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
    return a[first], b[first], k[first]


def directed_edge_columns(
    a: NDArray[np.int64],
    b: NDArray[np.int64],
    kind: NDArray[np.int64],
    people_global: dict[str, Arr],
    uid_start: int,
    params: SimParams,
    day: int = 0,
) -> dict[str, Arr]:
    """Turn undirected ties (global agent ids) into directed edge columns with trust priors.

    ``people_global`` must be indexable by global agent id (row == id).
    """
    m = a.shape[0]
    src = np.empty(2 * m, np.int64)
    dst = np.empty(2 * m, np.int64)
    src[0::2], dst[0::2] = a, b
    src[1::2], dst[1::2] = b, a
    k2 = np.repeat(kind, 2)
    uid = uid_start + np.arange(2 * m, dtype=np.int64)
    strength_map = kind_strength(params)
    strength = np.array([strength_map[EdgeKind(x)] for x in range(len(EdgeKind))], np.float32)[k2]
    base = np.array([BASE_TRUST[EdgeKind(x)] for x in range(len(EdgeKind))])[k2]

    rep = people_global["global_rep"]
    agree_shift = -0.6 * people_global["aggression"][dst]
    edu = people_global["education"]
    age = people_global["age_at_creation"]
    homophily = (
        (np.abs(edu[src].astype(np.int64) - edu[dst].astype(np.int64)) <= 1).astype(np.float64)
        + (np.abs(age[src] - age[dst]) < 10).astype(np.float64)
        + (people_global["religion"][src] == people_global["religion"][dst]).astype(np.float64)
    ) / 3.0
    # HEURISTIC: advice from children is trusted less (full weight from age 16)
    child_shift = -1.5 * (1.0 - np.clip((age[dst] - 8.0) / 8.0, 0.0, 1.0))
    prior = sigmoid(logit(base) + 0.8 * (rep[dst] - 0.5) + 0.5 * (homophily - 0.5) + agree_shift + child_shift)
    halo = 0.15 * people_global["attractiveness_z"][dst]
    return {
        "uid": uid,
        "src": src,
        "dst": dst,
        "kind": k2.astype(np.int8),
        "strength": strength,
        "prior_p": prior.astype(np.float32),
        "halo": halo.astype(np.float32),
        "created_day": np.full(2 * m, day, np.int32),
    }
