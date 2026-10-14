"""Individual attributes by conditional-chain sampling from a CountrySpec.

Order: (sex, age) -> region -> ethnicity, religion -> education | age, sex ->
employment | age, sex -> occupation | education -> income | occupation, age ->
personality, risk, values, wants_kids -> interests -> info tier | education,
internet | age -> discernment, honesty, aggression -> sensory | age -> body ->
time zone, chronotype, shift -> languages | ethnicity, education -> names ->
reputation, visibility, upbringing.

Coefficients marked HEURISTIC have no direct source and are listed in docs as
low-confidence modelling choices.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jahan.gen.names import NameTable, sample_first_names, sample_surnames
from jahan.gen.sampler import categorical_rows, row_index, sigmoid
from jahan.model.country import AgeRow, AgeShare, CountrySpec
from jahan.model.enums import (
    N_TOPICS,
    N_VALUES,
    Big5,
    EmpStatus,
    RaisedBy,
    Topic,
    ValueDim,
    WorkShift,
)
from jahan.model.schema import MAX_LANGS

Arr = NDArray[Any]

EMP_ORDER = [e.name.lower() for e in EmpStatus]
EDU_GROUPS: tuple[Literal["low", "mid", "high"], ...] = ("low", "mid", "high")
VALUE_ORDER = [e.name.lower() for e in ValueDim]


def _age_share(ages: NDArray[np.float64], rows: list[AgeShare], scale: float = 1.0) -> NDArray[np.float64]:
    idx = row_index(ages, rows)
    shares = np.array([r.share for r in rows] + [0.0], dtype=np.float64) / scale
    return shares[idx]  # idx -1 picks the trailing 0.0


def sample_religion(spec: CountrySpec, ethnicity: NDArray[np.int16], rng: np.random.Generator) -> Arr:
    """Religion | ethnicity where the country file gives a conditional table, else the national shares.

    Groups without their own table draw from the national shares re-weighted so the population
    total still matches the national distribution (iterative proportional fitting on one margin).
    """
    rel_codes = spec.religion_codes()
    eth_codes = spec.ethnicity_codes()
    national = np.array([spec.religion.shares[r] for r in rel_codes])
    eth_share = np.array([spec.ethnicity.shares[e] for e in eth_codes])
    cond = np.tile(national, (len(eth_codes), 1))
    fixed = np.zeros(len(eth_codes), dtype=bool)
    for i, e in enumerate(eth_codes):
        if e in spec.religion.by_ethnicity:
            cond[i] = [spec.religion.by_ethnicity[e].get(r, 0.0) for r in rel_codes]
            fixed[i] = True
    if fixed.any() and not fixed.all():
        # remaining mass the free groups must supply so that the national margin holds
        rest = national - (eth_share[fixed, None] * cond[fixed]).sum(axis=0)
        rest = np.clip(rest, 0.0, None)
        if rest.sum() > 0:
            cond[~fixed] = rest / rest.sum()
    out = np.empty(ethnicity.shape[0], dtype=np.int16)
    for i in range(len(eth_codes)):
        m = ethnicity == i
        if m.any():
            out[m] = rng.choice(len(rel_codes), size=int(m.sum()), p=cond[i] / cond[i].sum())
    return out


def internet_probability(spec: CountrySpec, age: NDArray[np.float64]) -> NDArray[np.float64]:
    """Age profile of internet use, rescaled so the population mean equals the World Bank total.

    The rescaling factor is solved so that ``mean(min(0.99, k * profile)) == target`` (bisection),
    which keeps the age gradient while matching the sourced national share.
    """
    base = _age_share(age, spec.internet.rows)
    target = spec.internet_target
    if target is None or base.size == 0 or base.mean() <= 0:
        return base
    lo, hi = 0.0, 10.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if np.minimum(0.99, k * base).mean() < target:
            lo = k
        else:
            hi = k
    return np.minimum(0.99, 0.5 * (lo + hi) * base)


def edu_group_of(education: NDArray[np.int8]) -> NDArray[np.int64]:
    return np.where(education <= 2, 0, np.where(education <= 5, 1, 2)).astype(np.int64)


def match_rows(rows: list[AgeRow], age: NDArray[np.float64], sex: NDArray[np.int8]) -> NDArray[np.int64]:
    """Row index per person: generic rows first, then sex-specific rows override (-1 if none)."""
    row_of = np.full(age.shape[0], -1, dtype=np.int64)
    for i, r in enumerate(rows):
        if r.sex is None:
            row_of[(age >= r.age_min) & (age < r.age_max)] = i
    for i, r in enumerate(rows):
        if r.sex is not None:
            s = 0 if r.sex == "F" else 1
            row_of[(age >= r.age_min) & (age < r.age_max) & (sex == s)] = i
    return row_of


def education_probs(rows: list[AgeRow]) -> NDArray[np.float64]:
    probs = np.zeros((len(rows), 9))
    for i, r in enumerate(rows):
        for k, v in r.shares.items():
            probs[i, int(k)] = v
    return probs


def _education(spec: CountrySpec, age: NDArray[np.float64], sex: NDArray[np.int8], rng: np.random.Generator) -> Arr:
    n = age.shape[0]
    edu = np.zeros(n, dtype=np.int8)
    edu[(age >= 6) & (age < 12)] = 1
    edu[(age >= 12) & (age < 18)] = 2
    rows = spec.education.rows
    probs = education_probs(rows)
    adult = age >= 18
    ri = match_rows(rows, age[adult], sex[adult])
    # ages beyond the last row use the oldest row of the same sex
    last = {
        s: max((i for i, r in enumerate(rows) if r.sex in (None, "FM"[s])), key=lambda i: rows[i].age_max)
        for s in (0, 1)
    }
    ri = np.where(ri < 0, np.where(sex[adult] == 0, last[0], last[1]), ri)
    edu[adult] = categorical_rows(rng, probs, ri).astype(np.int8)
    return edu


def _employment(spec: CountrySpec, age: NDArray[np.float64], sex: NDArray[np.int8], rng: np.random.Generator) -> Arr:
    n = age.shape[0]
    rows = spec.employment.rows
    probs = np.zeros((len(rows), len(EMP_ORDER)))
    for i, r in enumerate(rows):
        for k, v in r.shares.items():
            if k not in EMP_ORDER:
                raise ValueError(f"unknown employment status '{k}'")
            probs[i, EMP_ORDER.index(k)] = v
    row_of = match_rows(rows, age, sex)
    status = np.full(n, int(EmpStatus.CHILD), dtype=np.int8)
    has = row_of >= 0
    status[has] = categorical_rows(rng, probs, row_of[has]).astype(np.int8)
    status[age < 15] = int(EmpStatus.CHILD)
    return status


def _occupation(spec: CountrySpec, emp: Arr, edu_grp: NDArray[np.int64], rng: np.random.Generator) -> Arr:
    probs = np.zeros((3, 10))
    for gi, g in enumerate(EDU_GROUPS):
        for k, v in spec.occupation.by_edu_group[g].items():
            probs[gi, k] = v
    occ = np.full(emp.shape[0], -1, dtype=np.int8)
    m = emp == int(EmpStatus.EMPLOYED)
    occ[m] = categorical_rows(rng, probs, edu_grp[m]).astype(np.int8)
    return occ


def _income(
    spec: CountrySpec, emp: Arr, occ: Arr, age: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float32]:
    n = emp.shape[0]
    inc = np.zeros(n, dtype=np.float64)
    med = np.array([spec.income.by_occupation[k].median if k in spec.income.by_occupation else 0.0 for k in range(10)])
    sig = np.array([spec.income.by_occupation[k].sigma if k in spec.income.by_occupation else 0.5 for k in range(10)])
    m = occ >= 0
    o = occ[m].astype(np.int64)
    # HEURISTIC age-earnings profile
    a = age[m]
    mult = np.select([a < 25, a < 35, a < 55, a < 65], [0.6, 0.85, 1.05, 1.05], 0.9)
    inc[m] = np.exp(np.log(np.maximum(med[o], 1.0)) + sig[o] * rng.standard_normal(o.shape[0])) * mult
    ref = float(np.median(med[med > 0])) if (med > 0).any() else 1.0
    r = emp == int(EmpStatus.RETIRED)
    inc[r] = 0.45 * ref * np.exp(0.4 * rng.standard_normal(int(r.sum())))  # HEURISTIC pension level
    u = emp == int(EmpStatus.UNEMPLOYED)
    inc[u] = 0.25 * ref * np.exp(0.4 * rng.standard_normal(int(u.sum())))  # HEURISTIC benefits/informal
    return inc.astype(np.float32)


def _interests(age: NDArray[np.float64], edu_grp: NDArray[np.int64], rng: np.random.Generator) -> Arr:
    """HEURISTIC Dirichlet priors by age and education (no source; to be calibrated)."""
    n = age.shape[0]
    alpha = np.ones((n, N_TOPICS))
    young = age < 30
    old = age >= 50
    alpha[young, Topic.ENTERTAINMENT] += 1.0
    alpha[young, Topic.SPORTS] += 0.5
    alpha[young, Topic.TECH] += 0.5
    alpha[old, Topic.HEALTH] += 1.0
    alpha[old, Topic.POLITICS] += 0.5
    alpha[old, Topic.RELIGION] += 0.3
    hi = edu_grp == 2
    alpha[hi, Topic.SCIENCE] += 0.7
    alpha[hi, Topic.ECONOMY] += 0.7
    alpha[hi, Topic.WORLD] += 0.5
    alpha[hi, Topic.POLITICS] += 0.3
    g = rng.gamma(alpha)
    return (g / g.sum(axis=1, keepdims=True)).astype(np.float32)


def _languages(
    spec: CountrySpec,
    eth_codes: list[str],
    ethnicity: Arr,
    edu_grp: NDArray[np.int64],
    lang_codes: list[str],
    rng: np.random.Generator,
) -> tuple[Arr, Arr]:
    n = ethnicity.shape[0]
    lang = np.full((n, MAX_LANGS), -1, dtype=np.int16)
    prof = np.zeros((n, MAX_LANGS), dtype=np.float32)

    def lid(code: str) -> int:
        return lang_codes.index(code) if code in lang_codes else lang_codes.index("other")

    l1spec = spec.languages.l1
    l1_code = np.empty(n, dtype=object)
    for e_idx, e_code in enumerate(eth_codes):
        m = ethnicity == e_idx
        if not m.any():
            continue
        dist = l1spec.by_ethnicity.get(e_code, l1spec.default)
        codes = list(dist)
        probs = np.array([dist[c] for c in codes])
        l1_code[m] = np.array(codes, dtype=object)[rng.choice(len(codes), size=int(m.sum()), p=probs)]
    lang[:, 0] = np.array([lid(c) for c in l1_code], dtype=np.int16)
    prof[:, 0] = 1.0
    mult = np.asarray(spec.languages.l2_education_multiplier)[edu_grp]
    nxt = np.ones(n, dtype=np.int64)
    for l1, cands in spec.languages.l2_given_l1.items():
        m_l1 = l1_code == l1
        for code, p in cands.items():
            draw = m_l1 & (nxt < MAX_LANGS) & (rng.random(n) < np.clip(p * mult, 0, 1))
            draw &= ~(lang == lid(code)).any(axis=1)
            idx = np.flatnonzero(draw)
            lang[idx, nxt[idx]] = lid(code)
            prof[idx, nxt[idx]] = rng.uniform(0.5, 0.95, idx.shape[0]).astype(np.float32)
            nxt[idx] += 1
    return lang, prof


def generate_people(
    spec: CountrySpec, n: int, rng: np.random.Generator, lang_codes: list[str], names: NameTable
) -> dict[str, Arr]:
    a = spec.age_sex
    nb = len(a.total_pct)
    b = rng.choice(nb, size=n, p=np.asarray(a.total_pct))
    sex = (rng.random(n) < np.asarray(a.male_share)[b]).astype(np.int8)
    lo = np.asarray(a.bin_edges[:-1], dtype=np.float64)[b]
    hi = np.asarray(a.bin_edges[1:], dtype=np.float64)[b]
    age = lo + rng.random(n) * (hi - lo)
    male = sex == 1

    region = (rng.random(n) < spec.urban_share.value).astype(np.int16)
    eth_codes = spec.ethnicity_codes()
    ethnicity = rng.choice(len(eth_codes), size=n, p=np.array(list(spec.ethnicity.shares.values()))).astype(np.int16)
    religion = sample_religion(spec, ethnicity, rng)

    education = _education(spec, age, sex, rng)
    edu_grp = edu_group_of(education)
    emp = _employment(spec, age, sex, rng)
    occ = _occupation(spec, emp, edu_grp, rng)
    income = _income(spec, emp, occ, age, rng)
    field = np.where(education >= 5, rng.integers(0, 10, n), -1).astype(np.int8)  # PLACEHOLDER uniform field

    z = rng.standard_normal
    big5 = (np.asarray(spec.big5.mean) + np.asarray(spec.big5.sd) * z((n, 5))).astype(np.float32)
    O, C, E, A, N = (
        big5[:, i]
        for i in (Big5.OPENNESS, Big5.CONSCIENTIOUSNESS, Big5.EXTRAVERSION, Big5.AGREEABLENESS, Big5.NEUROTICISM)
    )
    # HEURISTIC: risk tolerance from openness, neuroticism, age, sex
    risk = sigmoid(0.5 * O - 0.4 * N - 0.02 * (age - 35) + 0.25 * male + 0.4 * z(n)).astype(np.float32)
    vmean = np.array([spec.values.mean[k] for k in VALUE_ORDER])
    values = rng.dirichlet(vmean * spec.values.concentration, size=n).astype(np.float32)
    fam = values[:, ValueDim.FAMILY]
    wants_kids = sigmoid(0.8 + 6.0 * (fam - vmean[ValueDim.FAMILY]) - 0.04 * np.maximum(age - 30, 0) + 0.5 * z(n))
    interests = _interests(age, edu_grp, rng)

    tier_p = np.array([spec.info_tier.by_edu_group[g] for g in ("low", "mid", "high")])
    info_tier = categorical_rows(rng, tier_p, edu_grp).astype(np.int8)
    internet = rng.random(n) < internet_probability(spec, age)
    focus = np.where(info_tier == 3, interests.argmax(axis=1), -1).astype(np.int8)
    discernment = np.clip(0.25 + 0.12 * info_tier + 0.025 * education + 0.06 * O + 0.08 * z(n), 0.05, 0.95)
    honesty = sigmoid(1.6 + 0.6 * A + 0.4 * C + 0.4 * z(n))  # HEURISTIC
    aggression = sigmoid(-3.0 - 0.9 * A + 0.4 * E + 0.4 * N + 0.5 * z(n))  # HEURISTIC

    def sensory(rows: list[AgeShare]) -> Arr:
        p = _age_share(age, rows, scale=100.0)
        imp = rng.random(n) < p
        severe = imp & (rng.random(n) < 0.25)  # HEURISTIC split moderate/severe
        return (imp.astype(np.int8) + severe.astype(np.int8)).astype(np.int8)

    vision = sensory(spec.sensory.vision_pct)
    hearing = sensory(spec.sensory.hearing_pct)
    attractiveness = z(n).astype(np.float32)
    strength = 0.8 * np.where(male, 1, -1) - 0.03 * np.maximum(age - 35, 0) - 1.5 * (age < 12) + 0.5 * z(n)
    health = np.clip(1.0 - 0.007 * np.maximum(age - 30, 0) + 0.05 * z(n), 0.05, 1.0)

    tz_rows = spec.time_zones.rows
    tz = np.array([r.offset_min for r in tz_rows], dtype=np.int16)[
        rng.choice(len(tz_rows), size=n, p=np.array([r.share for r in tz_rows]))
    ]
    chronotype = (z(n) + 0.7 * ((age >= 15) & (age < 25)) - 0.4 * (age >= 60)).astype(np.float32)
    shift = np.zeros(n, dtype=np.int8)
    emp_m = emp == int(EmpStatus.EMPLOYED)
    shift[emp_m] = rng.choice(
        [WorkShift.DAY, WorkShift.EVENING, WorkShift.NIGHT], size=int(emp_m.sum()), p=[0.84, 0.08, 0.08]
    )  # HEURISTIC non-day shift share

    lang, prof = _languages(spec, eth_codes, ethnicity, edu_grp, lang_codes, rng)
    rel_codes = spec.religion_codes()
    groups = names.group_index([eth_codes[i] for i in ethnicity], [rel_codes[i] for i in religion])
    first = sample_first_names(names, sex, rng, groups)
    last = sample_surnames(names, n, rng, groups)

    global_rep = rng.beta(4, 4, n)
    # HEURISTIC visibility: log10 of potential audience, community-capped for ordinary agents
    visibility = np.clip(
        1.0
        + 0.5 * (occ == 1)
        + 0.6 * (occ == 2)
        + 0.2 * (occ == 3)
        + 0.15 * E
        + 0.2 * internet
        + rng.exponential(0.2, n)
        - 0.5 * (age < 12),
        0.3,
        3.0,
    )
    childhood_ses = np.clip(sigmoid(0.25 * (education - 3) + 0.6 * z(n)), 0, 1)  # HEURISTIC
    parent_edu = np.clip(education - 1 + np.round(1.2 * z(n)), 0, 8).astype(np.int8)  # HEURISTIC
    raised_by = rng.choice([RaisedBy.TWO_PARENT, RaisedBy.SINGLE, RaisedBy.OTHER], size=n, p=[0.70, 0.25, 0.05]).astype(
        np.int8
    )  # PLACEHOLDER

    wealth = income.astype(np.float64) * 12 * np.maximum(age - 20, 0) / 10 * np.exp(z(n))  # PLACEHOLDER

    return {
        "sex": sex,
        "age": age,
        "region": region,
        "ethnicity": ethnicity,
        "religion": religion,
        "education": education,
        "field": field,
        "emp_status": emp,
        "occupation": occ,
        "income_month": income,
        "wealth": wealth.astype(np.float32),
        "big5": big5,
        "risk_tolerance": risk.astype(np.float32),
        "values": values,
        "wants_kids": wants_kids.astype(np.float32),
        "interests": interests,
        "info_tier": info_tier,
        "internet": internet,
        "focus_topic": focus,
        "discernment": discernment.astype(np.float32),
        "honesty": honesty.astype(np.float32),
        "aggression": aggression.astype(np.float32),
        "vision_impaired": vision,
        "hearing_impaired": hearing,
        "attractiveness_z": attractiveness,
        "strength_z": strength.astype(np.float32),
        "health": health.astype(np.float32),
        "tz_offset_min": tz,
        "chronotype": chronotype,
        "work_shift": shift,
        "lang": lang,
        "lang_prof": prof,
        "first_name_id": first,
        "last_name_id": last,
        "global_rep": global_rep.astype(np.float32),
        "visibility": visibility.astype(np.float32),
        "childhood_ses": childhood_ses.astype(np.float32),
        "parent_edu_max": parent_edu,
        "raised_by": raised_by,
    }


__all__ = ["generate_people", "edu_group_of", "N_VALUES"]
