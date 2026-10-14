"""Life events: deaths, births, marriages, divorces, labour-market moves and birthday transitions.

Rates come from the country spec: UN WPP life tables and age-specific fertility,
plus the ``vital`` block (marriage, divorce, job flows).  Individual propensities
(wanting children, marital status, aggression, partner age gap, shared ties) shift
who experiences an event; group totals are normalised back to the published rates.

All draws use the stateless RNG keyed by agent ids, so events are deterministic and
identical across baseline/treated branches unless the scenario changes their inputs.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from jahan.gen.names import sample_first_names
from jahan.gen.population_gen import EDU_GROUPS, EMP_ORDER, edu_group_of, internet_probability, match_rows
from jahan.gen.sampler import categorical_rows, sigmoid
from jahan.model.country import AgeRow, CountrySpec
from jahan.model.enums import EdgeKind, EmpStatus, EventKind, Marital, RaisedBy
from jahan.model.world import World
from jahan.rng import MASK64, Stream, u01
from jahan.sim.params import SimParams
from jahan.sim.ties import add_ties, retire_agent_ties, set_tie
from jahan.store.eventlog import EventLog

DAYS_PER_YEAR = 365.25
BIRTH_SPACING_DAYS = 450  # HEURISTIC minimum interval between births to the same mother
UNMARRIED_FERTILITY = 0.55  # HEURISTIC relative fertility of unmarried women (normalised per age group)


def _daily(p_year: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 1.0 - np.power(1.0 - np.clip(np.asarray(p_year, dtype=np.float64), 0.0, 0.999999), 1.0 / DAYS_PER_YEAR)


def _rng(world: World, stream: int, agent_id: int) -> np.random.Generator:
    return np.random.default_rng([world.sim_seed & MASK64, stream, world.day, int(agent_id)])


# ---------------------------------------------------------------- deaths
def mortality_daily(spec: CountrySpec, age: NDArray[np.float64], sex: NDArray[np.int8]) -> NDArray[np.float64]:
    lt = spec.life_table
    if lt is None:
        return np.zeros(age.shape[0])
    starts = np.asarray(lt.age_start, dtype=np.float64)
    widths = np.r_[np.diff(starts), 5.0]
    g = np.clip(np.searchsorted(starts, age, side="right") - 1, 0, starts.shape[0] - 1)
    q = np.where(sex == 0, np.asarray(lt.qx_female)[g], np.asarray(lt.qx_male)[g])
    m = np.where(q >= 0.999, 0.6, -np.log1p(-np.minimum(q, 0.999)) / widths[g])  # annual hazard
    return 1.0 - np.exp(-m / DAYS_PER_YEAR)


def deaths(world: World, log: EventLog) -> int:
    pop = world.pop
    alive = pop["alive"] & ~pop["is_notable"]
    age = world.ages()
    p = np.zeros(pop.n)
    for c, spec in enumerate(world.countries):
        m = alive & (pop["country"] == c)
        p[m] = mortality_daily(spec, age[m], pop["sex"][m])
    die = alive & (u01(world.sim_seed, Stream.DEATH, world.day, pop["id"]) < p)
    idx = np.flatnonzero(die)
    if idx.size == 0:
        return 0
    pop["alive"][idx] = False
    pop["death_day"][idx] = world.day
    sp = pop["spouse_id"][idx]
    has = sp >= 0
    pop["marital"][sp[has]] = Marital.WIDOWED
    pop["spouse_id"][sp[has]] = -1
    pop["spouse_id"][idx] = -1
    np.subtract.at(world.households["n_members"], pop["household_id"][idx], 1)
    if pop["workplace_id"][idx].max(initial=-1) >= 0:
        w = pop["workplace_id"][idx]
        np.subtract.at(world.workplaces["size"], w[w >= 0], 1)
    retire_agent_ties(world, pop["id"][idx])
    log.add(world.day, EventKind.DIED, pop["id"][idx], -1, -1, age[idx], pop["location_country"][idx])
    return int(idx.size)


# ---------------------------------------------------------------- births
def births(world: World, log: EventLog) -> int:
    pop = world.pop
    day = world.day
    age = world.ages()
    alive = pop["alive"]
    women = (
        alive
        & ~pop["is_notable"]
        & (pop["sex"] == 0)
        & (age >= 10)
        & (age < 55)
        & (day - pop["last_birth_day"] >= BIRTH_SPACING_DAYS)
    )
    p = np.zeros(pop.n)
    for c, spec in enumerate(world.countries):
        fr = spec.fertility
        if fr is None:
            continue
        mc = women & (pop["country"] == c)
        if not mc.any():
            continue
        starts = np.asarray(fr.age_start, dtype=np.float64)
        g = np.clip(np.searchsorted(starts, age, side="right") - 1, 0, starts.shape[0] - 1)
        asfr = np.asarray(fr.asfr_per_woman_year)[g]
        f = np.where(pop["marital"] == Marital.MARRIED, 1.0, UNMARRIED_FERTILITY) * (0.5 + pop["wants_kids"])
        # normalise propensities within each age group so group totals match the ASFR
        for gi in np.unique(g[mc]):
            sel = mc & (g == gi)
            mean_f = f[sel].mean()
            # ASFR is a rate (births per woman-year), so the daily probability is rate / 365.25
            p[sel] = asfr[sel] * f[sel] / max(mean_f, 1e-9) / DAYS_PER_YEAR
    mothers = np.flatnonzero(women & (u01(world.sim_seed, Stream.BIRTH, day, pop["id"]) < p))
    for m in mothers:
        _give_birth(world, int(m), log)
    return int(mothers.size)


def _give_birth(world: World, mother: int, log: EventLog) -> int:
    pop = world.pop
    c = int(pop["country"][mother])
    spec = world.countries[c]
    child_id = int(world.alloc_ids("agent", 1)[0])
    rng = _rng(world, Stream.BIRTH, child_id)
    father = int(pop["spouse_id"][mother]) if pop["marital"][mother] == Marital.MARRIED else -1
    srb = (spec.indicators.get("sex_ratio_at_birth_males_per_100_females") or 105.0) / 100.0
    sex = np.array([1 if rng.random() < srb / (1 + srb) else 0], np.int8)
    parents = [mother] + ([father] if father >= 0 else [])
    mid = pop["big5"][parents].mean(axis=0)
    big5 = (0.4 * mid + 0.9 * rng.standard_normal(5)).astype(np.float32)  # HEURISTIC heritability
    vmean = np.array([spec.values.mean[k] for k in spec.values.mean])
    values = rng.dirichlet((0.5 * pop["values"][mother] + 0.5 * vmean) * spec.values.concentration).astype(np.float32)
    interests = rng.dirichlet(np.ones(pop["interests"].shape[1])).astype(np.float32)
    hh = int(pop["household_id"][mother])
    lang = np.full(3, -1, np.int16)
    lang[0] = pop["lang"][mother, 0]
    prof = np.array([1.0, 0, 0], np.float32)
    surname_src = father if father >= 0 else mother
    world.pop.append(
        1,
        id=child_id,
        alive=True,
        country=c,
        birth_country=int(pop["location_country"][mother]),
        location_country=int(pop["location_country"][mother]),
        region=int(pop["region"][mother]),
        household_id=hh,
        first_name_id=sample_first_names(
            world.names[c],
            sex,
            rng,
            world.names[c].group_index(
                [spec.ethnicity_codes()[int(pop["ethnicity"][mother])]],
                [spec.religion_codes()[int(pop["religion"][mother])]],
            ),
        ),
        last_name_id=int(pop["last_name_id"][surname_src]),
        pop_weight=float(pop["pop_weight"][mother]),
        sex=sex,
        birth_day=world.day,
        ethnicity=int(pop["ethnicity"][mother]),
        religion=int(pop["religion"][mother]),
        marital=int(Marital.SINGLE),
        mother_id=mother,
        father_id=father,
        big5=big5[None, :],
        values=values[None, :],
        interests=interests[None, :],
        wants_kids=float(sigmoid(0.8 + 6.0 * (values[4] - vmean[4]))),
        risk_tolerance=float(sigmoid(0.5 * big5[0] - 0.4 * big5[4] + 0.4 * rng.standard_normal())),
        discernment=0.2,
        honesty=float(sigmoid(1.6 + 0.6 * big5[3] + 0.4 * big5[1] + 0.4 * rng.standard_normal())),
        aggression=float(sigmoid(-3.0 - 0.9 * big5[3] + 0.4 * big5[2] + 0.4 * big5[4] + 0.5 * rng.standard_normal())),
        parent_edu_max=int(pop["education"][parents].max()),
        raised_by=int(RaisedBy.TWO_PARENT if father >= 0 else RaisedBy.SINGLE),
        childhood_ses=float(pop["childhood_ses"][mother]),
        education=0,
        emp_status=int(EmpStatus.CHILD),
        info_tier=0,
        internet=False,
        lang=lang[None, :],
        lang_prof=prof[None, :],
        tz_offset_min=int(pop["home_tz_offset_min"][mother]),
        home_tz_offset_min=int(pop["home_tz_offset_min"][mother]),
        chronotype=float(rng.standard_normal()),
        attractiveness_z=float(rng.standard_normal()),
        strength_z=-1.5,
        health=1.0,
        global_rep=0.5,
        visibility=0.3,
    )
    world.beliefs.ensure_rows(world.pop.n)
    world.ptrust.ensure_rows(world.pop.n)
    pop = world.pop  # same object; refresh views after a possible reallocation
    pop["last_birth_day"][mother] = world.day
    for p_ in parents:
        pop["n_children"][p_] = min(int(pop["n_children"][p_]) + 1, 127)
    world.households["n_members"][hh] += 1
    members = np.flatnonzero((pop["household_id"] == hh) & pop["alive"])
    members = members[members != child_id]
    add_ties(world, np.full(members.size, child_id), members, EdgeKind.HOUSEHOLD)
    grand = [int(pop[k][p_]) for p_ in parents for k in ("mother_id", "father_id")]
    grand = [g for g in grand if g >= 0 and pop["alive"][g] and g not in members]
    if grand:
        add_ties(world, np.full(len(grand), child_id), np.array(grand), EdgeKind.KIN)
    log.add(world.day, EventKind.BORN, child_id, mother, -1, 0.0, int(pop["location_country"][child_id]))
    return child_id


# ---------------------------------------------------------------- marriages
def _marriage_profile(age: NDArray[np.float64], spec: CountrySpec) -> NDArray[np.float64]:
    v = spec.vital
    peak = v.marriage_age_peak if v else 29.0
    sd = v.marriage_age_sd if v else 8.0
    return np.where(age >= 18, np.exp(-0.5 * ((age - peak) / sd) ** 2), 0.0)


def marriages(world: World, log: EventLog) -> int:
    pop = world.pop
    day = world.day
    age = world.ages()
    alive = pop["alive"]
    unmarried = (
        alive
        & (age >= 18)
        & (pop["marital"] != Marital.MARRIED)
        & (pop["controlled_by"] < 0)
        & ~pop["is_notable"]
        & (pop["trip_end_day"] < 0)
    )
    total = 0
    for c, spec in enumerate(world.countries):
        if spec.vital is None:
            continue
        in_c = pop["country"] == c  # residents (travellers marry at home)
        n_c = int((alive & in_c).sum())
        expected = spec.vital.marriage_rate_per_1000 / 1000.0 * n_c / DAYS_PER_YEAR  # marriages per day
        women = unmarried & in_c & (pop["sex"] == 0)
        if not women.any() or expected <= 0:
            continue
        status_w = np.select([pop["marital"] == Marital.WIDOWED, pop["marital"] == Marital.DIVORCED], [0.3, 0.8], 1.0)
        g = _marriage_profile(age, spec) * status_w * women
        if g.sum() <= 0:
            continue
        p = np.minimum(expected * g / g.sum(), 0.5)
        brides = np.flatnonzero(women & (u01(world.sim_seed, Stream.MARRY, day, pop["id"]) < p))
        for w in brides:
            if _marry(world, int(w), spec, unmarried & in_c & (pop["sex"] == 1), age, log):
                total += 1
                unmarried[w] = False
                unmarried[pop["spouse_id"][w]] = False
    return total


def _marry(
    world: World, w: int, spec: CountrySpec, men_mask: NDArray[np.bool_], age: NDArray[np.float64], log: EventLog
) -> bool:
    pop = world.pop
    men = np.flatnonzero(men_mask)
    if men.size == 0:
        return False
    hh_spec = spec.household
    gap = age[men] - age[w]
    wgt = np.exp(-0.5 * ((gap - hh_spec.spouse_age_gap_mean) / max(hh_spec.spouse_age_gap_sd, 1.0)) ** 2)
    same_place = (pop["tz_offset_min"][men] == pop["tz_offset_min"][w]) & (pop["region"][men] == pop["region"][w])
    wgt *= np.where(same_place, 1.0, 0.1)
    wgt *= np.where(pop["ethnicity"][men] == pop["ethnicity"][w], 2.0, 1.0)  # HEURISTIC homogamy
    edu = edu_group_of(pop["education"])
    wgt *= np.where(edu[men] == edu[w], 1.5, 1.0)
    e = world.edges
    connected = np.unique(e["dst"][(e["src"] == w) & e["active"]])
    wgt *= np.where(np.isin(men, connected), 5.0, 1.0)  # people marry within their networks
    if wgt.sum() <= 0:
        return False
    rng = _rng(world, Stream.MARRY, w)
    h = int(men[rng.choice(men.size, p=wgt / wgt.sum())])
    pop["marital"][[w, h]] = Marital.MARRIED
    pop["spouse_id"][w] = h
    pop["spouse_id"][h] = w
    # new household for the couple; the wife's children under 18 move with her
    new_hh = world.households.n
    world.households.append(
        1, id=new_hh, country=int(pop["country"][w]), region=int(pop["region"][w]), n_members=0, income_pooled=0.0
    )
    world.counters["household"] = world.households.n
    kids = np.flatnonzero(
        pop["alive"]
        & ((pop["mother_id"] == w) | (pop["father_id"] == h))
        & (age < 18)
        & np.isin(pop["household_id"], pop["household_id"][[w, h]])
    )
    movers = np.unique(np.r_[[w, h], kids])
    np.subtract.at(world.households["n_members"], pop["household_id"][movers], 1)
    pop["household_id"][movers] = new_hh
    world.households["n_members"][new_hh] = movers.size
    pop["region"][movers] = pop["region"][w]
    pop["home_tz_offset_min"][movers] = pop["home_tz_offset_min"][w]
    home_now = movers[pop["trip_end_day"][movers] < 0]
    pop["tz_offset_min"][home_now] = pop["home_tz_offset_min"][w]
    adoption = world.names[int(pop["country"][w])].surname_adoption  # customary rate from the names file
    if pop["sex"][w] == 0 and _rng(world, Stream.MARRY + 100, w).random() < adoption:
        pop["last_name_id"][w] = pop["last_name_id"][h]
    a, b = np.meshgrid(movers, movers)
    iu = np.triu_indices(movers.size, 1)
    add_ties(world, a[iu], b[iu], EdgeKind.HOUSEHOLD)
    inlaws = [int(pop[k][x]) for x in (w, h) for k in ("mother_id", "father_id")]
    pairs = [(h, pop["mother_id"][w]), (h, pop["father_id"][w]), (w, pop["mother_id"][h]), (w, pop["father_id"][h])]
    pairs = [(x, int(y)) for x, y in pairs if y >= 0 and pop["alive"][y]]
    if pairs:
        add_ties(world, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), EdgeKind.KIN)
    _ = inlaws
    log.add(world.day, EventKind.MARRIED, w, h, -1, float(age[w]), int(pop["location_country"][w]))
    return True


# ---------------------------------------------------------------- divorces
def divorces(world: World, log: EventLog) -> int:
    pop = world.pop
    day = world.day
    alive = pop["alive"]
    total = 0
    for c, spec in enumerate(world.countries):
        if spec.vital is None:
            continue
        in_c = pop["country"] == c  # residents (travellers marry at home)
        n_c = int((alive & in_c).sum())
        wives = np.flatnonzero(
            alive
            & in_c
            & (pop["sex"] == 0)
            & (pop["marital"] == Marital.MARRIED)
            & (pop["spouse_id"] >= 0)
            & (pop["controlled_by"] < 0)
        )
        if wives.size == 0:
            continue
        husbands = pop["spouse_id"][wives]
        expected = spec.vital.divorce_rate_per_1000 / 1000.0 * n_c / DAYS_PER_YEAR
        # HEURISTIC: aggression of either partner and low family values raise the hazard
        wgt = (
            1.0
            + 2.0 * (pop["aggression"][wives] + pop["aggression"][husbands])
            + 1.5 * (0.2 - 0.5 * (pop["values"][wives, 4] + pop["values"][husbands, 4])).clip(0)
        )
        p = np.minimum(expected * wgt / wgt.sum(), 0.5)
        split = wives[u01(world.sim_seed, Stream.DIVORCE, day, pop["id"][wives]) < p]
        for w in split:
            _divorce(world, int(w), log)
            total += 1
    return total


def _divorce(world: World, w: int, log: EventLog) -> None:
    pop = world.pop
    h = int(pop["spouse_id"][w])
    pop["marital"][[w, h]] = Marital.DIVORCED
    pop["spouse_id"][[w, h]] = -1
    leaver = h  # HEURISTIC: children stay with the mother; the husband moves out
    old = int(pop["household_id"][leaver])
    new_hh = world.households.n
    world.households.append(
        1,
        id=new_hh,
        country=int(pop["country"][leaver]),
        region=int(pop["region"][leaver]),
        n_members=1,
        income_pooled=0.0,
    )
    world.counters["household"] = world.households.n
    world.households["n_members"][old] -= 1
    pop["household_id"][leaver] = new_hh
    set_tie(world, np.array([w]), np.array([h]), EdgeKind.ACQUAINTANCE, beta_penalty=3.0)
    kids = np.flatnonzero(pop["alive"] & (pop["father_id"] == h) & (pop["household_id"] == old))
    if kids.size:
        set_tie(world, np.full(kids.size, h), kids, EdgeKind.KIN)
    log.add(world.day, EventKind.DIVORCED, w, h, -1, 0.0, int(pop["location_country"][w]))


# ---------------------------------------------------------------- labour market
def _join_workplace(world: World, agent: int, kind: int) -> None:
    pop = world.pop
    wp = world.workplaces
    same = (wp["country"] == pop["country"][agent]) & (wp["region"] == pop["region"][agent]) & (wp["kind"] == kind)
    cands = np.flatnonzero(same & (np.arange(wp.n) != pop["workplace_id"][agent]))
    if cands.size == 0:
        return
    rng = _rng(world, Stream.JOB, agent)
    new = int(cands[rng.integers(0, cands.size)])
    _leave_workplace(world, agent)
    members = np.flatnonzero((pop["workplace_id"] == new) & pop["alive"])
    pop["workplace_id"][agent] = new
    wp["size"][new] += 1
    if members.size:
        pick = rng.choice(members, size=min(5, members.size), replace=False)
        add_ties(world, np.full(pick.size, agent), pick, EdgeKind.CLASSMATE if kind == 1 else EdgeKind.COLLEAGUE)


def _leave_workplace(world: World, agent: int) -> None:
    pop = world.pop
    old = int(pop["workplace_id"][agent])
    if old < 0:
        return
    world.workplaces["size"][old] -= 1
    pop["workplace_id"][agent] = -1
    e = world.edges
    rows = np.flatnonzero(
        ((e["src"] == agent) | (e["dst"] == agent)) & np.isin(e["kind"], [EdgeKind.COLLEAGUE, EdgeKind.CLASSMATE])
    )
    if rows.size:
        e["kind"][rows] = int(EdgeKind.ACQUAINTANCE)
        e["strength"][rows] = world.config.params.strength_acquaintance


def _draw_job(world: World, agent: int) -> None:
    """Give an agent an occupation (if none) and an income for it."""
    pop = world.pop
    spec = world.countries[int(pop["country"][agent])]
    rng = _rng(world, Stream.JOB + 200, agent)
    if pop["occupation"][agent] < 0:
        g = EDU_GROUPS[int(edu_group_of(pop["education"][agent : agent + 1])[0])]
        occ = spec.occupation.by_edu_group[g]
        keys = list(occ)
        pop["occupation"][agent] = keys[int(rng.choice(len(keys), p=np.array([occ[k] for k in keys])))]
    row = spec.income.by_occupation.get(int(pop["occupation"][agent]))
    if row is not None:
        pop["income_month"][agent] = float(math.exp(math.log(row.median) + row.sigma * rng.standard_normal()))


def labour(world: World, log: EventLog) -> int:
    pop = world.pop
    day = world.day
    alive = pop["alive"] & (pop["controlled_by"] < 0) & ~pop["is_notable"]
    moves = 0
    for c, spec in enumerate(world.countries):
        v = spec.vital
        if v is None:
            continue
        in_c = alive & (pop["country"] == c)
        u = u01(world.sim_seed, Stream.JOB, day, pop["id"])
        u2 = u01(world.sim_seed, Stream.JOB, day, pop["id"], k=1)
        employed = in_c & (pop["emp_status"] == EmpStatus.EMPLOYED)
        unemployed = in_c & (pop["emp_status"] == EmpStatus.UNEMPLOYED)
        p_loss = 1 - (1 - v.job_loss_per_month) ** (12 / DAYS_PER_YEAR)
        p_find = 1 - (1 - v.job_find_per_month) ** (12 / DAYS_PER_YEAR)
        p_change = _daily(v.job_change_per_year)
        lose = np.flatnonzero(employed & (u < p_loss))
        change = np.flatnonzero(employed & (u >= p_loss) & (u2 < p_change))
        find = np.flatnonzero(unemployed & (u < p_find))
        ref = float(np.median([r.median for r in spec.income.by_occupation.values()]))
        for a in lose:
            _leave_workplace(world, int(a))
            pop["emp_status"][a] = EmpStatus.UNEMPLOYED
            pop["income_month"][a] = 0.25 * ref
            log.add(day, EventKind.STATUS_CHANGED, int(a), -1, -1, float(EmpStatus.UNEMPLOYED), c, "job_loss")
        for a in change:
            _join_workplace(world, int(a), 0)
            pop["income_month"][a] *= np.float32(
                math.exp(0.05 + 0.1 * _rng(world, Stream.JOB + 300, int(a)).standard_normal())
            )
            log.add(day, EventKind.JOB_CHANGED, int(a), int(pop["workplace_id"][a]), -1, 0.0, c)
        for a in find:
            pop["emp_status"][a] = EmpStatus.EMPLOYED
            _draw_job(world, int(a))
            _join_workplace(world, int(a), 0)
            log.add(day, EventKind.STATUS_CHANGED, int(a), -1, -1, float(EmpStatus.EMPLOYED), c, "job_found")
        moves += int(lose.size + change.size + find.size)
    return moves


# ---------------------------------------------------------------- birthdays
TRANSITION_AGES = (5, 11, 15, 18, 25, 55, 65, 75)


def birthdays(world: World, log: EventLog) -> int:
    pop = world.pop
    alive = pop["alive"]
    bd = pop["birth_day"].astype(np.int64)
    a_now = np.floor((world.day - bd) / DAYS_PER_YEAR).astype(np.int64)
    a_prev = np.floor((world.day - 1 - bd) / DAYS_PER_YEAR).astype(np.int64)
    turned = alive & ~pop["is_notable"] & (a_now > a_prev) & np.isin(a_now, TRANSITION_AGES)
    changed = 0
    for a in np.flatnonzero(turned):
        changed += _transition(world, int(a), int(a_now[a]), log)
    return changed


def _status_row(spec: CountrySpec, age: float, sex: int) -> dict[str, float] | None:
    hit = None
    for r in spec.employment.rows:
        if r.age_min <= age < r.age_max and (r.sex is None or (r.sex == "F") == (sex == 0)):
            if hit is None or r.sex is not None:
                hit = r
    return hit.shares if hit else None


def _edu_row(spec: CountrySpec, age: float, sex: int) -> AgeRow | None:
    i = int(match_rows(spec.education.rows, np.array([age]), np.array([sex], np.int8))[0])
    return spec.education.rows[i] if i >= 0 else None


def _transition(world: World, a: int, age: int, log: EventLog) -> int:
    pop = world.pop
    c = int(pop["country"][a])
    spec = world.countries[c]
    rng = _rng(world, Stream.JOB + 400, a)
    sex = int(pop["sex"][a])
    if age in (5, 11, 15):
        pop["education"][a] = {5: 0, 11: 1, 15: 2}[age]
        if pop["emp_status"][a] in (EmpStatus.CHILD, EmpStatus.STUDENT):
            _leave_workplace(world, a)
            _join_workplace(world, a, 1)
    if age == 18:
        row = _edu_row(spec, 18.0, sex)
        if row:
            levels = np.array([int(k) for k in row.shares])
            probs = np.array(list(row.shares.values()))
            pop["education"][a] = max(int(pop["education"][a]), int(levels[rng.choice(levels.size, p=probs)]))
        g = int(edu_group_of(pop["education"][a : a + 1])[0])
        tiers = np.array(spec.info_tier.by_edu_group[EDU_GROUPS[g]])
        pop["info_tier"][a] = int(categorical_rows(rng, tiers[None, :], np.zeros(1, np.int64))[0])
        if rng.random() < float(internet_probability(spec, np.array([18.0]))[0]):
            pop["internet"][a] = True
    if age == 25:
        row = _edu_row(spec, 25.0, sex)
        if row:
            cur = int(pop["education"][a])
            levels = np.array([int(k) for k in row.shares])
            probs = np.array(list(row.shares.values())) * (levels >= cur)
            if probs.sum() > 0:
                pop["education"][a] = int(levels[rng.choice(levels.size, p=probs / probs.sum())])
    if age >= 15:
        shares = _status_row(spec, age + 0.01, sex)
        if shares:
            cur = int(pop["emp_status"][a])
            if cur == EmpStatus.RETIRED:
                return 1
            keys = list(shares)
            new = EMP_ORDER.index(keys[int(rng.choice(len(keys), p=np.array([shares[k] for k in keys])))])
            if new == cur:
                return 1
            if age in (55, 65, 75) and new == EmpStatus.STUDENT:
                return 1
            if cur == EmpStatus.EMPLOYED and new != EmpStatus.EMPLOYED:
                _leave_workplace(world, a)
                pop["income_month"][a] = (
                    0.45 * float(np.median([r.median for r in spec.income.by_occupation.values()]))
                    if new == EmpStatus.RETIRED
                    else 0.0
                )
            if new == EmpStatus.EMPLOYED and cur != EmpStatus.EMPLOYED:
                _leave_workplace(world, a)  # leaves a school class if any
                pop["emp_status"][a] = new
                _draw_job(world, a)
                _join_workplace(world, a, 0)
            if new == EmpStatus.STUDENT and cur != EmpStatus.STUDENT and pop["workplace_id"][a] < 0:
                _join_workplace(world, a, 1)
            if new not in (EmpStatus.EMPLOYED, EmpStatus.STUDENT) and cur in (EmpStatus.STUDENT, EmpStatus.CHILD):
                _leave_workplace(world, a)
            pop["emp_status"][a] = new
            log.add(world.day, EventKind.STATUS_CHANGED, a, -1, -1, float(new), c, f"age_{age}")
    return 1


def refresh_households(world: World) -> None:
    pop = world.pop
    alive = pop["alive"] & (pop["household_id"] >= 0)
    hh = world.households
    hh["income_pooled"] = np.bincount(
        pop["household_id"][alive], weights=pop["income_month"][alive].astype(np.float64), minlength=hh.n
    )[: hh.n].astype(np.float32)


def run_life_events(world: World, params: SimParams, log: EventLog) -> dict[str, int]:
    stats = {
        "deaths": deaths(world, log),
        "births": births(world, log),
        "marriages": marriages(world, log),
        "divorces": divorces(world, log),
        "labour_moves": labour(world, log),
        "transitions": birthdays(world, log),
    }
    refresh_households(world)
    return stats
