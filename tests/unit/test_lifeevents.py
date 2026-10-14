import numpy as np
import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EdgeKind, EventKind, Marital
from jahan.sim.lifeevents import _divorce, _give_birth, mortality_daily
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog


def test_mortality_matches_life_table(small_world_factory):
    w = small_world_factory(200)
    spec = w.countries[0]
    lt = spec.life_table
    # annual survival from the daily hazard reproduces 1 - qx over a 5-year interval
    age = np.array([47.0])
    p = mortality_daily(spec, age, np.array([1], np.int8))[0]
    q5 = 1 - (1 - p) ** (5 * 365.25)
    expected = lt.qx_male[lt.age_start.index(45)]
    assert abs(q5 - expected) / expected < 0.01


def test_birth_creates_linked_child(small_world_factory):
    w = small_world_factory(800, seed=2)
    pop = w.pop
    age = w.ages()
    mother = int(np.flatnonzero((pop["sex"] == 0) & (pop["marital"] == Marital.MARRIED) & (age > 25) & (age < 40))[0])
    n0 = pop.n
    log = EventLog()
    child = _give_birth(w, mother, log)
    assert pop.n == n0 + 1 and child == n0 and pop["id"][child] == child
    assert pop["mother_id"][child] == mother and pop["father_id"][child] == pop["spouse_id"][mother]
    assert pop["household_id"][child] == pop["household_id"][mother]
    assert pop["birth_day"][child] == w.day and w.ages()[child] == 0
    assert pop["last_name_id"][child] == pop["last_name_id"][pop["father_id"][child]]
    rows = w.find_edges(np.array([child]), np.array([mother]))
    assert rows[0] >= 0 and w.edges["kind"][rows[0]] == EdgeKind.HOUSEHOLD
    assert w.beliefs.m["aware"].shape[0] >= pop.n
    assert log.table().column("kind").to_pylist() == [EventKind.BORN]


def test_divorce_splits_household(small_world_factory):
    w = small_world_factory(800, seed=2)
    pop = w.pop
    wife = int(np.flatnonzero((pop["sex"] == 0) & (pop["marital"] == Marital.MARRIED))[0])
    husband = int(pop["spouse_id"][wife])
    hh = int(pop["household_id"][wife])
    _divorce(w, wife, EventLog())
    assert pop["marital"][wife] == pop["marital"][husband] == Marital.DIVORCED
    assert pop["spouse_id"][wife] == pop["spouse_id"][husband] == -1
    assert pop["household_id"][husband] != hh and pop["household_id"][wife] == hh
    r = w.find_edges(np.array([wife]), np.array([husband]))[0]
    assert w.edges["kind"][r] == EdgeKind.ACQUAINTANCE and w.edges["beta"][r] > 0


def test_a_year_of_life_keeps_invariants(small_world_factory):
    w = small_world_factory(3000, seed=6)
    rec = run_in_memory(w, 365)
    pop = w.pop
    alive = pop["alive"]
    kinds = np.array(rec.events.table().column("kind").to_pylist())
    assert (kinds == EventKind.BORN).any() and (kinds == EventKind.DIED).any() and (kinds == EventKind.MARRIED).any()
    # spouses are mutual, alive, opposite sex and co-resident
    sp = pop["spouse_id"]
    m = alive & (sp >= 0)
    assert np.array_equal(sp[sp[m]], np.flatnonzero(m))
    assert alive[sp[m]].all()
    assert (pop["household_id"][m] == pop["household_id"][sp[m]]).all()
    assert (pop["marital"][m] == Marital.MARRIED).all()
    # the dead have no active ties and household counts match residents
    e = w.edges
    dead = ~alive
    assert not (e["active"] & (dead[e["src"]] | dead[e["dst"]])).any()
    resident = alive & (pop["household_id"] >= 0)
    counts = np.bincount(pop["household_id"][resident], minlength=w.households.n)
    assert np.array_equal(counts, w.households["n_members"][: w.households.n])
    # nobody acts after death: no events for dead agents after their death day
    ev = rec.events.table().to_pydict()
    death_day = pop["death_day"]
    for day, kind, agent in zip(ev["day"], ev["kind"], ev["agent"], strict=True):
        if agent >= 0 and kind != EventKind.DIED and not alive[agent]:
            assert day <= death_day[agent]


@pytest.mark.slow
def test_vital_rates_match_inputs_pooled():
    births = deaths = marriages = divorces = person_years = 0.0
    spec = None
    for seed in (1, 2, 3):
        w = generate_world(RunConfig(n_agents=10_000, countries=["USA"], seed=seed, start_date="2026-10-13"))
        spec = w.countries[0]
        rec = run_in_memory(w, 365)
        k = np.array(rec.events.table().column("kind").to_pylist())
        births += (k == EventKind.BORN).sum()
        deaths += (k == EventKind.DIED).sum()
        marriages += (k == EventKind.MARRIED).sum()
        divorces += (k == EventKind.DIVORCED).sum()
        person_years += 10_000
    assert spec is not None and spec.vital is not None
    targets = {
        "births": (births, spec.indicators["crude_birth_rate_per_1000"]),
        "deaths": (deaths, spec.indicators["crude_death_rate_per_1000"]),
        "marriages": (marriages, spec.vital.marriage_rate_per_1000),
        "divorces": (divorces, spec.vital.divorce_rate_per_1000),
    }
    for name, (count, target) in targets.items():
        rate = 1000 * count / person_years
        assert abs(rate - target) <= max(0.25 * target, 2.0), (name, rate, target)
