"""Political camps: sampling, inheritance, per-camp trust in public figures, and how they read."""

import numpy as np
import pytest

from jahan import paths
from jahan.actors import engine
from jahan.actors.actions import CreateAgent
from jahan.config import RunConfig
from jahan.gen.politics_gen import camp_of, inherit_alignment, quantile_alignment
from jahan.gen.registry import validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.country import CAMP_TAU, load_country
from jahan.model.enums import Marital, OriginKind
from jahan.model.notables import (
    CAMP_TEMPLATE,
    CampApproval,
    camp_priors,
    camp_shares,
    camp_template,
    load_roster,
)
from jahan.query.persona import mind_snapshot
from jahan.query.views import agent_card
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario, Statement
from jahan.sim.lifeevents import _give_birth
from jahan.sim.runner import run_in_memory
from jahan.sim.trust import parasocial_prior
from jahan.store.eventlog import EventLog
from jahan.store.snapshot import load_snapshot, save_snapshot

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]


def world(countries=("USA",), n=3000, seed=8):
    return generate_world(RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13"))


def adults(w, c):
    return np.flatnonzero(w.pop["alive"] & ~w.pop["is_notable"] & (w.pop["country"] == c) & (w.ages() >= 18))


def test_quantile_map_gives_exact_camp_masses():
    pol = load_country(paths.data_dir(), "USA").politics
    u = (np.arange(100_000) + 0.5) / 100_000
    camps = camp_of(quantile_alignment(u, pol))
    assert abs((camps == 1).mean() - pol.camps["pos"]) < 1e-3
    assert abs((camps == 0).mean() - pol.camps["neutral"]) < 1e-3
    assert abs((camps == -1).mean() - pol.camps["neg"]) < 1e-3
    a = quantile_alignment(u, pol)
    assert a.min() >= -1 and a.max() <= 1 and (np.diff(a) >= 0).all()  # monotone
    # band edges: exact boundary draws stay in their own camp after the float32 cast
    edges = np.array([0.0, pol.camps["neg"], pol.camps["neg"] + pol.camps["neutral"]])
    assert camp_of(quantile_alignment(edges, pol)).tolist() == [-1, 0, 1]
    for camp in (-1, 0, 1):
        kids = inherit_alignment(np.full(3, 0.9 * camp), np.zeros(3), np.array([0.0, 0.5, 1.0 - 1e-12]), pol)
        assert (camp_of(kids) == camp).all()


def test_generated_adults_match_the_sourced_camp_shares():
    w = world(("USA", "IND"), n=4000, seed=5)
    for c, iso in enumerate(("USA", "IND")):
        pol = w.countries[c].politics
        camps = camp_of(w.pop["alignment"][adults(w, c)])
        assert abs((camps == 1).mean() - pol.camps["pos"]) < 0.01, iso
        assert abs((camps == -1).mean() - pol.camps["neg"]) < 0.01, iso
    # sourced shifts order groups: Tamil speakers sit below Hindi speakers on the NDA axis (Ipsos IndiaBus)
    spec = w.countries[1]
    rows = adults(w, 1)
    eth = w.pop["ethnicity"][rows]
    a = w.pop["alignment"][rows]
    hindi, tamil = spec.ethnicity_codes().index("hindi"), spec.ethnicity_codes().index("tamil")
    assert a[eth == tamil].mean() < a[eth == hindi].mean() - 0.3


def test_alignment_is_deterministic_and_seeded():
    a = world(n=1200, seed=4).pop["alignment"]
    b = world(n=1200, seed=4).pop["alignment"]
    c = world(n=1200, seed=5).pop["alignment"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a[:1000], c[:1000])


def test_children_keep_the_parents_camp_about_70_percent_of_the_time():
    pol = load_country(paths.data_dir(), "USA").politics
    rng = np.random.default_rng(1)
    parent = quantile_alignment(rng.random(6000), pol)
    child = inherit_alignment(parent, rng.random(6000), rng.random(6000), pol)
    same = (camp_of(child) == camp_of(parent)).mean()
    # 0.70 inherit + 0.30 fresh draws that land in the same camp by chance
    expected = 0.70 + 0.30 * sum(v**2 for v in pol.camps.values())
    assert abs(same - expected) < 0.03
    assert inherit_alignment(parent[:5], rng.random(5), rng.random(5), None).tolist() == [0.0] * 5


def test_births_inherit_the_mothers_camp():
    w = world(n=6000, seed=2)
    pop = w.pop
    age = w.ages()
    mothers = np.flatnonzero(
        (pop["sex"] == 0) & (pop["marital"] == Marital.MARRIED) & (age > 20) & (age < 42) & ~pop["is_notable"]
    )[:400]
    assert mothers.size >= 300
    log = EventLog()
    kids = [_give_birth(w, int(m), log) for m in mothers]
    same = (camp_of(w.pop["alignment"][kids]) == camp_of(w.pop["alignment"][mothers])).mean()
    pol = w.countries[0].politics
    assert abs(same - (0.70 + 0.30 * sum(v**2 for v in pol.camps.values()))) < 0.08


def test_camp_priors_reproduce_every_topline_and_reject_impossible_data():
    for iso in M1:
        spec = load_country(paths.data_dir(), iso)
        for e in load_roster(paths.data_dir(), iso).entries:
            shares = camp_shares(spec.politics, e.alignment)
            prior, _, status = camp_priors(e.approval.value, shares, e.approval_by_camp, camp_template(spec))
            if shares is None:
                assert status == "non_partisan"
                continue
            mean = sum(shares[k] * prior[k] for k in prior)
            assert abs(mean - e.approval.value) < (0.02 if status == "sourced" else 1e-6), e.key
            if spec.politics.placeholder:
                assert status in ("placeholder", "sourced", "sourced_shifted", "derived"), e.key
    shares = {"aligned": 0.3, "neutral": 0.4, "opposed": 0.3}
    with pytest.raises(ValueError):
        camp_priors(0.10, shares, CampApproval(aligned=0.9, source="x"))  # the rest would need to be negative
    prior, _, status = camp_priors(0.10, shares, CampApproval(aligned=0.25, source="x"))
    assert status == "derived" and prior["neutral"] == prior["opposed"]
    usa, nga = (camp_template(load_country(paths.data_dir(), iso)) for iso in ("USA", "NGA"))
    assert usa is not None and nga is not None
    assert usa == pytest.approx(CAMP_TEMPLATE)  # full gap where it was measured
    assert usa["aligned"] > nga["aligned"] > nga["neutral"] > nga["opposed"] > usa["opposed"]  # half elsewhere
    assert camp_template(load_country(paths.data_dir(), "IRN")) is None  # never on placeholder camps
    prior, _, status = camp_priors(0.5, shares, None, nga)
    assert status == "template" and prior["aligned"] > prior["neutral"] > prior["opposed"]
    assert camp_priors(0.5, shares, None)[2] == "placeholder"


def test_trust_in_the_president_splits_by_camp_and_averages_to_approval():
    w = world(("USA", "IRN"), n=4000, seed=8)
    col = next(i for i, e in enumerate(w.notables) if e["key"] == "USA.head_of_state")
    entry = w.notables[col]
    assert entry["camp_status"] == "sourced_shifted"
    rows = adults(w, 0)
    p = parasocial_prior(w, rows, col)
    camps = camp_of(w.pop["alignment"][rows])
    assert p[camps == 1].mean() > 0.7
    assert p[camps == -1].mean() < 0.15
    weights = w.pop["pop_weight"][rows].astype(np.float64)
    assert abs(np.average(p, weights=weights) - entry["approval"]["value"]) < 0.03
    # foreign residents still get the topline with the foreign discount, whatever their own lean
    foreign = adults(w, 1)
    assert parasocial_prior(w, foreign, col).mean() < entry["approval"]["value"]


def test_every_partisan_figure_averages_to_its_topline():
    w = world(tuple(M1), n=8000, seed=6)
    for col, e in enumerate(w.notables):
        rows = adults(w, int(e["country"]))
        p = parasocial_prior(w, rows, col)
        mean = np.average(p, weights=w.pop["pop_weight"][rows].astype(np.float64))
        assert abs(mean - e["approval"]["value"]) < 0.02, (e["key"], mean, e["approval"]["value"])


def test_opponents_hear_a_statement_but_stay_unconvinced():
    w = world(n=3000, seed=4)
    apply_scenario(
        w,
        Scenario(
            prompt="t",
            events=[
                Statement(speaker="USA.head_of_state", headline="The economy is booming", topic="economy", valence=1.0)
            ],
        ),
        EventLog(),
        1,
    )
    run_in_memory(w, 14)
    item = int(np.flatnonzero(w.items["origin_kind"] == OriginKind.NOTABLE)[0])
    slot = int(w.items["slot"][item])
    rows = adults(w, 0)
    aware = w.beliefs.m["aware"][rows, slot] == 1
    belief = w.beliefs.m["belief"][rows, slot]
    camps = camp_of(w.pop["alignment"][rows])
    aligned = belief[aware & (camps == 1)].mean()
    neutral = belief[aware & (camps == 0)].mean()
    opposed = belief[aware & (camps == -1)].mean()
    # measured 0.76 / 0.62 / 0.58 at seed 4: beliefs move only part of the way in two weeks, word of mouth
    # mixes the camps and discerning people drift toward the story's truth (0.75 at accuracy 0.5)
    assert aligned > 0.7
    assert aligned > neutral > opposed
    assert opposed < aligned - 0.12


def test_cards_describe_the_lean_in_the_countrys_own_terms(tmp_path):
    w = world(("USA", "IRN"), n=1500, seed=3)
    new = engine.create_agent(w, CreateAgent(country="USA", alignment=0.9), EventLog(), 1)
    assert "Politics: firmly in the Republican camp" in agent_card(w, new)
    assert "firmly in the Republican camp" in mind_snapshot(w, new)
    iran = int(adults(w, 1)[0])
    card = agent_card(w, iran)
    assert "Politics:" in card and "establishment" in card and "placeholder" in card
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    fig = agent_card(w, pres["agent_id"])
    assert "Starting trust by camp" in fig and "Gallup" in fig
    snap = save_snapshot(w, tmp_path / "s" / "day0")
    back = load_snapshot(snap)
    assert np.array_equal(back.pop["alignment"], w.pop["alignment"])
    assert back.notables[w.notables.index(pres)]["camp_prior"] == pres["camp_prior"]


def test_strict_mode_reports_placeholder_camps():
    problems = validate_data(paths.data_dir(), ["IRN", "CHN", "BRA"], strict=True)
    assert any("IRN: political camp shares are a placeholder" in p for p in problems)
    assert any("CHN: political camp shares are a placeholder" in p for p in problems)
    assert any("BRA.head_of_state" in p and "borrowed from US polling" in p for p in problems)


def test_central_bankers_are_non_partisan():
    for iso in M1:
        for e in load_roster(paths.data_dir(), iso).entries:
            if e.role == "central_bank_governor":
                assert abs(e.alignment) < CAMP_TAU, e.key
