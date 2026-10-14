import datetime as dt
from pathlib import Path

import numpy as np

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind, OriginKind
from jahan.model.notables import SIMULATED_LABEL, can_do, load_roles, load_roster, validate_rosters
from jahan.query.views import agent_card
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.sim.trust import parasocial_trust
from jahan.store.eventlog import EventLog

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]
EX = Path(__file__).resolve().parents[2] / "examples" / "scenarios"


def test_rosters_validate_and_cite_sources():
    assert validate_rosters(paths.data_dir(), M1, dt.date(2026, 10, 13)) == []
    # a world starting before the rosters were verified is rejected (anachronism)
    assert validate_rosters(paths.data_dir(), ["USA"], dt.date(2024, 1, 1))
    for iso in M1:
        roster = load_roster(paths.data_dir(), iso)
        assert roster is not None and roster.entries
        for e in roster.entries:
            assert e.sources and all(s.startswith("http") for s in e.sources)
            assert all(s.source.startswith("http") for s in e.stances)


def test_role_permissions():
    roles = load_roles(paths.data_dir())
    assert can_do(roles, "central_bank_governor", "set_interest_rate")
    assert not can_do(roles, "head_of_state", "set_interest_rate")
    assert not can_do(roles, "opinion_leader", "sanctions")


def test_notables_are_agents_outside_the_population_statistics():
    w = generate_world(RunConfig(n_agents=1500, countries=["USA", "IRN"], seed=3, start_date="2026-10-13"))
    notable = np.flatnonzero(w.pop["is_notable"])
    assert notable.size == len(w.notables) == 9  # 4 USA + 5 IRN roster entries
    assert (w.pop["pop_weight"][notable] == 0).all()
    assert (w.pop["household_id"][notable] == -1).all()
    names = {w.notables[int(w.pop["role_id"][i])]["holder_name"] for i in notable}
    assert {"Donald Trump", "Mojtaba Khamenei", "Masoud Pezeshkian", "Elon Musk"} <= names
    pres = next(e for e in w.notables if e["key"] == "USA.head_of_state")
    card = agent_card(w, pres["agent_id"])
    assert "Donald Trump" in card and SIMULATED_LABEL in card and "President of the United States" in card
    rec = run_in_memory(w, 120)
    t = rec.events.table().to_pydict()
    life = {EventKind.DIED, EventKind.MARRIED, EventKind.DIVORCED, EventKind.JOB_CHANGED, EventKind.STATUS_CHANGED}
    assert not any(k in life and a in set(notable.tolist()) for k, a in zip(t["kind"], t["agent"], strict=True))


def test_parasocial_trust_reflects_approval_alignment_and_nationality():
    w = generate_world(RunConfig(n_agents=3000, countries=["USA", "IRN"], seed=8, start_date="2026-10-13"))
    params = w.config.params
    pres = next(i for i, e in enumerate(w.notables) if e["key"] == "USA.head_of_state")
    rows = np.flatnonzero(~w.pop["is_notable"])
    t = parasocial_trust(w, rows, pres, params)
    usa = w.pop["country"][rows] == 0
    assert 0.2 < t[usa].mean() < 0.5  # around the ~35% approval prior
    assert t[~usa].mean() < t[usa].mean()  # foreign-leader discount


def test_statement_reaches_much_wider_than_an_ordinary_post():
    w = generate_world(RunConfig(n_agents=3000, countries=["USA"], seed=4, start_date="2026-10-13"))
    apply_scenario(w, Scenario.load(EX / "president_endorses.yaml"), EventLog(), 1)
    rec = run_in_memory(w, 20)
    items = np.flatnonzero(w.items["origin_kind"] == OriginKind.NOTABLE)
    assert items.size == 1
    created = [
        d
        for k, d in zip(
            rec.events.table().column("kind").to_pylist(), rec.events.table().column("detail").to_pylist(), strict=True
        )
        if k == EventKind.ITEM_CREATED
    ]
    assert any(d.startswith(SIMULATED_LABEL) and "Donald Trump" in d for d in created)
    slot = int(w.items["slot"][items[0]])
    aware = (w.beliefs.m["aware"][: w.pop.n, slot] == 1) & ~w.pop["is_notable"]
    assert aware.mean() > 0.15  # national broadcast reaches a large share within two weeks
    pres_id = int(w.items["origin_id"][items[0]])
    heard = w.beliefs.m["heard_from"][: w.pop.n, slot]
    assert (heard == pres_id).any()  # first exposures are attributed to the speaker, not the channel
