import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind
from jahan.sim.runner import run_in_memory
from jahan.sim.travel import trips_per_person

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]


def test_trips_follow_departure_rates_and_return_home():
    w = generate_world(RunConfig(n_agents=10_000, countries=M1, seed=6, start_date="2026-10-13"))
    rates = trips_per_person(w)
    assert rates[M1.index("DEU")] > rates[M1.index("IND")]  # Germans travel far more per person
    rec = run_in_memory(w, 120)
    t = rec.events.table().to_pydict()
    kind = np.array(t["kind"])
    starts = kind == EventKind.TRAVEL_START
    ends = kind == EventKind.TRAVEL_END
    assert starts.sum() > 20 and ends.sum() > 0
    home = np.array(t["country"])[starts]
    dest = np.array(t["other"])[starts]
    assert (home != dest).all()
    pop = w.pop
    away = pop["trip_end_day"] >= 0
    # travellers are where they say they are, in the destination's time zone; everyone else is home
    assert (pop["location_country"][away] != pop["country"][away]).all()
    assert (pop["location_country"][~away] == pop["country"][~away]).all()
    assert (pop["tz_offset_min"][~away] == pop["home_tz_offset_min"][~away]).all()
    # per-capita trips are higher from Germany than from India
    agents = np.array(t["agent"])[starts]
    by_country = np.bincount(pop["country"][agents], minlength=len(M1)) / np.bincount(pop["country"], minlength=len(M1))
    assert by_country[M1.index("DEU")] > by_country[M1.index("IND")]


def test_migrants_visit_their_country_of_birth():
    w = generate_world(RunConfig(n_agents=10_000, countries=M1, seed=7, start_date="2026-10-13"))
    rec = run_in_memory(w, 365)
    t = rec.events.table().to_pydict()
    pop = w.pop
    visits = [(a, d) for k, a, d in zip(t["kind"], t["agent"], t["other"], strict=True) if k == EventKind.TRAVEL_START]
    migrant_visits = [(a, d) for a, d in visits if pop["birth_country"][a] != pop["country"][a]]
    if migrant_visits:
        to_origin = np.mean([pop["birth_country"][a] == d for a, d in migrant_visits])
        assert to_origin > 0.4
