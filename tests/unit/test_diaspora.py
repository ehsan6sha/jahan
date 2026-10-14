import numpy as np

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EdgeKind
from jahan.sim.availability import compute_masks
from jahan.sim.contacts import sample_contacts

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]


def test_diaspora_links_countries():
    w = generate_world(RunConfig(n_agents=10_000, countries=M1, seed=5, start_date="2026-10-13"))
    pop = w.pop
    migrants = np.flatnonzero(pop["birth_country"] != pop["country"])
    assert migrants.size > 0
    usa, ind = M1.index("USA"), M1.index("IND")
    indian_in_us = migrants[(pop["country"][migrants] == usa) & (pop["birth_country"][migrants] == ind)]
    assert indian_in_us.size > 0
    hi = w.lang_id("hi")
    assert (pop["lang"][indian_in_us, 0] == hi).all()  # origin language first
    assert (pop["lang"][indian_in_us] == w.lang_id("en")).any(axis=1).all()  # destination language kept
    e = w.edges
    xb = e["kind"] == EdgeKind.CROSS_BORDER
    assert xb.any()
    assert (pop["country"][e["src"][xb]] != pop["country"][e["dst"][xb]]).all()
    # cross-border ties carry contact over a month despite time-zone differences
    hits = 0
    for day in range(30):
        w.day = day
        hits += int((sample_contacts(w, compute_masks(w, w.config.params), w.config.params) & xb).sum())
    assert hits > 0
