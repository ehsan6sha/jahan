import numpy as np

from jahan.gen.sampler import sigmoid
from jahan.sim.trust import apply_outcomes, effective_trust


def _pick_edge(w):
    e = w.edges
    r = int(np.flatnonzero(e["kind"] == 4)[0])  # a friend tie
    return r, int(e["src"][r]), int(e["dst"][r])


def test_prior_only_trust(small_world_factory):
    w = small_world_factory(300)
    p = w.config.params
    t = effective_trust(w, slice(None), p)
    e = w.edges
    expected = sigmoid(np.log(e["prior_p"] / (1 - e["prior_p"])) + e["halo"])
    assert np.allclose(t, expected, atol=1e-5)


def test_good_and_bad_outcomes_with_negativity_bias(small_world_factory):
    p = small_world_factory(300).config.params
    w_good = small_world_factory(300)
    w_bad = small_world_factory(300)
    r, a, b = _pick_edge(w_good)
    t0 = effective_trust(w_good, np.array([r]), p)[0]
    args = (np.array([a]), np.array([b]), np.array([0.8]))
    apply_outcomes(w_good, *args, np.array([0.9]), p)  # advised positive, outcome good
    apply_outcomes(w_bad, *args, np.array([0.1]), p)  # advised positive, outcome bad
    up = effective_trust(w_good, np.array([r]), p)[0] - t0
    down = t0 - effective_trust(w_bad, np.array([r]), p)[0]
    assert up > 0 and down > 0
    assert down > up  # bad outcomes weigh more


def test_halo_fades_with_evidence(small_world_factory):
    w = small_world_factory(300)
    p = w.config.params
    r, a, b = _pick_edge(w)
    w.edges["halo"][r] = 1.0
    with_halo = effective_trust(w, np.array([r]), p)[0]
    w.edges["n_out"][r] = 30
    assert effective_trust(w, np.array([r]), p)[0] < with_halo


def test_evidence_decays_toward_prior(small_world_factory):
    w = small_world_factory(300)
    p = w.config.params
    r, a, b = _pick_edge(w)
    w.edges["beta"][r] = 10.0
    w.edges["last_update_day"][r] = 0
    w.day = 2000
    apply_outcomes(w, np.array([a]), np.array([b]), np.array([0.8]), np.array([0.8]), p)
    assert w.edges["beta"][r] < 1.0  # old negative evidence has faded


def test_missing_edge_and_media_sources_ignored(small_world_factory):
    w = small_world_factory(300)
    p = w.config.params
    assert apply_outcomes(w, np.array([0]), np.array([-3]), np.array([0.8]), np.array([0.9]), p) == 0
