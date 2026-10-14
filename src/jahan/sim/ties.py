"""Add, upgrade and retire social ties during a run (births, marriages, job changes, divorces)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jahan.gen.graph_gen import BASE_TRUST, directed_edge_columns, kind_strength
from jahan.model.enums import EdgeKind
from jahan.model.world import World


def _people_global(world: World) -> dict:
    pop = world.pop
    return {
        "global_rep": pop["global_rep"],
        "aggression": pop["aggression"],
        "education": pop["education"],
        "religion": pop["religion"],
        "attractiveness_z": pop["attractiveness_z"],
        "age_at_creation": world.ages(),
    }


def add_ties(world: World, a: NDArray[np.int64], b: NDArray[np.int64], kind: EdgeKind) -> int:
    """Create undirected ties (two directed edges each) for pairs that are not yet connected.

    Pairs that already have an edge are upgraded if ``kind`` is closer (household > kin > friend ...).
    Returns the number of new ties.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    keep = a != b
    a, b = a[keep], b[keep]
    if a.size == 0:
        return 0
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pair = np.unique(lo * (1 << 32) + hi)
    lo, hi = pair >> 32, pair & ((1 << 32) - 1)
    existing = world.find_edges(lo, hi)
    has = existing >= 0
    if has.any():
        upgrade_ties(world, lo[has], hi[has], kind)
    lo, hi = lo[~has], hi[~has]
    if lo.size == 0:
        return 0
    cols = directed_edge_columns(
        lo,
        hi,
        np.full(lo.shape[0], int(kind), np.int64),
        _people_global(world),
        world.counters["edge"],
        world.config.params,
        day=world.day,
    )
    world.append_edges(cols)
    world.counters["edge"] += int(cols["uid"].shape[0])
    return int(lo.size)


_PRIORITY = [
    EdgeKind.HOUSEHOLD,
    EdgeKind.KIN,
    EdgeKind.FRIEND,
    EdgeKind.COLLEAGUE,
    EdgeKind.CLASSMATE,
    EdgeKind.CROSS_BORDER,
    EdgeKind.ACQUAINTANCE,
]


def _both_directions(world: World, a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    r = np.concatenate([world.find_edges(a, b), world.find_edges(b, a)])
    return r[r >= 0]


def upgrade_ties(world: World, a: NDArray[np.int64], b: NDArray[np.int64], kind: EdgeKind) -> None:
    """Set the tie kind to ``kind`` where it is closer than the current kind; reactivate the tie."""
    rows = _both_directions(world, a, b)
    if rows.size == 0:
        return
    e = world.edges
    rank = {k: i for i, k in enumerate(_PRIORITY)}
    cur = np.array([rank[EdgeKind(int(k))] for k in e["kind"][rows]])
    better = cur > rank[kind]
    r = rows[better]
    e["kind"][r] = int(kind)
    e["strength"][r] = kind_strength(world.config.params)[kind]
    e["prior_p"][r] = np.maximum(e["prior_p"][r], BASE_TRUST[kind])
    e["active"][rows] = True


def set_tie(
    world: World, a: NDArray[np.int64], b: NDArray[np.int64], kind: EdgeKind, beta_penalty: float = 0.0
) -> None:
    """Force the tie kind (e.g. spouse -> acquaintance after divorce) and optionally add negative evidence."""
    rows = _both_directions(world, a, b)
    if rows.size == 0:
        return
    e = world.edges
    e["kind"][rows] = int(kind)
    e["strength"][rows] = kind_strength(world.config.params)[kind]
    if beta_penalty:
        e["beta"][rows] += np.float32(beta_penalty)


def retire_agent_ties(world: World, agents: NDArray[np.int64]) -> None:
    e = world.edges
    dead = np.isin(e["src"], agents) | np.isin(e["dst"], agents)
    e["active"][dead] = False
