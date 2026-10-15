"""Daily aggregate metrics (long format: day, country, item, metric, value)."""

from __future__ import annotations

import numpy as np

from jahan.model.enums import EdgeKind
from jahan.model.world import World
from jahan.sim.params import SimParams
from jahan.sim.questions import record_odds
from jahan.sim.trust import effective_trust
from jahan.store.eventlog import MetricsLog

TRUST_EVERY = 7  # trust means are costlier; record weekly


def record(world: World, params: SimParams, stats: dict[str, int], out: MetricsLog) -> None:
    day = world.day
    pop = world.pop
    n = pop.n
    alive = pop["alive"]
    w = pop["pop_weight"].astype(np.float64) * alive
    loc = pop["location_country"]
    adult = world.ages() >= params.min_adopt_age
    for name, v in stats.items():
        out.add(day, -1, -1, name, v)
    for key in ("gold_index", "us_stocks_index"):
        if key in world.macro:
            out.add(day, -1, -1, key, float(world.macro[key]))
    market = world.macro.get("market")
    if market is not None:  # the two parts of the US stock index (worlds made since people trade)
        last = market["last"]
        out.add(day, -1, -1, "us_stocks_scripted_pct", float(last["scripted_pct"]))
        out.add(day, -1, -1, "us_stocks_people_pct", float(last["people_pct"]))
        out.add(day, -1, -1, "us_stocks_scripted_index", float(market["scripted_index"]))
        out.add(day, -1, -1, "us_stocks_people_index", float(market["people_index"]))
    bel = world.beliefs.m
    for c in range(world.n_countries):
        mc = loc == c
        wc = w * mc
        tot = wc.sum()
        out.add(day, c, -1, "alive", float((alive & mc).sum()))
        if tot <= 0:
            continue
        out.add(day, c, -1, "sentiment", float(np.average(pop["sentiment"], weights=wc)))
        fx = world.macro.get("fx")
        if fx:
            out.add(day, c, -1, "fx_local_per_usd", float(fx[c]))
        for slot in np.flatnonzero(world.beliefs.slot_item >= 0):
            item = int(world.beliefs.slot_item[slot])
            aware = bel["aware"][:n, slot] == 1
            adopted = bel["adopted_day"][:n, slot] >= 0
            out.add(day, c, item, "aware_share", float((wc * aware).sum() / tot))
            wa = wc * adult
            if wa.sum() > 0:
                out.add(day, c, item, "adopted_share_adults", float((wa * adopted).sum() / wa.sum()))
            out.add(day, c, item, "n_aware", float((aware & mc).sum()))
            out.add(day, c, item, "n_adopted", float((adopted & mc).sum()))
            m = aware & mc
            if m.any() and w[m].sum() > 0:  # public figures carry no population weight
                out.add(day, c, item, "mean_belief_aware", float(np.average(bel["belief"][:n, slot][m], weights=w[m])))
    record_odds(world, out)
    if day % TRUST_EVERY == 0 and world.edges.n:
        t = effective_trust(world, slice(None), params)
        kind = world.edges["kind"]
        for k in EdgeKind:
            m = kind == k
            if m.any():
                out.add(day, -1, -1, f"trust_mean.{k.name.lower()}", float(t[m].mean()))
