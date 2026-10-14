"""One simulated day.  The phase order is part of the reproducibility contract.

0 scheduled world events      scenario news/macro steps due today, auto-headlines, item expiry;
                              trips start and end (location and time zone switch)
1 availability masks          sleep, work, school by local time zone
1b actions                    messages and posts by controlled agents and public figures are delivered
                              (receivers asleep after the send hour get them the next day)
2 media exposure
3 contact sampling
4 word-of-mouth diffusion; claims about products move views of those products
5 mood                        economic sentiment drifts toward the news each agent knows
6 decisions and realised outcomes -> trust updates
7 life events                 deaths, births, marriages, divorces, labour moves, birthday transitions
8 metrics
9 advance the day
"""

from __future__ import annotations

import numpy as np

from jahan.actors.engine import deliver_actions
from jahan.model.world import World
from jahan.sim import metrics
from jahan.sim.availability import compute_masks
from jahan.sim.contacts import sample_contacts
from jahan.sim.decisions import decide_and_realise
from jahan.sim.diffusion import diffuse
from jahan.sim.lifeevents import run_life_events
from jahan.sim.media import media_exposure
from jahan.sim.travel import travel
from jahan.sim.world_events import apply_due, propagate_entity_claims, update_sentiment
from jahan.store.eventlog import EventLog, MetricsLog


def run_day(
    world: World,
    log: EventLog,
    mlog: MetricsLog,
    trace: set[int] | None = None,
    trace_rows: list[dict] | None = None,
) -> dict[str, int]:
    params = world.config.params
    stats: dict[str, int] = {}
    stats.update(apply_due(world, log))
    if params.travel:
        stats.update(travel(world, params, log))
    masks = compute_masks(world, params)
    stats.update(deliver_actions(world, masks, params, log))
    stats.update(media_exposure(world, masks, params, log))
    contacted = sample_contacts(world, masks, params)
    stats["contacts"] = int(contacted.sum()) // 2
    if trace and trace_rows is not None:
        e = world.edges
        for r in np.flatnonzero(contacted & np.isin(e["src"], list(trace))):
            trace_rows.append(
                {
                    "type": "contact",
                    "day": world.day,
                    "agent": int(e["src"][r]),
                    "other": int(e["dst"][r]),
                    "kind": int(e["kind"][r]),
                }
            )
    stats.update(diffuse(world, contacted, params, log, trace, trace_rows))
    stats["claims_applied"] = propagate_entity_claims(world, params)
    update_sentiment(world, params)
    stats.update(decide_and_realise(world, params, log))
    if params.life_events:
        stats.update(run_life_events(world, params, log))
    metrics.record(world, params, stats, mlog)
    world.day += 1
    return stats
