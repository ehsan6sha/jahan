"""One simulated day.  The phase order is part of the reproducibility contract.

0 scheduled world events      scenario news/macro steps due today, auto-headlines, item expiry;
                              watched questions past their resolve date close, and a quiet day passes for the
                              others (odds of "by a date" questions fall, "holds through" ones rise);
                              trips start and end (location and time zone switch)
1 availability masks          sleep, work, school by local time zone
1b actions                    messages and posts by controlled agents and public figures are delivered
                              (receivers asleep after the send hour get them the next day)
2 media exposure
3 contact sampling
4 word-of-mouth diffusion; claims about products move views of those products
4c event odds                 people who know a watched question update it on stories that bear on it;
                              with everyday activity, traders who met talk about it
4b reposts                    convinced, online people repost stories on social media (from tomorrow)
5 mood                        economic sentiment drifts toward the news each agent knows (with everyday
                              activity: also lower while unemployed, and pulled toward people met today)
5b markets                    holders of stock trade on the news they know and move the index (trading days);
                              with everyday activity, every country's market, acted on as people look at
                              their accounts, plus cash-outs that differ from normal (sim/economy.py)
6 decisions and realised outcomes -> trust updates
7 life events                 deaths, births, marriages, divorces, labour moves, birthday transitions
8 metrics
9 advance the day
"""

from __future__ import annotations

import numpy as np

from jahan.actors.engine import deliver_actions
from jahan.model.world import World
from jahan.sim import economy, market, metrics, questions
from jahan.sim.availability import compute_masks
from jahan.sim.contacts import sample_contacts
from jahan.sim.decisions import decide_and_realise
from jahan.sim.diffusion import diffuse
from jahan.sim.lifeevents import run_life_events
from jahan.sim.media import media_exposure
from jahan.sim.social import reposts
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
    everyday = economy.enabled(world)
    opened = market.open_day(world)
    eopened = economy.open_day(world)
    stats.update(apply_due(world, log))
    stats.update(questions.resolve_due(world, log))
    odds_parts: dict[str, dict[str, float]] = {"time": questions.pass_time(world)}
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
    odds_parts["news"] = {}
    odds_moved = questions.apply_bearings(world, params, log, odds_parts["news"])
    if odds_moved:
        stats["odds_beliefs_moved"] = odds_moved
    odds_parts["talk"] = questions.traders_talk(world, contacted, params)
    stats.update(reposts(world, params, log))
    news = update_sentiment(world, params, contacted)
    if everyday:
        stats.update(economy.close_day(world, params, log, eopened, news))
    else:
        stats.update(market.close_day(world, params, log, opened))
    stats.update(decide_and_realise(world, params, log))
    if params.life_events:
        stats.update(run_life_events(world, params, log))
    metrics.record(world, params, stats, mlog, odds_parts)
    world.day += 1
    return stats
