"""Insert notable public figures from the rosters as agents.

Notables are full agents (they can be possessed by a human in 1c) but have population weight 0,
belong to no household and are exempt from random life events.  Their influence travels through
media pushes and through the parasocial trust every agent holds in them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jahan.model.enums import N_TOPICS, EmpStatus, InfoTier, Sex
from jahan.model.notables import load_roster
from jahan.model.schema import MAX_LANGS
from jahan.model.world import ParasocialStore, World

DEFAULT_AGE = 55.0


def add_notables(world: World, data_dir: Path) -> int:
    start = world.config.start_date
    added = 0
    value_order = ["self", "money", "culture", "freedom", "family", "security"]
    for c, spec in enumerate(world.countries):
        roster = load_roster(data_dir, spec.iso3)
        if roster is None:
            continue
        main_tz = max(spec.time_zones.rows, key=lambda r: r.share).offset_min
        for e in roster.entries:
            age = (start.year - e.birth_year) if e.birth_year else DEFAULT_AGE
            langs = [world.lang_id(x) for x in (e.languages or spec.languages_official)][:MAX_LANGS]
            lang = np.full(MAX_LANGS, -1, np.int16)
            lang[: len(langs)] = langs
            values = np.array([e.values.get(k, 0.0) for k in value_order], np.float32)
            agent_id = int(world.alloc_ids("agent", 1)[0])
            world.pop.append(
                1,
                id=agent_id,
                alive=True,
                country=c,
                birth_country=c,
                location_country=c,
                household_id=-1,
                pop_weight=0.0,
                sex=int(Sex.F if e.sex == "F" else Sex.M),
                birth_day=-int(age * 365.25),
                marital=0,
                values=values[None, :],
                interests=np.full((1, N_TOPICS), 1.0 / N_TOPICS, np.float32),
                education=7,
                emp_status=int(EmpStatus.EMPLOYED),
                occupation=1,
                info_tier=int(InfoTier.SPECIALIST),
                internet=True,
                discernment=0.8,
                lang=lang[None, :],
                lang_prof=np.where(lang >= 0, 1.0, 0.0).astype(np.float32)[None, :],
                tz_offset_min=main_tz,
                home_tz_offset_min=main_tz,
                global_rep=float(e.approval.value),
                visibility=float(e.visibility),
                is_notable=True,
                role_id=len(world.notables),
            )
            entry = e.model_dump(mode="json")
            entry.update({"agent_id": agent_id, "country": c, "as_of": roster.as_of.isoformat()})
            world.notables.append(entry)
            added += 1
    world.ptrust = ParasocialStore(world.pop.capacity, len(world.notables) + ParasocialStore.RESERVED)
    return added


def notable_column(world: World, agent_id: int) -> int:
    """Parasocial-trust column of a notable agent, or -1."""
    row = world.agent_row(agent_id)
    if not world.pop["is_notable"][row]:
        return -1
    return int(world.pop["role_id"][row])
