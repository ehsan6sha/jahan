"""Build a World from a RunConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.allocate import allocate
from jahan.gen.diaspora import add_diaspora, load_stock
from jahan.gen.graph_gen import directed_edge_columns, undirected_ties
from jahan.gen.household_gen import assemble_households
from jahan.gen.media_gen import channel_rows
from jahan.gen.names import load_names
from jahan.gen.notables_gen import add_notables
from jahan.gen.politics_gen import sample_alignment
from jahan.gen.population_gen import generate_people
from jahan.gen.workplace_gen import assign_workplaces
from jahan.model.country import load_country
from jahan.model.investors import load_investors
from jahan.model.markets import load_markets
from jahan.model.notables import load_roles
from jahan.model.world import BeliefStore, World
from jahan.rng import MASK64

AGENT_HEADROOM = 1.5
EDGE_HEADROOM = 1.3


def load_languages(data_dir: Path) -> list[str]:
    with (data_dir / "global" / "languages.yaml").open(encoding="utf-8") as f:
        return list(yaml.safe_load(f)["languages"])


def load_world_shell(config: RunConfig, data_dir: Path | None = None) -> World:
    dd = data_dir or paths.data_dir()
    specs = [load_country(dd, c) for c in config.countries]
    names = [load_names(dd, s.names.file) for s in specs]
    return World(
        config=config,
        countries=specs,
        names=names,
        languages=load_languages(dd),
        sim_seed=config.seed,
        roles=load_roles(dd),
        markets=load_markets(dd),
    )


def generate_world(config: RunConfig, data_dir: Path | None = None) -> World:
    world = load_world_shell(config, data_dir)
    specs = world.countries
    alloc = allocate(config.n_agents, [s.population.value for s in specs], config.allocation)
    world.pop.reserve(int(config.n_agents * AGENT_HEADROOM) + 16)

    tie_parts: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = []
    ages_at_creation: list[NDArray[np.float64]] = []
    for cidx, (spec, n_c) in enumerate(zip(specs, alloc, strict=True)):
        if n_c == 0:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([config.seed & MASK64, cidx, 1]))
        people = generate_people(spec, n_c, rng, world.languages, world.names[cidx])
        hres = assemble_households(people, spec, rng, world.names[cidx].surname_adoption)
        wp_local, wp_rows = assign_workplaces(people, rng)
        wp_kind = np.array([r[0] for r in wp_rows], dtype=np.int64)
        a, b, k = undirected_ties(people, hres.household, wp_local, wp_kind, rng)

        id0 = world.counters["agent"]
        ids = world.alloc_ids("agent", n_c)
        hh0 = world.households.n
        wp0 = world.workplaces.n

        def glob(local: NDArray[np.int64], offset: int) -> NDArray[np.int64]:
            return np.where(local >= 0, local + offset, -1)

        age = people.pop("age")
        ages_at_creation.append(age)
        n_children = np.bincount(hres.mother[hres.mother >= 0], minlength=n_c) + np.bincount(
            hres.father[hres.father >= 0], minlength=n_c
        )
        world.pop.append(
            n_c,
            id=ids,
            country=cidx,
            birth_country=cidx,
            location_country=cidx,
            home_tz_offset_min=people["tz_offset_min"],
            household_id=hres.household + hh0,
            birth_day=-np.floor(age * 365.25).astype(np.int32),
            marital=hres.marital,
            spouse_id=glob(hres.spouse, id0),
            mother_id=glob(hres.mother, id0),
            father_id=glob(hres.father, id0),
            n_children=np.minimum(n_children, 127).astype(np.int8),
            workplace_id=glob(wp_local, wp0),
            pop_weight=np.float32(spec.population.value / n_c),
            **people,
        )
        start = world.pop.n - n_c
        world.pop["alignment"][start:] = sample_alignment(spec, people, age, hres.mother, ids, config.seed)
        # households
        members = np.bincount(hres.household, minlength=hres.n_households)
        income = np.bincount(
            hres.household, weights=people["income_month"].astype(np.float64), minlength=hres.n_households
        )
        first_member = np.full(hres.n_households, -1, np.int64)
        first_member[hres.household[::-1]] = np.arange(n_c)[::-1]
        world.households.append(
            hres.n_households,
            id=np.arange(hh0, hh0 + hres.n_households, dtype=np.int32),
            country=cidx,
            region=people["region"][first_member],
            n_members=members.astype(np.int16),
            income_pooled=income.astype(np.float32),
        )
        world.counters["household"] = world.households.n
        if wp_rows:
            world.workplaces.append(
                len(wp_rows),
                id=np.arange(wp0, wp0 + len(wp_rows), dtype=np.int32),
                country=cidx,
                kind=wp_kind.astype(np.int8),
                region=np.array([r[1] for r in wp_rows], np.int16),
                size=np.array([r[2] for r in wp_rows], np.int32),
            )
        tie_parts.append((a + id0, b + id0, k))
        for row in channel_rows(spec, cidx, world.languages):
            name = row.pop("name")
            cid = world.channels.n
            world.channels.append(1, id=cid, **row)
            world.channel_names.append(name)

    people_global: dict[str, Any] = {
        c: world.pop[c] for c in ("global_rep", "aggression", "education", "religion", "attractiveness_z")
    }
    people_global["age_at_creation"] = np.concatenate(ages_at_creation) if ages_at_creation else np.zeros(0)
    total_ties = sum(p[0].shape[0] for p in tie_parts)
    world.edges.reserve(int(2 * total_ties * EDGE_HEADROOM) + 16)
    for a, b, k in tie_parts:
        cols = directed_edge_columns(a, b, k, people_global, world.counters["edge"], config.params)
        world.edges.append(cols["uid"].shape[0], **cols)
        world.counters["edge"] += int(cols["uid"].shape[0])

    dd = data_dir or paths.data_dir()
    if config.diaspora_file and len(specs) > 1 and (dd / config.diaspora_file).exists():
        rng = np.random.default_rng(np.random.SeedSequence([config.seed & MASK64, 0xD1A5, 1]))
        add_diaspora(world, load_stock(dd / config.diaspora_file), rng)
    if config.notables:
        add_notables(world, dd)

    world.beliefs = BeliefStore(world.pop.capacity, config.max_slots)
    world.ensure_product_arrays()
    investors = load_investors(dd)
    world.init_macro(None if investors is None else investors.to_world())
    world.day = 0
    return world
