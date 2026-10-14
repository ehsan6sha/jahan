"""First-generation migrants and their ties to the origin country (UN DESA migrant stock).

For every destination D and origin O in the world, a number of D's residents proportional to
stock[D][O] / population(D) become O-born migrants: their birth country is O, their first
language becomes O's official language (D's official language is added as a second language),
and each gets 2-4 family ties to adults in O.  Residents who already speak O's language are
preferred, so e.g. Chinese-speaking Americans are the likeliest China-born migrants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from jahan.model.enums import EdgeKind
from jahan.model.schema import MAX_LANGS
from jahan.model.world import World
from jahan.sim.ties import add_ties


def load_stock(path: Path) -> dict[str, dict[str, int]]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["stock"]


def add_diaspora(world: World, stock: dict[str, dict[str, int]], rng: np.random.Generator) -> int:
    pop = world.pop
    codes = [c.iso3 for c in world.countries]
    age = world.ages()
    adult = (age >= 20) & (age < 75)
    migrants_total = 0
    pending_a: list[int] = []
    pending_b: list[int] = []
    for d_idx, dest in enumerate(codes):
        n_d = int((pop["country"] == d_idx).sum())
        pop_d = world.countries[d_idx].population.value
        dest_lang = world.lang_id(world.countries[d_idx].languages_official[0])
        for origin, count in sorted(stock.get(dest, {}).items()):
            if origin not in codes:
                continue
            o_idx = codes.index(origin)
            expected = n_d * count / pop_d
            k = int(np.floor(expected) + (rng.random() < expected - np.floor(expected)))
            if k == 0:
                continue
            o_lang = world.lang_id(world.countries[o_idx].languages_official[0])
            cand = np.flatnonzero(adult & (pop["country"] == d_idx) & (pop["birth_country"] == d_idx))
            if cand.size == 0:
                continue
            speaks = (pop["lang"][cand] == o_lang).any(axis=1)
            w = np.where(speaks, 5.0, 1.0)
            chosen = rng.choice(cand, size=min(k, cand.size), replace=False, p=w / w.sum())
            home = np.flatnonzero((pop["country"] == o_idx) & (age >= 15))
            for m in chosen:
                pop["birth_country"][m] = o_idx
                langs = [int(x) for x in pop["lang"][m] if x >= 0 and x not in (o_lang, dest_lang)]
                new = [o_lang, dest_lang, *langs][:MAX_LANGS]
                pop["lang"][m] = new + [-1] * (MAX_LANGS - len(new))
                pop["lang_prof"][m] = [1.0, 0.8, 0.6][: len(new)] + [0.0] * (MAX_LANGS - len(new))
                if home.size:
                    # relatives at home: weighted toward people of a similar age or a generation older
                    gap = age[home] - age[m]
                    wt = np.exp(-0.5 * (np.minimum(np.abs(gap), np.abs(gap - 27)) / 6.0) ** 2) + 1e-6
                    rel = rng.choice(home, size=min(int(rng.integers(2, 5)), home.size), replace=False, p=wt / wt.sum())
                    pending_a += [int(m)] * rel.size
                    pending_b += [int(r) for r in rel]
                migrants_total += 1
    if pending_a:
        add_ties(world, np.asarray(pending_a, np.int64), np.asarray(pending_b, np.int64), EdgeKind.CROSS_BORDER)
    return migrants_total
