"""Snapshots: every table as .npz plus a JSON sidecar.

No RNG state is stored (draws are pure functions of seed, stream, day and ids).
Country and name files are re-read from the data directory; their hashes are
recorded and checked so a snapshot is never silently combined with edited data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import load_world_shell
from jahan.model.country import generated_files
from jahan.model.schema import (
    AGENT_COLUMNS,
    CHANNEL_COLUMNS,
    EDGE_COLUMNS,
    HOUSEHOLD_COLUMNS,
    ITEM_COLUMNS,
    PRODUCT_COLUMNS,
    PUSH_COLUMNS,
    SCHEMA_VERSION,
    WORKPLACE_COLUMNS,
)
from jahan.model.world import BeliefStore, ParasocialStore, World
from jahan.soa import SoATable

TABLES = {
    "pop.": ("pop", AGENT_COLUMNS),
    "hh.": ("households", HOUSEHOLD_COLUMNS),
    "wp.": ("workplaces", WORKPLACE_COLUMNS),
    "edge.": ("edges", EDGE_COLUMNS),
    "ch.": ("channels", CHANNEL_COLUMNS),
    "item.": ("items", ITEM_COLUMNS),
    "prod.": ("products", PRODUCT_COLUMNS),
    "push.": ("pushes", PUSH_COLUMNS),
}


class SnapshotError(RuntimeError):
    pass


def data_hashes(config: RunConfig, data_dir: Path) -> dict[str, str]:
    out = {}
    files = [data_dir / "global" / "languages.yaml"]
    for c in config.countries:
        files.append(data_dir / "countries" / f"{c}.yaml")
        files.extend(generated_files(data_dir, c))
        files.append(data_dir / "names" / f"{c}.yaml")
    if config.diaspora_file and (data_dir / config.diaspora_file).exists():
        files.append(data_dir / config.diaspora_file)
    if config.notables:
        files.extend(f for f in (data_dir / "notables" / f"{c}.yaml" for c in config.countries) if f.exists())
        if (data_dir / "roles.yaml").exists():
            files.append(data_dir / "roles.yaml")
    for f in files:
        out[str(f.relative_to(data_dir)).replace("\\", "/")] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def world_fingerprint(world: World) -> str:
    """sha256 of the simulation state (tables, beliefs, parasocial trust, day, schedule, macro, notables).

    Used to prove that a replayed session turn is applied to exactly the state it was recorded on.
    """
    h = hashlib.sha256()
    for prefix, (attr, _spec) in TABLES.items():
        for name, arr in sorted(getattr(world, attr).to_arrays(prefix).items()):
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
    for name, arr in sorted(world.beliefs.to_arrays(world.pop.n).items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.ascontiguousarray(world.ptrust.alpha[: world.pop.n]).tobytes())
    h.update(np.ascontiguousarray(world.ptrust.beta[: world.pop.n]).tobytes())
    for a in (world.product_price, world.product_available, world.product_price_usd):
        h.update(np.ascontiguousarray(a).tobytes())
    meta = {
        "day": world.day,
        "seed": world.sim_seed,
        "counters": world.counters,
        "scheduled": world.scheduled,
        "macro": world.macro,
        "notables": world.notables,
        "products": world.product_names,
    }
    # normalise through JSON so in-memory and reloaded worlds (int vs str keys) hash the same
    h.update(json.dumps(json.loads(json.dumps(meta, default=str)), sort_keys=True).encode())
    return h.hexdigest()


def save_snapshot(world: World, path: Path, data_dir: Path | None = None, label: str = "") -> Path:
    dd = data_dir or paths.data_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for prefix, (attr, _spec) in TABLES.items():
        arrays.update(getattr(world, attr).to_arrays(prefix))
    arrays.update(world.beliefs.to_arrays(world.pop.n))
    arrays["product_price"] = world.product_price
    arrays["product_available"] = world.product_available
    arrays["product_price_usd"] = world.product_price_usd
    arrays["ptrust.alpha"] = world.ptrust.alpha[: world.pop.n]
    arrays["ptrust.beta"] = world.ptrust.beta[: world.pop.n]
    np.savez(path.with_suffix(".npz"), **arrays)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "day": world.day,
        "sim_seed": world.sim_seed,
        "counters": world.counters,
        "scheduled": world.scheduled,
        "channel_names": world.channel_names,
        "macro": world.macro,
        "notables": world.notables,
        "product_names": world.product_names,
        "config": json.loads(world.config.model_dump_json()),
        "data_hashes": data_hashes(world.config, dd),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path.with_suffix(".npz")


def load_snapshot(path: Path, data_dir: Path | None = None, check_data: bool = True) -> World:
    dd = data_dir or paths.data_dir()
    path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise SnapshotError(f"snapshot {path} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["schema_version"] != SCHEMA_VERSION:
        raise SnapshotError(f"snapshot schema {meta['schema_version']} != code schema {SCHEMA_VERSION}")
    config = RunConfig.model_validate(meta["config"])
    if check_data:
        now = data_hashes(config, dd)
        changed = [k for k, v in meta["data_hashes"].items() if now.get(k) != v]
        if changed:
            raise SnapshotError(
                f"data files changed since this snapshot was written: {changed}. "
                "Regenerate the world or restore the data files."
            )
    world = load_world_shell(config, dd)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for prefix, (attr, spec) in TABLES.items():
        headroom = 1.5 if attr in ("pop", "edges", "items") else 1.2
        setattr(world, attr, SoATable.from_arrays(spec, arrays, prefix, headroom=headroom))
    world.beliefs = BeliefStore.from_arrays(arrays, world.pop.capacity)
    world.product_price = arrays["product_price"]
    world.product_available = arrays["product_available"]
    world.product_price_usd = arrays["product_price_usd"]
    world.macro = dict(meta["macro"])
    world.notables = list(meta.get("notables", []))
    world.product_names = list(meta.get("product_names", []))
    pa, pb = arrays["ptrust.alpha"], arrays["ptrust.beta"]
    world.ptrust = ParasocialStore(world.pop.capacity, int(pa.shape[1]))
    world.ptrust.alpha[: pa.shape[0]] = pa
    world.ptrust.beta[: pb.shape[0]] = pb
    world.day = int(meta["day"])
    world.sim_seed = int(meta["sim_seed"])
    world.counters = {k: int(v) for k, v in meta["counters"].items()}
    world.scheduled = list(meta["scheduled"])
    world.channel_names = list(meta["channel_names"])
    world.invalidate_edge_index()
    return world
