"""Data registry: provenance for every distribution."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from jahan.model.country import DataError, load_country
from jahan.model.notables import load_roles, validate_rosters

Confidence = Literal["high", "medium", "low", "provisional", "placeholder"]
WEAK = {"provisional", "placeholder"}


class RegistryEntry(BaseModel):
    id: str
    country: str | None
    variable: str
    source: str
    url: str
    year: int
    retrieved: str | None = None
    license: str
    confidence: Confidence
    notes: str = ""


def load_registry(data_dir: Path) -> dict[str, RegistryEntry]:
    with (data_dir / "registry.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    entries = [RegistryEntry.model_validate(e) for e in raw["entries"]]
    out: dict[str, RegistryEntry] = {}
    for e in entries:
        if e.id in out:
            raise DataError(f"duplicate registry id {e.id}")
        out[e.id] = e
    return out


def validate_data(
    data_dir: Path, countries: list[str], strict: bool = False, start_date: dt.date | None = None
) -> list[str]:
    """Return a list of problems (empty = valid).  Notable rosters are checked too."""
    problems: list[str] = list(validate_rosters(data_dir, countries, start_date))
    try:
        reg = load_registry(data_dir)
    except Exception as e:
        return [f"registry: {e}"]
    for iso in countries:
        try:
            spec = load_country(data_dir, iso)
        except DataError as e:
            problems.append(str(e))
            continue
        for sid in spec.source_ids():
            entry = reg.get(sid)
            if entry is None:
                problems.append(f"{iso}: source id '{sid}' not in registry")
            elif strict and entry.confidence in WEAK:
                problems.append(f"{iso}: source '{sid}' is {entry.confidence} (strict mode)")
        names_path = data_dir / spec.names.file
        if not names_path.exists():
            problems.append(f"{iso}: names file {names_path} missing")
    if strict:
        try:
            roles = load_roles(data_dir)
        except Exception as e:  # noqa: BLE001 - reported by validate_rosters already
            roles = {}
            problems.append(f"roles.yaml: {e}")
        for role, rs in roles.items():
            for kind, pol in rs.policies.items():
                if pol.effect == "macro" and pol.confidence in WEAK:
                    problems.append(f"roles.yaml: {role}.{kind} elasticity is {pol.confidence} (strict mode)")
    return problems


def confidence_summary(data_dir: Path, countries: list[str]) -> dict[str, dict[str, str]]:
    reg = load_registry(data_dir)
    out: dict[str, dict[str, str]] = {}
    for iso in countries:
        spec = load_country(data_dir, iso)
        out[iso] = {sid: reg[sid].confidence if sid in reg else "missing" for sid in spec.source_ids()}
    return out
