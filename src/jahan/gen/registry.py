"""Data registry: provenance for every distribution."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from jahan.model.country import CountrySpec, DataError, load_country
from jahan.model.investors import band_problems, load_investors
from jahan.model.markets import load_markets
from jahan.model.notables import (
    camp_priors,
    camp_shares,
    camp_template,
    load_roles,
    load_roster,
    validate_rosters,
)
from jahan.model.questions import odds_problems
from jahan.model.social import load_social, validate_social

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


def _politics_placeholders(data_dir: Path, iso: str, spec: CountrySpec) -> list[str]:
    """Strict mode: camp shares and per-camp approvals that have no source."""
    out = []
    if spec.politics is None:
        out.append(
            f"{iso}: no politics block; every resident trusts public figures from the same starting point (strict mode)"
        )
    elif spec.politics.placeholder:
        out.append(f"{iso}: political camp shares are a placeholder (strict mode)")
    try:
        roster = load_roster(data_dir, iso)
    except DataError:
        return out  # reported by validate_rosters
    for e in roster.entries if roster is not None else []:
        try:
            _, _, status = camp_priors(
                e.approval.value,
                camp_shares(spec.politics, e.alignment),
                e.approval_by_camp,
                camp_template(spec),
            )
        except ValueError:
            continue  # reported by validate_rosters
        if status == "placeholder":
            out.append(
                f"notables/{iso}: {e.key} has no approval by camp; every camp starts at the topline (strict mode)"
            )
        elif status == "template":
            out.append(
                f"notables/{iso}: {e.key} has no approval by camp; a gap borrowed from US polling is used (strict mode)"
            )
    return out


def _social_problems(data_dir: Path, countries: list[str], reg: dict[str, RegistryEntry], strict: bool) -> list[str]:
    """data/feeds/social.yaml: accounts name roster figures, and platform shares cite the registry."""
    if not (data_dir / "feeds" / "social.yaml").exists():
        return []
    out = validate_social(data_dir, countries)
    if out:
        return out
    for iso, usage in load_social(data_dir).platform_usage.countries.items():
        if iso not in countries:
            continue
        entry = reg.get(usage.source)
        if entry is None:
            out.append(f"feeds/social.yaml: {iso} source id '{usage.source}' not in registry")
        elif strict and entry.confidence in WEAK:
            out.append(f"feeds/social.yaml: {iso} source '{usage.source}' is {entry.confidence} (strict mode)")
    return out


def _markets_problems(data_dir: Path, reg: dict[str, RegistryEntry], strict: bool) -> list[str]:
    """data/global/markets.yaml: every estimate cites the registry; strict mode lists weak ones."""
    try:
        cfg = load_markets(data_dir)
    except Exception as e:  # noqa: BLE001 - reported as a data problem
        return [f"global/markets.yaml: {e}"]
    if cfg is None:
        return []
    out = []
    for name, ind in cfg.indicators.items():
        for eff in ind.surprise + ind.guidance:
            if eff.target not in cfg.targets:
                out.append(f"global/markets.yaml: {name} uses unknown target {eff.target}")
    for sid in cfg.source_ids():
        entry = reg.get(sid)
        if entry is None:
            out.append(f"global/markets.yaml: source id '{sid}' not in registry")
        elif strict and entry.confidence in WEAK:
            out.append(f"global/markets.yaml: source '{sid}' is {entry.confidence} (strict mode)")
    return out


def _investors_problems(data_dir: Path, reg: dict[str, RegistryEntry], strict: bool) -> list[str]:
    """data/global/investors.yaml: bands cover 0-100; every source cites the registry; strict lists weak ones."""
    try:
        cfg = load_investors(data_dir)
    except Exception as e:  # noqa: BLE001 - reported as a data problem
        return [f"global/investors.yaml: {e}"]
    if cfg is None:
        return []
    out = [f"global/investors.yaml: {p}" for p in band_problems(cfg)]
    for sid in cfg.source_ids():
        entry = reg.get(sid)
        if entry is None:
            out.append(f"global/investors.yaml: source id '{sid}' not in registry")
        elif strict and entry.confidence in WEAK:
            out.append(f"global/investors.yaml: source '{sid}' is {entry.confidence} (strict mode)")
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
            problems += _politics_placeholders(data_dir, iso, spec)
    problems += _social_problems(data_dir, countries, reg, strict)
    problems += _markets_problems(data_dir, reg, strict)
    problems += _investors_problems(data_dir, reg, strict)
    problems += odds_problems(data_dir, reg)
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
