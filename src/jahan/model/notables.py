"""Notable public figures (real names by default) and role permissions.

Rosters live in ``data/notables/{ISO3}.yaml``; each entry records when the office holder was verified
and the sources for every stance.  Roles and permitted actions live in ``data/roles.yaml``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jahan.model.country import DataError
from jahan.model.enums import TOPIC_NAMES, VALUE_NAMES

SIMULATED_LABEL = "[simulated]"


class Approval(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    source: str
    url: str | None = None


class Stance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    position: float = Field(ge=-1.0, le=1.0)
    summary: str
    source: str

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}")
        return v

    @field_validator("source")
    @classmethod
    def _cited(cls, v: str) -> str:
        if not v.startswith("http"):
            raise ValueError("every stance needs a source URL")
        return v


class NotableEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    role: str
    office: str
    holder_name: str
    sex: Literal["F", "M"]
    birth_year: int | None = None
    verified: str
    sources: list[str]
    visibility: float = Field(ge=0.0, le=10.0)
    audience_note: str = ""
    approval: Approval
    platforms: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    values: dict[str, float]
    stances: list[Stance] = Field(default_factory=list)
    communication_style: str = ""
    confidence: Literal["high", "medium", "low", "provisional", "placeholder"] = "low"

    @field_validator("values")
    @classmethod
    def _values(cls, v: dict[str, float]) -> dict[str, float]:
        bad = set(v) - set(VALUE_NAMES)
        if bad:
            raise ValueError(f"unknown value dimensions {sorted(bad)}")
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"values must sum to 1 (got {total:.3f})")
        return v

    @field_validator("sources")
    @classmethod
    def _sources(cls, v: list[str]) -> list[str]:
        if not v or not all(s.startswith("http") for s in v):
            raise ValueError("each notable needs at least one source URL")
        return v


class NotableRoster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: dt.date
    entries: list[NotableEntry]


class PolicyParam(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    min: float
    max: float


class PolicySpec(BaseModel):
    """How a policy becomes engine events.  ``per_unit`` elasticities are placeholders (see roles.yaml)."""

    model_config = ConfigDict(extra="forbid")
    effect: Literal["macro", "news", "product_price", "product"]
    variable: Literal["sentiment", "fx_local_per_usd", "gold"] | None = None
    scope: Literal["own_country", "target_country"] | None = None
    param: PolicyParam | None = None
    per_unit: float = 0.0
    days: int = Field(1, ge=1)
    salience_bonus: float = 0.0
    confidence: Literal["high", "medium", "low", "provisional", "placeholder"] = "placeholder"
    note: str = ""


class RoleSpec(BaseModel):
    label: str
    can_post: list[Literal["national", "global"]] = Field(default_factory=list)
    meeting_every_days: int | None = None
    policies: dict[str, PolicySpec] = Field(default_factory=dict)


def load_roster(data_dir: Path, iso3: str) -> NotableRoster | None:
    path = data_dir / "notables" / f"{iso3}.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        try:
            return NotableRoster.model_validate(yaml.safe_load(f))
        except Exception as e:
            raise DataError(f"notables/{iso3}.yaml: {e}") from e


def load_roles(data_dir: Path) -> dict[str, RoleSpec]:
    path = data_dir / "roles.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {k: RoleSpec.model_validate(v) for k, v in raw["roles"].items()}


def validate_rosters(data_dir: Path, countries: list[str], start_date: dt.date | None = None) -> list[str]:
    problems: list[str] = []
    try:
        roles = load_roles(data_dir)
    except Exception as e:
        return [f"roles.yaml: {e}"]
    keys: set[str] = set()
    for iso in countries:
        try:
            roster = load_roster(data_dir, iso)
        except DataError as e:
            problems.append(str(e))
            continue
        if roster is None:
            continue
        if start_date is not None and roster.as_of > start_date:
            problems.append(f"notables/{iso}: as_of {roster.as_of} is after the world start date {start_date}")
        for entry in roster.entries:
            if entry.role not in roles:
                problems.append(f"notables/{iso}: {entry.key} has unknown role {entry.role}")
            if entry.key in keys:
                problems.append(f"notables/{iso}: duplicate key {entry.key}")
            keys.add(entry.key)
            if not entry.key.startswith(f"{iso}."):
                problems.append(f"notables/{iso}: key {entry.key} must start with '{iso}.'")
    return problems


def can_do(roles: dict[str, RoleSpec], role: str, policy: str) -> bool:
    spec = roles.get(role)
    return spec is not None and policy in spec.policies
