"""Notable public figures (real names by default) and role permissions.

Rosters live in ``data/notables/{ISO3}.yaml``; each entry records when the office holder was verified
and the sources for every stance.  Roles and permitted actions live in ``data/roles.yaml``.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jahan.model.country import CAMP_TAU, CountrySpec, DataError, Politics, load_country
from jahan.model.enums import TOPIC_NAMES, VALUE_NAMES
from jahan.model.mentions import script_of

SIMULATED_LABEL = "[simulated]"
# The script a country's main language is written in; its figures need at least one alias in that script.
NATIVE_SCRIPT = {"fa": "arabic", "ar": "arabic", "ur": "arabic", "zh": "cjk", "hi": "devanagari", "mr": "devanagari"}
Confidence = Literal["high", "medium", "low", "provisional", "placeholder"]


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


class CampApproval(BaseModel):
    """Approval of a figure within each political camp of their country (polls by party or camp)."""

    model_config = ConfigDict(extra="forbid")
    aligned: float | None = Field(None, ge=0.0, le=1.0)
    neutral: float | None = Field(None, ge=0.0, le=1.0)
    opposed: float | None = Field(None, ge=0.0, le=1.0)
    source: str
    url: str | None = None
    confidence: Confidence = "low"


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
    aliases: list[str] = Field(default_factory=list)  # names that count as "naming" the figure in the news
    alignment: float = Field(0.0, ge=-1.0, le=1.0)  # position on the country's political axis
    approval_by_camp: CampApproval | None = None
    platforms: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    values: dict[str, float]
    stances: list[Stance] = Field(default_factory=list)
    communication_style: str = ""
    confidence: Confidence = "low"

    @model_validator(mode="after")
    def _camps(self) -> NotableEntry:
        if self.approval_by_camp is not None and abs(self.alignment) < CAMP_TAU:
            raise ValueError("a non-partisan figure (|alignment| < 1/3) cannot have approval by camp")
        if any(len(a.strip()) < 2 for a in self.aliases):
            raise ValueError("aliases need at least two characters")
        return self

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


CampStatus = Literal["sourced", "sourced_shifted", "derived", "template", "placeholder", "non_partisan"]
CAMPS = ("aligned", "neutral", "opposed")
# HEURISTIC: the partisan gap borrowed for figures without approval by camp (Gallup, October 2025, Trump:
# Republicans 91%, independents 33%, Democrats 6%), shifted to each figure's own approval.  It is used in
# full in the country it was measured in; elsewhere the log-odds distances from the neutral camp are halved,
# because US polarisation is unusually high and multiparty systems see more cross-camp approval.
CAMP_TEMPLATE = {"aligned": 0.91, "neutral": 0.33, "opposed": 0.06}
TEMPLATE_HOME = "USA"
TEMPLATE_GAP_ELSEWHERE = 0.5


def camp_shares(politics: Politics | None, alignment: float) -> dict[str, float] | None:
    """Adult shares of the camps relative to a figure: aligned with them, neutral, opposed (None if non-partisan)."""
    if politics is None or abs(alignment) < CAMP_TAU:
        return None
    camps = politics.camps
    if alignment > 0:
        return {"aligned": camps["pos"], "neutral": camps["neutral"], "opposed": camps["neg"]}
    return {"aligned": camps["neg"], "neutral": camps["neutral"], "opposed": camps["pos"]}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def camp_template(spec: CountrySpec) -> dict[str, float] | None:
    """The borrowed partisan gap for this country, or None where camp shares are a placeholder (a borrowed
    gap on placeholder camps would invent structure)."""
    if spec.politics is None or spec.politics.placeholder:
        return None
    scale = 1.0 if spec.iso3 == TEMPLATE_HOME else TEMPLATE_GAP_ELSEWHERE
    mid = _logit(CAMP_TEMPLATE["neutral"])
    return {c: _sigmoid(mid + scale * (_logit(v) - mid)) for c, v in CAMP_TEMPLATE.items()}


def _shift_to(topline: float, shares: dict[str, float], camps: dict[str, float]) -> tuple[dict[str, float], float]:
    """One common log-odds shift of ``camps`` so the camp-weighted mean equals ``topline`` (bisection)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        m = sum(shares[c] * _sigmoid(_logit(camps[c]) + mid) for c in CAMPS)
        lo, hi = (mid, hi) if m < topline else (lo, mid)
    delta = (lo + hi) / 2
    return {c: _sigmoid(_logit(camps[c]) + delta) for c in CAMPS}, delta


def camp_priors(
    topline: float,
    shares: dict[str, float] | None,
    given: CampApproval | None,
    template: dict[str, float] | None = None,
) -> tuple[dict[str, float], float, CampStatus]:
    """Approval of a figure in each camp, consistent with the topline (camp-weighted mean = topline).

    - ``non_partisan``: no camp shares (the figure sits in the neutral band, or the country has no axis).
    - ``template``: no camp data; ``template`` (from ``camp_template``) shifted to the topline.
    - ``placeholder``: no camp data and no template, every camp gets the topline.
    - ``sourced``: all three given and their weighted mean is within 0.02 of the topline.
    - ``sourced_shifted``: all three given but from a different moment; one common log-odds shift is solved
      so the weighted mean equals the current topline (the shift is returned).
    - ``derived``: some camps given; the rest share one value solved from the topline (ValueError if it
      falls outside [0, 1]).
    """
    flat = {c: topline for c in CAMPS}
    if shares is None:
        return flat, 0.0, "non_partisan"
    vals = {c: getattr(given, c) for c in CAMPS} if given is not None else {}
    known = {c: v for c, v in vals.items() if v is not None}
    if not known:
        if template is not None:
            prior, delta = _shift_to(topline, shares, template)
            return prior, delta, "template"
        return flat, 0.0, "placeholder"
    if len(known) == 3:
        mean = sum(shares[c] * known[c] for c in CAMPS)
        if abs(mean - topline) <= 0.02:
            return {c: float(known[c]) for c in CAMPS}, 0.0, "sourced"
        prior, delta = _shift_to(topline, shares, {c: float(known[c]) for c in CAMPS})
        return prior, delta, "sourced_shifted"
    rest = [c for c in CAMPS if c not in known]
    rest_share = sum(shares[c] for c in rest)
    value = (topline - sum(shares[c] * known[c] for c in known)) / max(rest_share, 1e-12)
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"camp approvals {known} cannot average to the topline {topline:.3f} "
            f"(the other camps would need {value:.3f})"
        )
    return {c: float(known.get(c, value)) for c in CAMPS}, 0.0, "derived"


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
        try:
            spec = load_country(data_dir, iso)
            politics, script, template = (
                spec.politics,
                NATIVE_SCRIPT.get(spec.languages_official[0]),
                camp_template(spec),
            )
        except Exception:  # the country file's own problems are reported by validate_data
            politics, script, template = None, None, None
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
            if not entry.aliases:
                problems.append(f"notables/{iso}: {entry.key} needs at least one alias (how the news names them)")
            elif script is not None and not any(script_of(a) == script for a in entry.aliases):
                problems.append(f"notables/{iso}: {entry.key} needs an alias written in {script} script")
            try:
                camp_priors(
                    entry.approval.value,
                    camp_shares(politics, entry.alignment),
                    entry.approval_by_camp,
                    template,
                )
            except ValueError as e:
                problems.append(f"notables/{iso}: {entry.key}: {e}")
    return problems


def can_do(roles: dict[str, RoleSpec], role: str, policy: str) -> bool:
    spec = roles.get(role)
    return spec is not None and policy in spec.policies
