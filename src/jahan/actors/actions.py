"""Typed actions by agents under human control or LLM-driven public figures.

Every action is a scenario event (``type`` discriminator), so it goes through the same ledger
(pre-action snapshot, reset, experiments, transcripts) as any other prompt.  None of these types is
offered to the free-text scenario parser: a prompt cannot possess an agent or speak for one.

Free text never sets the size of an effect.  A message or post carries a ``Claim``: topic, an optional
product, a one-line headline and a stance quantized to five levels.  The engine sets salience from the
speaker's visibility and credibility from each receiver's trust in the speaker.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jahan.model.enums import TOPIC_NAMES

STANCES = (-1.0, -0.5, 0.0, 0.5, 1.0)
AUDIENCES = ("circle", "community", "national", "global")


def quantize_stance(x: float) -> float:
    """Nearest of the five stance levels; free text cannot express anything finer."""
    return min(STANCES, key=lambda s: (abs(s - float(x)), s))


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: str = Field(min_length=1, max_length=300)
    topic: str = "world"
    stance: float = 0.0  # -1 (bad news) .. +1 (good news); quantized
    about_product: str | None = None
    text: str = ""  # the speaker's own words, kept for transcripts and replies only

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v

    @field_validator("stance")
    @classmethod
    def _stance(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("stance must be in [-1, 1]")
        return quantize_stance(v)


class _Act(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_day_offset: int = Field(0, ge=0)
    session: int = -1  # human session id; -1 = public-figure autonomy or a script


class Control(_Act):
    """A person takes control of an agent (on) or hands it back to autopilot (off)."""

    type: Literal["control"] = "control"
    agent: int
    on: bool = True


class Message(_Act):
    """One-to-one: the target must listen to the actor (an edge target -> actor)."""

    type: Literal["message"] = "message"
    actor: int
    target: int
    claim: Claim
    utc_hour: int = Field(ge=0, le=23)


class Post(_Act):
    """Public post; the reach actually used is capped by the actor's visibility and role."""

    type: Literal["post"] = "post"
    actor: int
    audience: Literal["circle", "community", "national", "global"] = "circle"
    claim: Claim
    utc_hour: int = Field(ge=0, le=23)
    duration_days: int = Field(3, ge=1, le=30)
    salience_bonus: float = Field(0.0, ge=0.0, le=0.2)  # set by the engine for directives, never by text


class Decision(_Act):
    type: Literal["decision"] = "decision"
    actor: int
    product: str
    choice: Literal["adopt", "reject", "delay"]


class PriceChange(_Act):
    type: Literal["price_change"] = "price_change"
    actor: int = -1
    product: str
    pct: float = Field(ge=-90, le=200)
    countries: list[str] | None = None  # None = everywhere the product is sold


class CreateAgent(_Act):
    """A new person in the world (population weight 0).  Unset fields are drawn from a resident of the
    same country, sex and age band, so the profile stays consistent with that country's data."""

    type: Literal["create_agent"] = "create_agent"
    country: str
    sex: Literal["F", "M"] | None = None
    age: int | None = Field(None, ge=0, le=105)
    first_name: str | None = None
    last_name: str | None = None
    region: Literal["urban", "rural"] | None = None
    education: int | None = Field(None, ge=0, le=8)  # ISCED level
    income_month: float | None = Field(None, ge=0)  # local currency
    languages: list[str] | None = None
    values: dict[str, float] | None = None
    visibility: float | None = Field(None, ge=0.0, le=10.0)
    friends: int = Field(5, ge=0, le=50)
    acquaintances: int = Field(5, ge=0, le=50)
    household_of: int | None = None  # join this agent's household instead of living alone
    notable_role: str | None = None  # a role from roles.yaml makes it a public figure (reserved column)
    office: str | None = None
    description: str = ""  # free text the profile was drawn from (record only)

    @field_validator("start_day_offset")
    @classmethod
    def _now(cls, v: int) -> int:
        if v:
            raise ValueError("agents are created immediately (start_day_offset must be 0)")
        return v


ACTOR_TYPES = (Control, Message, Post, Decision, PriceChange, CreateAgent)
ActorEvent = Control | Message | Post | Decision | PriceChange | CreateAgent
