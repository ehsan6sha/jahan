"""Scenario specification (what a prompt is translated into).

In Gate 0 scenarios are hand-written YAML; in 1b an LLM parser produces the same
objects from free text.  The LLM only ever produces these parameters; outcomes
come from the engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jahan.actors.actions import ACTOR_TYPES, Control, CreateAgent, Decision, Message, Post, PriceChange
from jahan.model.enums import CHANNEL_KIND_NAMES, TOPIC_NAMES, VALUE_NAMES


class MediaPush(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_kinds: list[str]
    countries: list[str] | None = None  # None = every country where the product launches
    start_day_offset: int = 0
    duration_days: int = Field(30, ge=1)
    intensity: float = Field(1.0, ge=0)

    @field_validator("channel_kinds")
    @classmethod
    def _kinds(cls, v: list[str]) -> list[str]:
        bad = [k for k in v if k not in CHANNEL_KIND_NAMES]
        if bad:
            raise ValueError(f"unknown channel kinds {bad}; known: {sorted(CHANNEL_KIND_NAMES)}")
        return v


class ProductLaunch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["product_launch"] = "product_launch"
    name: str
    topic: str = "tech"
    attributes: dict[str, float] = Field(default_factory=dict)  # value dimension -> weight
    true_quality: float = Field(ge=0, le=1)
    price_usd: float | None = Field(None, ge=0)
    price_local: dict[str, float] = Field(default_factory=dict)  # ISO3 -> price in local currency
    countries: list[str]
    available_from_day_offset: int = 0
    salience: float = Field(0.6, ge=0, le=1)
    valence: float = Field(1.0, ge=-1, le=1)  # tone of the launch messaging
    release_utc_hour: int = Field(14, ge=0, le=23)
    pushes: list[MediaPush] = Field(default_factory=list)
    owner_agent: int = -1  # agent whose company makes it (may change its price); -1 = none

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v

    @field_validator("attributes")
    @classmethod
    def _attrs(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if k not in VALUE_NAMES]
        if bad:
            raise ValueError(f"unknown value dimensions {bad}; known: {sorted(VALUE_NAMES)}")
        return v


DEFAULT_SENTIMENT_WEIGHT = {"economy": 1.0, "world": 0.8, "politics": 0.6}


class NewsEvent(BaseModel):
    """A story pushed through media channels (and then word of mouth).

    ``valence`` says how good the story makes things look (-1 bad .. +1 good).  ``accuracy`` is how
    true it is: the engine's notion of reality for the story is 0.5 + 0.5 * valence * accuracy, so
    discerning agents drift toward it.  ``about_product`` links a claim to a launched product (e.g. a
    safety rumour); then reality is the product's true quality and agents' views of the product move.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["news"] = "news"
    headline: str
    topic: str = "world"
    valence: float = Field(ge=-1, le=1)
    accuracy: float = Field(1.0, ge=0, le=1)
    salience: float = Field(0.7, ge=0, le=1)
    countries: list[str] | None = None  # None = every country in the world
    channel_kinds: list[str] = Field(default_factory=lambda: ["tv", "online", "social", "radio", "newspaper"])
    start_day_offset: int = Field(0, ge=0)
    duration_days: int = Field(7, ge=1)
    release_utc_hour: int = Field(12, ge=0, le=23)
    intensity: float = Field(1.0, ge=0)
    sentiment_weight: float | None = None  # default by topic (economy 1.0, world 0.8, politics 0.6, else 0.2)
    about_product: str | None = None
    language: str | None = None  # restrict to channels in this language (e.g. a Hindi headline); None = all
    # where the story comes from: a scenario prompt, a user-fed future event, or a real fetched headline.
    # Set by the code path that creates the event (the parser always overrides it), never by a model.
    origin: Literal["scenario", "user_feed", "web"] = "scenario"

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v

    @field_validator("channel_kinds")
    @classmethod
    def _kinds(cls, v: list[str]) -> list[str]:
        bad = [k for k in v if k not in CHANNEL_KIND_NAMES]
        if bad:
            raise ValueError(f"unknown channel kinds {bad}")
        return v

    def weight(self) -> float:
        return (
            self.sentiment_weight
            if self.sentiment_weight is not None
            else DEFAULT_SENTIMENT_WEIGHT.get(self.topic, 0.2)
        )


class MacroStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_day_offset: int = Field(0, ge=0)
    days: int = Field(1, ge=1)
    pct_per_day: float  # percent change per day (for sentiment: points on a -100..100 scale)


class MacroPath(BaseModel):
    """Scheduled moves of a macro variable, e.g. gold +1%/day for 7 days then -10% in one day.

    variable ``fx_local_per_usd``: local currency per US dollar (a rising dollar = positive pct);
    ``country`` None applies it to every non-USD country.  ``gold``: global gold price index.
    ``sentiment``: direct shift of economic mood in a country (or everywhere).
    Large moves generate automatic headlines in the affected countries when ``auto_news`` is true.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["macro"] = "macro"
    variable: Literal["fx_local_per_usd", "gold", "sentiment"]
    country: str | None = None
    steps: list[MacroStep]
    auto_news: bool = True


class Statement(BaseModel):
    """A public statement by a notable figure (key from data/notables, e.g. USA.head_of_state).

    Reach comes from the speaker's visibility and the requested audience (capped by the role); agents
    believe it according to their parasocial trust in the speaker.  Text attributed to a real person is
    labelled [simulated].
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["statement"] = "statement"
    speaker: str
    headline: str
    topic: str = "politics"
    valence: float = Field(ge=-1, le=1)
    accuracy: float = Field(0.5, ge=0, le=1)
    about_product: str | None = None
    audience: Literal["national", "global"] = "national"
    start_day_offset: int = Field(0, ge=0)
    duration_days: int = Field(3, ge=1)
    release_utc_hour: int = Field(14, ge=0, le=23)

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v


class SocialPostEvent(BaseModel):
    """A real post fetched from a social network, pushed through the social channels in its language.

    Reach comes from ``author_visibility`` (a roster figure's visibility, else log10 of followers) and
    ``engagement`` (likes and reposts), both computed by the ingest code from the platform's numbers, never
    by a model.  ``intensity`` is the platform's share of the country's internet users.  Created only by
    `jahan ingest`; a free-text prompt cannot produce it.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["social"] = "social"
    platform: Literal["x", "bluesky", "mastodon", "telegram"]
    post_id: str
    author_handle: str
    author_name: str = ""
    author_key: str | None = None  # roster key when a public figure wrote (or reposted) it
    author_visibility: float = Field(ge=0.0, le=10.0)
    engagement: float = Field(0.0, ge=0.0, le=1.0)
    headline: str
    topic: str = "world"
    valence: float = Field(0.0, ge=-1, le=1)
    accuracy: float = Field(0.5, ge=0, le=1)
    language: str
    countries: list[str]
    url: str = ""
    mentions: list[str] = Field(default_factory=list)
    is_repost: bool = False
    start_day_offset: int = Field(0, ge=0)
    duration_days: int = Field(2, ge=1)
    release_utc_hour: int = Field(12, ge=0, le=23)
    intensity: float = Field(1.0, ge=0.0, le=1.0)

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v


class SocialDigest(BaseModel):
    """What one social fetch saw, for display and grounding: per-platform counts, problems, the keys of
    posts already ingested, recent posts by public figures, trending topics and a feed per country.
    Applied into ``world.macro["social"]`` so a reset restores it.  Trending has no behavioural effect."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["social_digest"] = "social_digest"
    fetch_day: str
    start_day_offset: int = Field(0, ge=0)
    stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    problems: list[str] = Field(default_factory=list)
    seen: list[str] = Field(default_factory=list)
    figures: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # roster key -> recent posts
    trending: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # ISO -> topics
    feed: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # ISO -> rows


# What a free-text prompt may produce.  Actor events (control, messages, posts, ...) and fetched social
# posts are deliberately not in this union: only the actor commands and `jahan ingest` create them.
ScenarioEvent = Annotated[ProductLaunch | NewsEvent | MacroPath | Statement, Field(discriminator="type")]
AnyEvent = Annotated[
    ProductLaunch
    | NewsEvent
    | MacroPath
    | Statement
    | SocialPostEvent
    | SocialDigest
    | Control
    | Message
    | Post
    | Decision
    | PriceChange
    | CreateAgent,
    Field(discriminator="type"),
]


class PromptScenario(BaseModel):
    """The schema the LLM parser sees and must satisfy (world events only)."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = ""
    events: list[ScenarioEvent]
    horizon_days: int = Field(180, ge=1)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = ""
    events: list[AnyEvent]
    horizon_days: int = Field(180, ge=1)

    def has_actor_events(self) -> bool:
        return any(isinstance(ev, ACTOR_TYPES) for ev in self.events)

    @classmethod
    def load(cls, path: Path) -> Scenario:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
