"""Scenario specification (what a prompt is translated into).

In Gate 0 scenarios are hand-written YAML; in 1b an LLM parser produces the same
objects from free text.  The LLM only ever produces these parameters; outcomes
come from the engine.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jahan.actors.actions import ACTOR_TYPES, Control, CreateAgent, Decision, Message, Post, PriceChange
from jahan.model.enums import CHANNEL_KIND_NAMES, TOPIC_NAMES, VALUE_NAMES
from jahan.model.questions import QUESTION_ID


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


class Bearing(BaseModel):
    """How a story bears on a watched yes/no question: +1 makes "Yes" much more likely, -1 "No".  People who
    know the question and learn the story shift their belief; the engine sets the size, never the text."""

    model_config = ConfigDict(extra="forbid")
    question: str  # a watched question's id (see `jahan watch list`)
    direction: float = Field(ge=-1, le=1)


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
    bears_on: list[Bearing] = Field(default_factory=list)  # watched questions this story makes more or less likely

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
    ``us_stocks``: US stock index (S&P 500), news in the USA.
    ``sentiment``: direct shift of economic mood in a country (or everywhere).
    Large moves generate automatic headlines in the affected countries when ``auto_news`` is true.
    For a Fed rate decision or a jobs report use an ``announcement`` instead: the engine computes the market
    moves from published estimates.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["macro"] = "macro"
    variable: Literal["fx_local_per_usd", "gold", "us_stocks", "sentiment"]
    country: str | None = None
    steps: list[MacroStep]
    auto_news: bool = True


class Announcement(BaseModel):
    """A scheduled policy decision or data release whose market reaction the engine computes.

    ``indicator`` ``us_policy_rate``: a Federal Reserve decision; ``actual`` is the change in the target range
    in basis points (25 for a quarter-point hike, 0 for a hold) and ``expected`` what markets expected the day
    before in the same unit (the probability-weighted change: a 92% chance of a 25 bp hike is 23).
    ``us_payrolls``: a US jobs report; ``actual`` and ``expected`` (the consensus forecast) are the monthly
    change in nonfarm payrolls in thousands.  Never guess ``expected``: ask when the text does not give it.
    ``guidance_bp`` (rate decisions only): how far the announcement moved 2-year rate expectations beyond the
    decision itself (hawkish > 0); leave 0 unless the user states it.
    US stocks, gold and the euro's dollar rate move on the release day by published event-study estimates
    times the surprise (data/global/markets.yaml), never by numbers in the prompt; the headline becomes news.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["announcement"] = "announcement"
    indicator: Literal["us_policy_rate", "us_payrolls"]
    actual: float
    expected: float
    guidance_bp: float = 0.0
    headline: str
    valence: float = Field(0.0, ge=-1, le=1)  # how good the news looks to ordinary people
    salience: float = Field(0.9, ge=0, le=1)  # HEURISTIC: a top national economic headline
    countries: list[str] = Field(default_factory=lambda: ["USA"])  # where the headline is news
    channel_kinds: list[str] = Field(default_factory=lambda: ["tv", "online", "social", "radio", "newspaper"])
    start_day_offset: int = Field(0, ge=0)
    duration_days: int = Field(3, ge=1)
    release_utc_hour: int = Field(18, ge=0, le=23)  # FOMC statements: 14:00 New York time

    @field_validator("channel_kinds")
    @classmethod
    def _kinds(cls, v: list[str]) -> list[str]:
        bad = [k for k in v if k not in CHANNEL_KIND_NAMES]
        if bad:
            raise ValueError(f"unknown channel kinds {bad}")
        return v

    def news(self) -> NewsEvent:
        """The headline as a news story (the market moves are added by the engine)."""
        return NewsEvent(
            headline=self.headline,
            topic="economy",
            valence=self.valence,
            accuracy=1.0,
            salience=self.salience,
            countries=self.countries,
            channel_kinds=self.channel_kinds,
            start_day_offset=self.start_day_offset,
            duration_days=self.duration_days,
            release_utc_hour=self.release_utc_hour,
        )


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
    bears_on: list[Bearing] = Field(default_factory=list)  # watched questions this statement bears on

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v


class QuestionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["polymarket", "manual"]
    url: str = ""
    slug: str = ""
    event_slug: str = ""
    clob_token_id: str | None = None


class QuestionEvent(BaseModel):
    """Start (or stop) watching a yes/no question: simulated traders hold beliefs about it, and the simulated
    odds are their average.  Created by `jahan watch add/remove odds:...`, never by a free-text prompt.

    ``price`` is the market's "Yes" price on ``fetch_date`` (the traders' starting point); ``prior`` is used
    when there is no market price.  ``trader_share`` and ``skew`` come from data/global/odds.yaml when the
    command builds the event, so the event alone replays the same world.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["question"] = "question"
    action: Literal["open", "close"] = "open"
    id: str
    question: str = ""
    description: str = Field("", max_length=400)
    resolves_by: dt.date | None = None
    source: QuestionSource | None = None
    price: float | None = Field(None, ge=0, le=1)
    fetch_date: str | None = None
    prior: float | None = Field(None, gt=0, lt=1)
    countries: list[str] | None = None  # whose residents trade on it; None = every country
    partisan: float = Field(0.0, ge=-1, le=1)  # +1: the governing camp leans Yes (HEURISTIC shift)
    topic: str = "world"
    trader_share: float = Field(0.10, gt=0, le=1)
    adult_age: float = 18.0
    skew: dict[str, float] = Field(
        default_factory=lambda: {
            "male_under_35": 2.0,
            "male_35_plus": 1.0,
            "female_under_35": 1.0,
            "female_35_plus": 0.5,
        }
    )
    sources: list[str] = Field(default_factory=list)  # registry ids behind trader_share and the aggregation

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not QUESTION_ID.fullmatch(v):
            raise ValueError("a question id is 3-200 lower-case letters, digits or dashes")
        return v

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if v not in TOPIC_NAMES:
            raise ValueError(f"unknown topic {v}; known: {sorted(TOPIC_NAMES)}")
        return v

    def start_price(self) -> float:
        if self.price is not None:
            return float(self.price)
        return float(self.prior if self.prior is not None else 0.5)


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
ScenarioEvent = Annotated[ProductLaunch | NewsEvent | MacroPath | Announcement | Statement, Field(discriminator="type")]
AnyEvent = Annotated[
    ProductLaunch
    | NewsEvent
    | MacroPath
    | Announcement
    | Statement
    | SocialPostEvent
    | SocialDigest
    | QuestionEvent
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
