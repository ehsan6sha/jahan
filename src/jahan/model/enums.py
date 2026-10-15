"""Integer enums stored in SoA columns.  Values are part of the snapshot format: only append."""

from __future__ import annotations

from enum import IntEnum


class Sex(IntEnum):
    F = 0
    M = 1


class Marital(IntEnum):
    SINGLE = 0
    MARRIED = 1
    DIVORCED = 2
    WIDOWED = 3


class EmpStatus(IntEnum):
    CHILD = 0
    STUDENT = 1
    EMPLOYED = 2
    UNEMPLOYED = 3
    HOMEMAKER = 4
    RETIRED = 5
    INACTIVE = 6  # not in labour force for other reasons (disability, caring, discouraged)


EMP_STATUS_NAMES = {e.name.lower(): e for e in EmpStatus}


class InfoTier(IntEnum):
    MINIMAL = 0
    MAJOR_NEWS = 1
    WELL_INFORMED = 2
    SPECIALIST = 3


class EdgeKind(IntEnum):
    HOUSEHOLD = 0
    KIN = 1
    COLLEAGUE = 2
    CLASSMATE = 3
    FRIEND = 4
    ACQUAINTANCE = 5
    CROSS_BORDER = 6


class ChannelKind(IntEnum):
    TV = 0
    NEWSPAPER = 1
    ONLINE = 2
    SOCIAL = 3
    RADIO = 4
    SPECIALIST = 5


CHANNEL_KIND_NAMES = {e.name.lower(): e for e in ChannelKind}


class OriginKind(IntEnum):
    AGENT = 0
    CHANNEL = 1
    WEB = 2
    USER_FEED = 3
    SCENARIO = 4
    MACRO = 5
    SYNTHETIC = 6
    NOTABLE = 7
    HUMAN = 8
    SOCIAL = 9  # a real post fetched from a social network (not by a public figure on the roster)


class Topic(IntEnum):
    POLITICS = 0
    ECONOMY = 1
    TECH = 2
    HEALTH = 3
    SPORTS = 4
    ENTERTAINMENT = 5
    SCIENCE = 6
    RELIGION = 7
    FAMILY = 8
    LOCAL = 9
    WORLD = 10
    LIFESTYLE = 11


N_TOPICS = len(Topic)
TOPIC_NAMES = {e.name.lower(): e for e in Topic}


class ValueDim(IntEnum):
    SELF = 0
    MONEY = 1
    CULTURE = 2
    FREEDOM = 3
    FAMILY = 4
    SECURITY = 5


N_VALUES = len(ValueDim)
VALUE_NAMES = {e.name.lower(): e for e in ValueDim}


class Big5(IntEnum):
    OPENNESS = 0
    CONSCIENTIOUSNESS = 1
    EXTRAVERSION = 2
    AGREEABLENESS = 3
    NEUROTICISM = 4


class EventKind(IntEnum):
    BORN = 1
    DIED = 2
    MARRIED = 3
    DIVORCED = 4
    JOB_CHANGED = 5
    STATUS_CHANGED = 6
    INFO_EXPOSED = 7
    ADOPTED = 8
    OUTCOME_REALIZED = 9
    SCENARIO_INJECTED = 10
    ITEM_CREATED = 11
    MACRO_UPDATE = 12
    WEB_INGESTED = 13
    USER_FEED = 14
    INTERVIEW = 15
    AGENT_CREATED = 16
    POSSESSED = 17
    RELEASED = 18
    HUMAN_ACTION = 19
    NOTABLE_ACTION = 20
    ACTION_REJECTED = 21
    ACTION_SKIPPED = 22
    AUTOPILOT_TURN = 23
    SNAPSHOT = 24
    WEB_INGEST_FAILED = 25
    TRAVEL_START = 26
    TRAVEL_END = 27
    SOCIAL_SHARED = 28  # an ordinary person reposted a tracked item on social media
    ITEM_EVICTED = 29  # the tracked-item pool was full; the oldest item no longer broadcast lost its slot
    # A running story (e.g. a currency slide) got a new headline.  Logs written before 2026-10-15 used 28 for
    # this too, which collided with SOCIAL_SHARED; use `is_item_update` to read both.
    ITEM_UPDATED = 30
    QUESTION_OPENED = 31  # a watched question: item = its belief slot's item, value = starting price
    QUESTION_CLOSED = 32  # value = the simulated odds when it closed, detail = why
    ODDS_RATED = 33  # a story's bearing on a question: item = the story, value = direction, detail = question id


def is_item_update(kind: int, detail: str | None) -> bool:
    """A headline update of a running story.  Older logs stored these as kind 28 (shared with reposts), which
    carry no detail text, so a kind-28 row with a detail is a legacy update."""
    return kind == EventKind.ITEM_UPDATED or (kind == EventKind.SOCIAL_SHARED and bool(detail))


class WorkShift(IntEnum):
    NONE = 0
    DAY = 1
    EVENING = 2
    NIGHT = 3


class RaisedBy(IntEnum):
    TWO_PARENT = 0
    SINGLE = 1
    OTHER = 2


# ISCED 2011 attainment levels used as codes 0..8.
ISCED_LEVELS = list(range(9))


def edu_group(isced: int) -> int:
    """0 = low (ISCED 0-2), 1 = mid (3-5), 2 = high (6-8)."""
    if isced <= 2:
        return 0
    if isced <= 5:
        return 1
    return 2
