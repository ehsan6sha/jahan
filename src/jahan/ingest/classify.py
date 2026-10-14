"""Turn headlines into scenario news events.

With an LLM: batches of headlines are classified into topic, valence (-1 bad .. +1 good for the reader's
country) and salience.  Without one: headlines keep feed order as salience and are neutral (they raise
awareness but do not move mood).  Only the top ``per_country`` headlines per country become tracked
items, which bounds memory and keeps classification cost small.
"""

from __future__ import annotations

import datetime as dt
import json

from jahan.ingest.feeds import Headline
from jahan.llm.provider import LLMClient, extract_json
from jahan.model.enums import TOPIC_NAMES
from jahan.scenario.spec import NewsEvent, Scenario, ScenarioEvent

CLASSIFY_SYSTEM = """Classify news headlines for a society simulator.  For each headline return
{"i": index, "topic": one of [politics, economy, tech, health, sports, entertainment, science, religion,
family, local, world, lifestyle], "valence": number from -1 (bad news for ordinary people in that country)
to 1 (good news), "salience": number from 0 (minor) to 1 (everyone is talking about it)}.
Answer with a JSON list only.  Judge only from the headline text."""

BATCH = 40


def _fallback(h: Headline) -> dict:
    return {"topic": "world", "valence": 0.0, "salience": max(0.3, 0.75 - 0.05 * h.rank)}


def classify(headlines: list[Headline], client: LLMClient | None, sim_day: int) -> list[dict]:
    if client is None:
        return [_fallback(h) for h in headlines]
    out: list[dict] = []
    for start in range(0, len(headlines), BATCH):
        chunk = headlines[start : start + BATCH]
        listing = "\n".join(f"{i}. [{h.country}, {h.lang}] {h.title}" for i, h in enumerate(chunk))
        raw = client.complete(  # MissingRecording / BudgetExceeded propagate: replay must fail closed
            "classifier",
            CLASSIFY_SYSTEM,
            [{"role": "user", "content": listing}],
            max_tokens=4000,
            purpose="classify_headlines",
            sim_day=sim_day,
        )
        try:
            rows = {int(r["i"]): r for r in extract_json(raw)}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            rows = {}
        for i, h in enumerate(chunk):
            r = rows.get(i)
            if r is None or r.get("topic") not in TOPIC_NAMES:
                out.append(_fallback(h))
                continue
            try:
                valence = float(max(-1.0, min(1.0, float(r.get("valence", 0.0)))))
                salience = float(max(0.0, min(1.0, float(r.get("salience", 0.5)))))
            except (TypeError, ValueError):  # e.g. "positive" instead of a number
                out.append(_fallback(h))
                continue
            out.append({"topic": r["topic"], "valence": valence, "salience": salience})
    return out


def to_scenario(
    headlines: list[Headline],
    labels: list[dict],
    per_country: int,
    duration_days: int = 2,
    prompt: str = "",
    news_day: dt.date | None = None,
) -> Scenario:
    """Top headlines per country (by salience, then feed order) as NewsEvents in their country and language.

    ``news_day`` is the real UTC date of the fetch: stories published before it are already out at 00:00.
    """
    by_country: dict[str, list[tuple[Headline, dict]]] = {}
    seen: set[str] = set()
    for h, lab in zip(headlines, labels, strict=True):
        if h.key in seen:
            continue
        seen.add(h.key)
        by_country.setdefault(h.country, []).append((h, lab))
    events: list[ScenarioEvent] = []
    for iso in sorted(by_country):
        ranked = sorted(by_country[iso], key=lambda x: (-x[1]["salience"], x[0].rank))[:per_country]
        for h, lab in ranked:
            events.append(
                NewsEvent(
                    headline=h.title[:300],
                    topic=lab["topic"],
                    valence=lab["valence"],
                    accuracy=1.0,
                    salience=lab["salience"],
                    countries=[iso],
                    channel_kinds=h.channel_kinds,
                    start_day_offset=0,
                    duration_days=duration_days,
                    release_utc_hour=release_hour(h, news_day),
                    intensity=1.0,
                    language=h.lang,
                    origin="web",
                )
            )
    return Scenario(
        prompt=prompt or f"Live news ingest ({len(events)} headlines)",
        events=events,
        horizon_days=max(30, duration_days),
    )


def release_hour(h: Headline, news_day: dt.date | None = None) -> int:
    """UTC hour the story came out on the news day: its publication hour, 0 if it was published on an
    earlier day (it is already out when the day starts), noon when the feed gives no time."""
    if not h.published_utc:
        return 12
    try:
        when = dt.datetime.fromisoformat(h.published_utc)
    except ValueError:
        return 12
    if news_day is not None and when.date() < news_day:
        return 0
    return when.hour
