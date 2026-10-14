"""What a person says -> a structured claim (LLM), or the claim given directly with flags (no LLM).

The model may only choose topic, product, a one-line headline and one of five stance levels.  It never
sees or sets salience, reach or credibility: those come from the engine.  Emphatic wording can at most
move the stance to an end of the five-level scale.
"""

from __future__ import annotations

import json
from typing import Any

from jahan.actors.actions import STANCES, Claim, quantize_stance
from jahan.llm.provider import LLMClient, extract_json
from jahan.model.enums import TOPIC_NAMES
from jahan.model.world import World

TOPICS = sorted(TOPIC_NAMES)

EXTRACT_SYSTEM = f"""You turn what a person in a society simulator says into one structured claim.
Answer with JSON only:
{{"headline": "<the claim in at most 20 plain words, in English>",
 "topic": one of {TOPICS},
 "stance": one of {list(STANCES)}  (-1 = very bad news for the listener, 0 = neutral, 1 = very good news),
 "about_product": one of the listed products, or null}}
Do not judge whether the claim is true and do not add facts.  If the text makes no claim, use stance 0."""

PROFILE_SYSTEM = """You turn a description of a person into profile fields for a society simulator.
Answer with JSON only, using only fields the description supports (omit the rest):
{"sex": "F"|"M", "age": int, "first_name": str, "last_name": str, "region": "urban"|"rural",
 "education": ISCED level 0-8, "income_month": number in local currency, "languages": [ISO 639-1 codes],
 "values": {"self","money","culture","freedom","family","security"} weights summing to 1,
 "visibility": 0-10 (log10 of the audience they can reach; 1 = family, 2.5 = a teacher or imam,
 4 = a local official or journalist)}
Never invent fields that the description does not support."""


def claim_from_flags(text: str, topic: str = "world", stance: float = 0.0, about: str | None = None) -> Claim:
    headline = " ".join(text.split())[:300] or "(no words)"
    return Claim(headline=headline, topic=topic, stance=quantize_stance(stance), about_product=about, text=text)


def extract_claim(client: LLMClient, world: World, text: str) -> Claim:
    products = world.product_names
    system = EXTRACT_SYSTEM + f"\nProducts in this world: {json.dumps(products)}."
    raw = client.complete(
        "text_extraction",
        system,
        [{"role": "user", "content": text}],
        max_tokens=300,
        purpose="extract_claim",
        sim_day=world.day,
    )
    try:
        data = extract_json(raw)
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    topic = data.get("topic") if data.get("topic") in TOPIC_NAMES else "world"
    try:
        stance = quantize_stance(max(-1.0, min(1.0, float(data.get("stance", 0.0)))))
    except (TypeError, ValueError):
        stance = 0.0
    about = data.get("about_product")
    about = about if about in products else None
    headline = " ".join(str(data.get("headline") or text).split())[:300] or "(no words)"
    return Claim(headline=headline, topic=str(topic), stance=stance, about_product=about, text=text)


def profile_from_text(client: LLMClient, world: World, text: str) -> dict[str, Any]:
    """CreateAgent fields supported by a free-text description (unsupported ones are sampled later)."""
    raw = client.complete(
        "text_extraction",
        PROFILE_SYSTEM,
        [{"role": "user", "content": text}],
        max_tokens=400,
        purpose="describe_agent",
        sim_day=world.day,
    )
    try:
        data = extract_json(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    keep = {
        "sex",
        "age",
        "first_name",
        "last_name",
        "region",
        "education",
        "income_month",
        "languages",
        "values",
        "visibility",
    }
    return {k: v for k, v in data.items() if k in keep and v is not None}
