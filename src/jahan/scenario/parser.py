"""Free-text prompt -> validated Scenario (or clarifying questions), via the LLM layer.

The model receives the Scenario JSON schema and facts about the current world, and must answer with
JSON: either {"clarifying_questions": [...]} or {"scenario": {...}}.  Invalid output is retried once
with the validation error.  The model only proposes parameters; the engine computes all effects.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from jahan.llm.provider import LLMClient, extract_json
from jahan.model.world import World
from jahan.scenario.spec import NewsEvent, PromptScenario, Scenario

SYSTEM = """You translate a user's what-if prompt into a structured scenario for the Jahan society simulator.
Rules:
- Answer with JSON only: either {"clarifying_questions": ["..."]} when essential facts are missing
  (e.g. price, which countries, how large a move), or {"scenario": <object matching the schema>}.
- Use only the event types and fields in the schema.  Never describe outcomes; the simulator computes them.
- Day offsets are relative to the injection day.  Percent moves for macro paths are per day.
- Product prices: give price_usd unless the user states local prices.
- Statements must use a speaker key from the list of notable figures.
- A Federal Reserve rate decision or a US jobs report is an "announcement": give the announced value and
  what markets expected the day before; the engine computes how stocks, gold and the dollar react from
  published estimates.  If the text does not say what markets expected, ask a clarifying question; never
  guess it.  Do not add macro paths for the same announcement, and never invent market percentages.
- Keep "prompt" equal to the user's text.
"""

FEED_SYSTEM = (
    SYSTEM
    + """
This prompt describes FUTURE world events the user wants to feed into the simulation (news and market
moves only).  Use only "news", "macro" and "announcement" event types.
"""
)


@dataclass
class ParseResult:
    scenario: Scenario | None
    questions: list[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""


def world_facts(world: World) -> str:
    fx = world.macro.get("fx", [])
    lines = [
        f"Current simulated date: {world.date().isoformat()} (world day {world.day}).",
        "Countries (ISO3, currency, local currency per USD now): "
        + "; ".join(
            f"{c.iso3} {c.currency} {fx[i] if i < len(fx) else c.fx_local_per_usd:,.4g}"
            for i, c in enumerate(world.countries)
        ),
        f"Gold index {world.macro.get('gold_index', 100.0):.1f}, US stocks index "
        f"{world.macro.get('us_stocks_index', 100.0):.1f} (both 100 at the world's start).",
        f"Products already launched: {', '.join(world.product_names) or 'none'}.",
        "Notable figures (speaker keys): "
        + "; ".join(f"{e['key']} = {e['holder_name']}, {e['office']}" for e in world.notables),
        "Topics: politics, economy, tech, health, sports, entertainment, science, religion, family, local, "
        "world, lifestyle.",
        "Channel kinds: tv, newspaper, online, social, radio, specialist.",
        "Value dimensions for product attributes: self, money, culture, freedom, family, security.",
    ]
    return "\n".join(lines)


def parse_prompt(
    client: LLMClient,
    world: World,
    prompt: str,
    feed: bool = False,
    answers: dict[str, str] | None = None,
    first_future_offset: int | None = None,
) -> ParseResult:
    """Free text to a Scenario.  ``first_future_offset``: days from now to the first day without real news."""
    # The model sees (and must satisfy) the world-event schema only; actor events cannot come from a prompt.
    schema = json.dumps(PromptScenario.model_json_schema(), separators=(",", ":"))
    facts = world_facts(world)
    if first_future_offset is not None:
        first = world.date() + dt.timedelta(days=first_future_offset)
        facts += (
            f"\nReal news exists up to {first - dt.timedelta(days=1)}.  The first simulated day without real "
            f"news is {first} (start_day_offset {first_future_offset}); unless the user says otherwise, the "
            "events start there."
        )
    user = f"World facts:\n{facts}\n\nScenario JSON schema:\n{schema}\n\nUser prompt:\n{prompt}"
    if answers:
        user += "\n\nThe user answered your questions:\n" + "\n".join(f"- {q}: {a}" for q, a in answers.items())
    messages = [{"role": "user", "content": user}]
    role = "feed_parser" if feed else "scenario_parser"
    system = FEED_SYSTEM if feed else SYSTEM
    raw = client.complete(role, system, messages, max_tokens=2000, purpose="parse_prompt", sim_day=world.day)
    for attempt in range(2):
        try:
            data = extract_json(raw)
            if isinstance(data, dict) and data.get("clarifying_questions"):
                return ParseResult(None, [str(q) for q in data["clarifying_questions"]], raw)
            payload = data.get("scenario", data) if isinstance(data, dict) else data
            sc = Scenario.model_validate(PromptScenario.model_validate(payload).model_dump(mode="json"))
            if feed and any(ev.type not in ("news", "macro", "announcement") for ev in sc.events):
                raise ValueError("feed prompts may only contain news, macro and announcement events")
            for ev in sc.events:  # the model cannot pass a made-up story off as real fetched news
                if isinstance(ev, NewsEvent):
                    ev.origin = "user_feed" if feed else "scenario"
            return ParseResult(sc, [], raw)
        except (ValueError, ValidationError) as e:
            if attempt == 1:
                return ParseResult(None, [], raw, error=str(e))
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was invalid: {e}. Reply with corrected JSON only."},
            ]
            raw = client.complete(
                role, system, messages, max_tokens=2000, purpose="parse_prompt_retry", sim_day=world.day
            )
    return ParseResult(None, [], raw, error="unreachable")
