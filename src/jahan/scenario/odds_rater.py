"""Claude rates how new headlines bear on watched questions (run before each simulated day, outside the core).

For each open question, every tracked story created since the question was added and not yet rated is sent in
batches with the question text; Claude returns a direction from -1 (makes "No" much more likely) to +1 (makes
"Yes" much more likely), 0 when unrelated.  Claude never estimates the probability itself: the simulated people
who learn the story move their beliefs (sim/questions.py), and the odds are their average.

Ratings are stored in ``world.macro["odds"]`` as claims (so a reset restores them) and every answer is recorded,
so replays reproduce them without calling the model.  A row Claude did not answer cleanly is stored as 0
("fallback") so it is not asked again.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from jahan.llm.budget import BudgetExceeded
from jahan.llm.provider import LLMClient, extract_json
from jahan.llm.recordings import MissingRecording
from jahan.model.enums import EventKind, OriginKind
from jahan.model.world import World
from jahan.sim.questions import open_questions, question_items
from jahan.store.eventlog import EventLog

BATCH = 40
MAX_PER_QUESTION_PER_DAY = 200  # the rest waits for the next day

RATE_SYSTEM = """You judge how news headlines bear on one yes/no prediction-market question, for a society
simulator.  For each headline return {"i": index, "direction": number from -1 (makes "No" much more likely)
to 1 (makes "Yes" much more likely), "why": at most 12 words}.  Most headlines have no bearing: give them 0.
Judge only from the headline and the question; never estimate the probability itself.  Answer with a JSON
list only."""


def candidates(world: World, q: dict[str, Any]) -> list[int]:
    """Tracked stories created since the question was added and not rated yet (oldest first)."""
    it = world.items
    if it.n == 0:
        return []
    products = {int(x) for x in world.products["item_id"]}
    skip = products | question_items(world) | {int(k) for k in q["claims"]}
    ids = np.flatnonzero(
        (it["slot"] >= 0) & (it["created_day"] >= int(q["added_day"])) & (it["active_until"] >= world.day)
    )
    return [int(i) for i in ids if int(i) not in skip][:MAX_PER_QUESTION_PER_DAY]


def _listing(world: World, items: list[int]) -> str:
    labels = world.macro.get("item_labels", {})
    lines = []
    for k, i in enumerate(items):
        origin = OriginKind(int(world.items["origin_kind"][i])).name.lower()
        day = int(world.items["created_day"][i])
        lines.append(f"{k}. [{origin}, day {day}] {labels.get(str(i), f'item {i}')}")
    return "\n".join(lines)


def rate_new_items(client: LLMClient, world: World, log: EventLog) -> int:
    """Rate every unrated story against every open question.  Returns how many ratings were stored."""
    st = world.macro.get("odds")
    if st is None or st.get("rating_stopped"):
        return 0
    done = 0
    for qid, q in open_questions(world):
        items = candidates(world, q)
        for start in range(0, len(items), BATCH):
            chunk = items[start : start + BATCH]
            header = f"Question: {q['question']}\nResolves by: {q.get('resolves_by') or 'not stated'}\n"
            if q.get("description"):
                header += f"Resolution notes: {str(q['description'])[:400]}\n"
            try:
                raw = client.complete(
                    "classifier",
                    RATE_SYSTEM,
                    [{"role": "user", "content": header + "Headlines:\n" + _listing(world, chunk)}],
                    max_tokens=3000,
                    purpose="rate_odds",
                    sim_day=world.day,
                )
            except MissingRecording:
                raise  # replay must fail closed
            except (BudgetExceeded, RuntimeError) as e:
                st["rating_stopped"] = str(e)[:200]
                log.add(world.day, EventKind.ACTION_SKIPPED, -1, -1, -1, 0.0, -1, f"odds rating stopped: {e}"[:300])
                return done
            try:
                rows = {int(r["i"]): r for r in extract_json(raw) if isinstance(r, dict)}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                rows = {}
            for k, item in enumerate(chunk):
                r = rows.get(k)
                try:
                    d = float(np.clip(float(r["direction"]), -1.0, 1.0)) if r is not None else None
                except (KeyError, TypeError, ValueError):
                    d = None
                by = "llm" if d is not None else "fallback"
                why = str(r.get("why", ""))[:120] if r is not None else ""
                q["claims"][str(item)] = {
                    "direction": 0.0 if d is None else d,
                    "by": by,
                    "day": world.day,
                    "applied_day": None,
                    "why": why,
                    "moved": 0.0,
                }
                log.add(
                    world.day,
                    EventKind.ODDS_RATED,
                    -1,
                    -1,
                    item,
                    0.0 if d is None else d,
                    -1,
                    f"{qid} [{by}] {why}"[:300],
                )
                done += 1
    return done


class OddsRater:
    """A before-day hook: rates new stories against watched questions (see ``jahan run --odds``)."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.rated = 0

    def __call__(self, world: World, log: EventLog) -> None:
        self.rated += rate_new_items(self.client, world, log)

    def summary(self) -> str:
        return f"Claude rated {self.rated} headlines against watched questions."
