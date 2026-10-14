"""Records of posts made inside the world (by players, created people and public figures) and how far they got.

``world.macro["posts"]`` maps a post id to who posted what, where, and, updated daily while the post's item is
tracked, how many simulated people have heard it, how many people they stand for (population weights), how
many were convinced by the poster, and how often ordinary people reposted it.  When the item stops being
tracked the numbers freeze.  Records older than ``KEEP_DAYS`` are dropped.  Everything lives in
``world.macro``, so a reset restores it and a replay rebuilds it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jahan.model.world import World

KEEP_DAYS = 60


def record_post(
    world: World,
    *,
    actor: int,
    item: int,
    kind: str,
    audience: str,
    headline: str,
    text: str = "",
    requested: str | None = None,
    session: int = -1,
    hour: int = 12,
    circle: int = 0,
    community: int = 0,
    channels: int = 0,
    stance: float = 0.0,
) -> int:
    """Add a post record (``kind`` post | statement); returns its id."""
    posts = world.macro.setdefault("posts", {})
    pid = int(world.macro.get("next_post_id", 0))
    world.macro["next_post_id"] = pid + 1
    posts[str(pid)] = {
        "day": world.day,
        "kind": kind,
        "actor": int(actor),
        "item": int(item),
        "session": int(session),
        "audience": audience,
        "requested": requested or audience,
        "headline": headline,
        "text": text,
        "hour": int(hour),
        "stance": float(stance),
        "circle": int(circle),
        "community": int(community),
        "channels": int(channels),
        "reached": 0,
        "reached_people": 0.0,
        "believed": 0,
        "reposts": 0,
        "repost_people": 0.0,
        "final": False,
    }
    return pid


def update_post_stats(world: World) -> None:
    """Refresh reach numbers of live posts; freeze those whose item is no longer tracked; drop old records."""
    posts = world.macro.get("posts")
    if not posts:
        return
    pop = world.pop
    n = pop.n
    bel = world.beliefs.m
    for key in [k for k, v in posts.items() if v["day"] < world.day - KEEP_DAYS]:
        del posts[key]
    for rec in posts.values():
        if rec.get("final"):
            continue
        item = int(rec["item"])
        slot = int(world.items["slot"][item]) if 0 <= item < world.items.n else -1
        if slot < 0:
            rec["final"] = True
            continue
        aware = (bel["aware"][:n, slot] == 1) & pop["alive"][:n] & (pop["id"][:n] != rec["actor"])
        # convinced by the poster: their latest accepted source is the poster and, for a claim with a side,
        # their view leans the poster's way
        convinced = aware & (bel["heard_from"][:n, slot] == rec["actor"])
        stance = float(rec.get("stance", 0.0))
        if stance:
            convinced &= (bel["belief"][:n, slot].astype(np.float64) - 0.5) * stance > 0
        rec["reached"] = int(aware.sum())
        rec["reached_people"] = round(float(pop["pop_weight"][:n][aware].sum()), 1)
        rec["believed"] = int(convinced.sum())


def posts_of(world: World, actor: int, since_day: int) -> list[dict[str, Any]]:
    """This agent's post records from ``since_day`` on, oldest first."""
    rows = [dict(v, id=int(k)) for k, v in world.macro.get("posts", {}).items() if v["actor"] == actor]
    return sorted((r for r in rows if r["day"] >= since_day), key=lambda r: (r["day"], r["id"]))


def people(x: float) -> str:
    """5_612_000 -> 'about 5.6 million'."""
    if x >= 1e9:
        return f"about {x / 1e9:.1f} billion"
    if x >= 1e6:
        return f"about {x / 1e6:.1f} million"
    if x >= 1e3:
        return f"about {x / 1e3:,.0f} thousand"
    return f"about {x:,.0f}"
