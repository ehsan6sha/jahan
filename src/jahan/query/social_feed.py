"""The social feed of a world: real posts from the last fetches and posts made inside the world.

Real rows come from ``world.macro["social"]["feed"]`` (written by a fetch's digest), joined with the live reach
of the posts that became tracked items.  Virtual rows come from ``world.macro["posts"]`` (players' posts and
public figures' statements, with reach and reposts).  Ordinary people found by topic search never appear: their
posts only count towards trending topics.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from jahan.actors.engine import display_name
from jahan.model.world import World

REAL, VIRTUAL = "real", "virtual"


def _date(world: World, day: int) -> str:
    return (world.config.start_date + dt.timedelta(days=int(day))).isoformat()


def _item_of_post(world: World) -> dict[str, int]:
    """post key ("x:123") -> item id, for real posts that became tracked items."""
    out = {}
    for item, meta in world.macro.get("social", {}).get("items", {}).items():
        out[f"{meta.get('platform')}:{meta.get('post_id')}"] = int(item)
    return out


def _live_reach(world: World, item: int) -> dict[str, Any]:
    slot = int(world.items["slot"][item]) if 0 <= item < world.items.n else -1
    if slot < 0:
        return {"reached": None, "reached_people": None}
    n = world.pop.n
    aware = (world.beliefs.m["aware"][:n, slot] == 1) & world.pop["alive"][:n]
    return {
        "reached": int(aware.sum()),
        "reached_people": round(float(world.pop["pop_weight"][:n][aware].sum()), 1),
    }


def feed_rows(world: World, country: str | None = None, kind: str = "all", limit: int = 50) -> list[dict[str, Any]]:
    """Newest first.  ``kind``: real | virtual | all."""
    rows: list[dict[str, Any]] = []
    codes = [c.iso3 for c in world.countries]
    if kind in (REAL, "all"):
        items = _item_of_post(world)
        for iso, feed in world.macro.get("social", {}).get("feed", {}).items():
            if country and iso != country:
                continue
            for r in feed:
                item = items.get(str(r.get("post_key")), -1) if r.get("tracked") else -1
                reach = _live_reach(world, item) if item >= 0 else {"reached": None, "reached_people": None}
                rows.append(
                    {
                        "kind": REAL,
                        "day": int(r.get("world_day", 0)),
                        "date": str(r.get("created_utc", ""))[:10] or _date(world, int(r.get("world_day", 0))),
                        "country": iso,
                        "platform": r.get("platform"),
                        "handle": r.get("handle"),
                        "name": r.get("name"),
                        "followers": r.get("followers"),
                        "lang": r.get("lang"),
                        "text": r.get("text"),
                        "url": r.get("url"),
                        "topic": r.get("topic"),
                        "valence": r.get("valence"),
                        "likes": r.get("likes"),
                        "reposts": r.get("reposts"),
                        "views": r.get("views"),
                        "is_repost": bool(r.get("is_repost")),
                        "tracked": item >= 0,
                        "item": item,
                        **reach,
                        "_order": (int(r.get("world_day", 0)), str(r.get("created_utc", ""))),
                    }
                )
    if kind in (VIRTUAL, "all"):
        for pid, p in world.macro.get("posts", {}).items():
            row = world.agent_row(int(p["actor"])) if 0 <= int(p["actor"]) < world.pop.n else -1
            iso = codes[int(world.pop["country"][row])] if row >= 0 else None
            if country and iso != country:
                continue
            rows.append(
                {
                    "kind": VIRTUAL,
                    "id": int(pid),
                    "day": int(p["day"]),
                    "date": _date(world, int(p["day"])),
                    "country": iso,
                    "post_kind": p.get("kind", "post"),
                    "actor": int(p["actor"]),
                    "name": display_name(world, row) if row >= 0 else str(p["actor"]),
                    "audience": p["audience"],
                    "headline": p["headline"],
                    "text": p.get("text") or p["headline"],
                    "item": int(p["item"]),
                    "reached": int(p.get("reached", 0)),
                    "reached_people": float(p.get("reached_people", 0.0)),
                    "believed": int(p.get("believed", 0)),
                    "reposts": int(p.get("reposts", 0)),
                    "repost_people": float(p.get("repost_people", 0.0)),
                    "channels": int(p.get("channels", 0)),
                    "_order": (int(p["day"]), f"{int(p.get('hour', 12)):02d}:{int(pid):08d}"),
                }
            )
    rows.sort(key=lambda r: r["_order"], reverse=True)
    for r in rows:
        del r["_order"]
    return rows[:limit]


def figure_recent_posts(world: World, key: str, limit: int = 5) -> list[dict[str, Any]]:
    """A public figure's latest real posts from the fetches (oldest first)."""
    rows = world.macro.get("social", {}).get("figures", {}).get(key, [])
    return list(rows)[-limit:]


def recent_posts_lines(world: World, key: str) -> list[str]:
    """Lines for a figure's prompt: what they really posted lately (fetched text, labelled as real)."""
    rows = figure_recent_posts(world, key)
    if not rows:
        return []
    fetched = world.macro.get("social", {}).get("last_fetch", {}).get("fetch_day", "?")
    out = [f"What you said publicly lately (real posts, fetched {fetched}):"]
    for r in rows:
        verb = "reposted" if r.get("is_repost") else "posted"
        out.append(
            f"- {str(r.get('created_utc', ''))[:10]} {verb} on {r.get('platform')}: {str(r.get('text', ''))[:200]}"
        )
    return out


def trending_lines(world: World, iso: str, limit: int = 3) -> list[str]:
    rows = world.macro.get("social", {}).get("trending", {}).get(iso, [])[:limit]
    if not rows:
        return []
    out = ["Trending on social media in your country (post counts from the last fetch; uncalibrated):"]
    for r in rows:
        out.append(f"- {r['topic']}: {r['n']} posts, average tone {r['mean_valence']:+.2f}")
    return out


def feed_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No posts yet. Fetch real posts with `jahan ingest`, or post with `jahan post`."
    out = []
    for r in rows:
        if r["kind"] == REAL:
            who = f"@{r['handle']}" if r.get("handle") else str(r.get("name") or "?")
            reach = (
                f"reached {r['reached']:,} people in the world"
                if r.get("reached") is not None
                else "not tracked (shown for context only)"
            )
            out.append(
                f"[{r['date']}] {r['country']} {r['platform']} {who}: {r['text']}\n"
                f"    {r.get('topic')}, tone {float(r.get('valence') or 0):+.1f}; "
                f"{r.get('likes') or 0:,} likes; {reach}" + (f"\n    {r['url']}" if r.get("url") else "")
            )
        else:
            out.append(
                f"[{r['date']}] {r['country']} in the world, {r['name']} ({r['post_kind']} to {r['audience']}): "
                f"{r['text']}\n    reached {r['reached']:,} simulated people, {r['believed']:,} convinced, "
                f"{r['reposts']:,} reposts  [simulated]"
            )
    return "\n".join(out)
