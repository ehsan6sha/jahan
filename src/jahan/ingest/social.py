"""Real posts from social networks that can be read for free, as material for in-world news.

Sources and accounts live in ``data/feeds/social.yaml``.  Every fetch goes through ``feeds.fetch_url``, so it
is recorded per (purpose, url, fetch day) and a replay never touches the network.  Every URL is built from
the fetch day, never from the clock, so a replay asks for exactly the recorded URLs.

* X through an unofficial reader (unifapi.com): searches like ``(from:POTUS OR from:realDonaldTrump)
  include:nativeretweets since:2026-10-12``.  The operator ``include:nativeretweets`` matters: some accounts
  (@POTUS) mostly repost, and without it the search returned posts from 2025.  At most ``max_searches``
  searches per ingest (the free tier allows about ten per ten minutes): public figures first, then topics.
* Bluesky (official public API): author feeds and post search.
* Mastodon (official API of each server): an account's statuses and hashtag timelines.
* Telegram public channel previews (t.me/s/<channel>).

A failing query becomes a problem line ("ISO/platform: ..."), never an abort.  Posts by public figures keep
their reposts (they chose to amplify them); topic searches drop reposts and replies.  Posts by ordinary people
(fewer than ``PUBLIC_FOLLOWERS`` followers) only count towards trending topics: their words are not spread,
stored in the world or shown.  The raw responses stay in the run's recordings (git-ignored), for replay.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jahan.ingest.classify import release_hour
from jahan.ingest.feeds import fetch_url
from jahan.llm.recordings import Recordings
from jahan.model.social import Account, SocialConfig, load_social
from jahan.scenario.spec import Scenario, SocialDigest, SocialPostEvent

PURPOSE = {"x": "social_x", "bluesky": "social_bsky", "mastodon": "social_masto", "telegram": "social_tg"}
MAX_PER_QUERY = 30  # posts kept per query
_MENTION = re.compile(r"(?<![\w@])@([A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?)")
_TAGS = re.compile(r"<[^>]+>")


@dataclass
class SocialPost:
    platform: str
    post_id: str
    author_handle: str
    author_name: str
    author_followers: int  # -1 = unknown
    text: str
    created_utc: str
    url: str
    lang: str
    country: str
    query_kind: str  # figure | channel | topic
    query_key: str  # roster key, channel handle, or "lang:topic"
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    quotes: int = 0
    views: int = 0
    is_repost: bool = False
    is_reply: bool = False
    mentions: list[str] = field(default_factory=list)
    rank: int = 0

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.post_id}"

    @property
    def title(self) -> str:
        """One line of text, as ``classify`` and the item label use it."""
        return " ".join(self.text.split())[:300]

    @property
    def published_utc(self) -> str:
        return self.created_utc


def mentions_in(text: str) -> list[str]:
    """@handles in a post, in order, without duplicates."""
    out: list[str] = []
    for m in _MENTION.findall(text):
        h = "@" + m
        if h not in out:
            out.append(h)
    return out


def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _iso(ts: str) -> str:
    """An ISO UTC timestamp from a platform's time string ("" if unreadable)."""
    if not ts:
        return ""
    try:
        d = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    return d.astimezone(dt.UTC).isoformat()


def _recent(post: SocialPost, since: dt.date) -> bool:
    if not post.created_utc:
        return True  # undated: keep; the ranking uses engagement, not time
    return dt.datetime.fromisoformat(post.created_utc).date() >= since


def parse_count(s: str) -> int:
    """Telegram-style counts: "770K" -> 770000, "1.2M" -> 1200000, "714" -> 714."""
    s = s.strip().replace(",", "")
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(s[-1:].upper(), 1)
    try:
        return int(round(float(s[:-1] if mult > 1 else s) * mult))
    except ValueError:
        return -1


# ---------------------------------------------------------------- parsers (pure functions of the bytes)
def parse_x(body: bytes, country: str, lang: str) -> list[SocialPost]:
    """Posts from an unifapi X search; ``query_kind``/``query_key`` are filled in by the caller."""
    data = json.loads(body.decode("utf-8"))
    out = []
    for item in (data.get("posts") or {}).get("items") or []:
        author = item.get("author") or {}
        text = str(item.get("text") or "")
        m = item.get("metrics") or {}
        handle = str(author.get("handle") or "")
        out.append(
            SocialPost(
                platform="x",
                post_id=str(item.get("id") or ""),
                author_handle=handle,
                author_name=str(author.get("name") or handle),
                author_followers=_int(author.get("followers")) if author.get("followers") is not None else -1,
                text=text,
                created_utc=_iso(str(item.get("createdAt") or "")),
                url=str(item.get("url") or ""),
                lang=str(item.get("language") or lang),
                country=country,
                query_kind="",
                query_key="",
                likes=_int(m.get("likes")),
                reposts=_int(m.get("reposts")),
                replies=_int(m.get("replies")),
                quotes=_int(m.get("quotes")),
                views=_int(m.get("views")),
                is_repost=text.startswith("RT @") or bool(item.get("repostedBy")),
                is_reply=bool(item.get("inReplyToPostId")),
                mentions=mentions_in(text),
            )
        )
    return [p for p in out if p.post_id]


def _bsky_post(
    p: dict[str, Any], country: str, kind: str, key: str, lang: str, followers: int, repost: bool
) -> SocialPost:
    author = p.get("author") or {}
    record = p.get("record") or {}
    handle = str(author.get("handle") or "")
    rkey = str(p.get("uri") or "").rsplit("/", 1)[-1]
    text = str(record.get("text") or "")
    langs = record.get("langs") or []
    return SocialPost(
        platform="bluesky",
        post_id=str(p.get("uri") or ""),
        author_handle=handle,
        author_name=str(author.get("displayName") or handle),
        author_followers=followers,
        text=text,
        created_utc=_iso(str(record.get("createdAt") or "")),
        url=f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else "",
        lang=str(langs[0]) if langs else lang,
        country=country,
        query_kind=kind,
        query_key=key,
        likes=_int(p.get("likeCount")),
        reposts=_int(p.get("repostCount")),
        replies=_int(p.get("replyCount")),
        quotes=_int(p.get("quoteCount")),
        is_repost=repost,
        mentions=mentions_in(text),
    )


def parse_bsky_feed(body: bytes, country: str, kind: str, key: str, lang: str, followers: int) -> list[SocialPost]:
    data = json.loads(body.decode("utf-8"))
    out = []
    for it in data.get("feed") or []:
        repost = "reasonRepost" in str((it.get("reason") or {}).get("$type", ""))
        out.append(_bsky_post(it.get("post") or {}, country, kind, key, lang, followers, repost))
    return [p for p in out if p.post_id]


def parse_bsky_search(body: bytes, country: str, key: str, lang: str) -> list[SocialPost]:
    data = json.loads(body.decode("utf-8"))
    out = [_bsky_post(p, country, "topic", key, lang, -1, False) for p in data.get("posts") or []]
    return [p for p in out if p.post_id]


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    return html.unescape(_TAGS.sub("", s)).strip()


def parse_mastodon(body: bytes, country: str, kind: str, key: str, lang: str) -> list[SocialPost]:
    out = []
    for st in json.loads(body.decode("utf-8")) or []:
        repost = st.get("reblog") is not None
        src = st["reblog"] if repost else st
        acct = src.get("account") or {}
        text = _strip_html(str(src.get("content") or ""))
        out.append(
            SocialPost(
                platform="mastodon",
                post_id=str(st.get("uri") or st.get("id") or ""),
                author_handle=str(acct.get("acct") or ""),
                author_name=str(acct.get("display_name") or acct.get("acct") or ""),
                author_followers=_int(acct.get("followers_count")) if acct.get("followers_count") is not None else -1,
                text=text,
                created_utc=_iso(str(st.get("created_at") or "")),
                url=str(src.get("url") or st.get("url") or ""),
                lang=str(src.get("language") or lang),
                country=country,
                query_kind=kind,
                query_key=key,
                likes=_int(src.get("favourites_count")),
                reposts=_int(src.get("reblogs_count")),
                replies=_int(src.get("replies_count")),
                quotes=_int(src.get("quotes_count")),
                is_repost=repost,
                mentions=mentions_in(text),
            )
        )
    return [p for p in out if p.post_id and p.text]


_TG_SPLIT = re.compile(r'<div class="tgme_widget_message[ "][^>]*data-post="([^"]+)"')
_TG_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_TG_TIME = re.compile(r'<time datetime="([^"]+)"')
_TG_VIEWS = re.compile(r'<span class="tgme_widget_message_views">([^<]+)</span>')
_TG_SUBS = re.compile(r'<span class="counter_value">([^<]+)</span>\s*<span class="counter_type">subscribers</span>')


def parse_telegram(body: bytes, country: str, account: Account) -> tuple[list[SocialPost], int]:
    """Posts from a public channel preview, and the subscriber count (-1 if not shown)."""
    page = body.decode("utf-8", errors="replace")
    subs = _TG_SUBS.search(page)
    followers = parse_count(subs.group(1)) if subs else (account.followers or -1)
    marks = list(_TG_SPLIT.finditer(page))
    out = []
    for i, m in enumerate(marks):
        chunk = page[m.start() : marks[i + 1].start() if i + 1 < len(marks) else len(page)]
        tm = _TG_TEXT.search(chunk)
        if not tm:
            continue
        text = _strip_html(tm.group(1))
        when = _TG_TIME.search(chunk)
        views = _TG_VIEWS.search(chunk)
        post = m.group(1)
        out.append(
            SocialPost(
                platform="telegram",
                post_id=post,
                author_handle=account.handle,
                author_name=account.name or account.handle,
                author_followers=followers,
                text=text,
                created_utc=_iso(when.group(1)) if when else "",
                url=f"https://t.me/{post}",
                lang=account.lang or "",
                country=country,
                query_kind="channel",
                query_key=account.handle,
                views=parse_count(views.group(1)) if views else 0,
                mentions=mentions_in(text),
            )
        )
    return [p for p in out if p.text], followers


# ---------------------------------------------------------------- collection
@dataclass
class _Ctx:
    cfg: SocialConfig
    recordings: Recordings
    fetch_day: str
    since: dt.date
    fetcher: Callable[[str], bytes] | None
    problems: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def get(self, platform: str, url: str) -> bytes:
        return fetch_url(url, self.recordings, self.fetch_day, PURPOSE[platform], self.fetcher)

    def count(self, platform: str, what: str, n: int = 1) -> None:
        st = self.stats.setdefault(platform, {"queries": 0, "posts": 0, "failed": 0})
        st[what] = st.get(what, 0) + n


def _x_query(handles: list[str], since: dt.date) -> str:
    who = " OR ".join(f"from:{h}" for h in handles)
    if len(handles) > 1:
        who = f"({who})"
    return f"{who} include:nativeretweets since:{since.isoformat()}"


def _x_topic_query(q: str, lang: str, since: dt.date) -> str:
    return f"{q} lang:{lang} since:{since.isoformat()}"  # replies are dropped after parsing


@dataclass
class _XJob:
    country: str
    query: str
    lang: str
    accounts: dict[str, tuple[str, str]]  # casefolded handle -> (query_kind, query_key); empty for a topic
    topic_key: str = ""


def collect_social(
    data_dir: Path,
    countries: list[str],
    recordings: Recordings,
    fetch_day: str,
    fetcher: Callable[[str], bytes] | None = None,
    disabled: set[str] | None = None,
) -> tuple[list[SocialPost], list[str], dict[str, dict[str, int]]]:
    """Posts for the given countries in a fixed order, the problems met, and per-platform counts.

    Order: countries as given; within a country, public figures (by roster key), channels (file order),
    topics (by language, then topic, then platform).  The X search budget goes to the accounts of every
    country first, then to topic searches.
    """
    cfg = load_social(data_dir)
    off = set(disabled or set()) | {p for p, s in cfg.sources.items() if not s.enabled}
    day = dt.date.fromisoformat(fetch_day)
    ctx = _Ctx(cfg, recordings, fetch_day, day - dt.timedelta(days=cfg.lookback_days), fetcher)
    posts: list[SocialPost] = []
    x_accounts: list[_XJob] = []
    x_topics: list[_XJob] = []
    for iso in countries:
        cs = cfg.countries.get(iso)
        if cs is None:
            continue
        lang0 = _country_lang(data_dir, iso)
        grouped: dict[str, tuple[str, str]] = {}
        order: list[str] = []
        for key in sorted(cs.figures):
            for a in cs.figures[key]:
                if a.platform == "x" and a.own_query:
                    x_accounts.append(
                        _XJob(iso, _x_query([a.handle], ctx.since), lang0, {a.handle.casefold(): ("figure", key)})
                    )
                elif a.platform == "x":
                    grouped[a.handle.casefold()] = ("figure", key)
                    order.append(a.handle)
                elif a.platform not in off:
                    posts += _fetch_account(ctx, iso, "figure", key, a, lang0)
        for a in cs.channels:
            if a.platform == "x":
                grouped[a.handle.casefold()] = ("channel", a.handle)
                order.append(a.handle)
        if order:
            x_accounts.append(_XJob(iso, _x_query(order, ctx.since), lang0, grouped))
        for a in cs.channels:
            if a.platform != "x" and a.platform not in off:
                posts += _fetch_account(ctx, iso, "channel", a.handle, a, a.lang or lang0)
        for lang in sorted(cs.topics):
            for topic in sorted(cs.topics[lang]):
                for platform, q in sorted(cs.topics[lang][topic].items()):
                    key = f"{lang}:{topic}"
                    if platform == "x":
                        x_topics.append(_XJob(iso, _x_topic_query(q, lang, ctx.since), lang, {}, key))
                    elif platform not in off:
                        posts += _search(ctx, iso, platform, q, lang, key)
    if "x" not in off:
        posts += _run_x(ctx, x_accounts + x_topics)
    out = [
        p
        for p in posts
        if p.text and not (p.query_kind == "topic" and (p.is_repost or p.is_reply)) and _recent(p, ctx.since)
    ]
    seen: set[str] = set()
    uniq = []
    for p in out:
        if p.key not in seen:
            seen.add(p.key)
            uniq.append(p)
    for i, p in enumerate(uniq):
        p.rank = i
    return uniq, ctx.problems, ctx.stats


def _country_lang(data_dir: Path, iso: str) -> str:
    with (data_dir / "countries" / f"{iso}.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return str((raw.get("languages_official") or ["en"])[0])


def _run_x(ctx: _Ctx, jobs: list[_XJob]) -> list[SocialPost]:
    src = ctx.cfg.sources["x"]
    out: list[SocialPost] = []
    budget = src.max_searches
    for job in jobs:
        if budget <= 0:
            ctx.problems.append(
                f"{job.country}/x: search budget ({src.max_searches} per ingest) used up; skipped {job.query}"
            )
            continue
        budget -= 1
        url = f"{src.search_url}{urllib.parse.quote(job.query, safe='')}"
        ctx.count("x", "queries")
        try:
            got = parse_x(ctx.get("x", url), job.country, job.lang)
        except Exception as e:  # noqa: BLE001 - one failing search must not stop the others
            ctx.count("x", "failed")
            ctx.problems.append(f"{job.country}/x: {job.query}: {e}")
            if "429" in str(e) or "403" in str(e):
                budget = 0  # rate-limited or refused: the remaining searches would fail the same way
            continue
        kept = []
        for p in got:
            if job.topic_key:
                p.query_kind, p.query_key = "topic", job.topic_key
            elif p.author_handle.casefold() in job.accounts:
                p.query_kind, p.query_key = job.accounts[p.author_handle.casefold()]
            else:
                continue  # an account the search did not ask for
            kept.append(p)
        kept = kept[:MAX_PER_QUERY]
        ctx.count("x", "posts", len(kept))
        out += kept
    return out


def _fetch_account(ctx: _Ctx, iso: str, kind: str, key: str, a: Account, lang: str) -> list[SocialPost]:
    platform = a.platform
    ctx.count(platform, "queries")
    try:
        if platform == "bluesky":
            src = ctx.cfg.sources["bluesky"]
            q = urllib.parse.urlencode({"actor": a.handle, "filter": "posts_no_replies", "limit": MAX_PER_QUERY})
            got = parse_bsky_feed(
                ctx.get(platform, f"{src.author_feed_url}?{q}"), iso, kind, key, lang, a.followers or -1
            )
        elif platform == "mastodon":
            base = f"https://{a.instance}/api/v1"
            acct = json.loads(ctx.get(platform, f"{base}/accounts/lookup?acct={urllib.parse.quote(a.handle)}"))
            q = urllib.parse.urlencode({"limit": 20, "exclude_replies": "true"})
            got = parse_mastodon(ctx.get(platform, f"{base}/accounts/{acct['id']}/statuses?{q}"), iso, kind, key, lang)
        elif platform == "telegram":
            src = ctx.cfg.sources["telegram"]
            got, _ = parse_telegram(ctx.get(platform, f"{src.preview_url}{urllib.parse.quote(a.handle)}"), iso, a)
            if not got:
                raise ValueError("no public posts on the preview page (wrong handle or a private channel)")
        else:
            raise ValueError(f"unsupported platform {platform}")
    except Exception as e:  # noqa: BLE001 - one failing account must not stop the others
        ctx.count(platform, "failed")
        ctx.problems.append(f"{iso}/{platform}: {a.handle}: {e}")
        return []
    got = got[:MAX_PER_QUERY]
    ctx.count(platform, "posts", len(got))
    return got


def _search(ctx: _Ctx, iso: str, platform: str, q: str, lang: str, key: str) -> list[SocialPost]:
    ctx.count(platform, "queries")
    try:
        if platform == "bluesky":
            src = ctx.cfg.sources["bluesky"]
            since = f"{ctx.since.isoformat()}T00:00:00Z"
            params = urllib.parse.urlencode({"q": q, "lang": lang, "limit": 25, "sort": "top", "since": since})
            got = parse_bsky_search(ctx.get(platform, f"{src.search_url}?{params}"), iso, key, lang)
        elif platform == "mastodon":
            tag = urllib.parse.quote(q.lstrip("#"))
            got = parse_mastodon(
                ctx.get(platform, f"https://mastodon.social/api/v1/timelines/tag/{tag}?limit=20"),
                iso,
                "topic",
                key,
                lang,
            )
        else:
            raise ValueError(f"no topic search for {platform}")
    except Exception as e:  # noqa: BLE001 - one failing search must not stop the others
        ctx.count(platform, "failed")
        ctx.problems.append(f"{iso}/{platform}: {q}: {e}")
        return []
    got = got[:MAX_PER_QUERY]
    ctx.count(platform, "posts", len(got))
    return got


# ---------------------------------------------------------------- posts -> scenario
# Ordinary people found by topic search (fewer followers than this) are never named, quoted or spread in
# the world: their posts only count towards trending topics.  Public figures, institutions and channels are.
PUBLIC_FOLLOWERS = 100_000
FEED_ROWS = 30  # feed rows per country in a digest
FIGURE_POSTS = 5  # recent posts kept per public figure


def author_visibility(post: SocialPost, roster_visibility: dict[str, float]) -> float:
    """log10 of the audience: a roster figure's visibility, else from followers (HEURISTIC: 2.0, a few
    hundred people, when the platform does not say)."""
    if post.query_kind == "figure" and post.query_key in roster_visibility:
        return float(roster_visibility[post.query_key])
    if post.author_followers < 0:
        return 2.0
    return float(min(max(math.log10(post.author_followers + 1), 1.0), 9.0))


def engagement(post: SocialPost) -> float:
    """0..1 from interactions (HEURISTIC: a million likes-and-reposts is 1)."""
    return float(min(max(math.log10(1 + post.likes + 2 * post.reposts + 2 * post.quotes) / 6.0, 0.0), 1.0))


def platform_share(cfg: SocialConfig, iso: str, platform: str) -> float:
    """The platform's users as a share of the country's internet users (DataReportal), else a default."""
    usage = cfg.platform_usage.countries.get(iso)
    users = usage.platforms_m.get(platform) if usage else None
    if usage is None or users is None or not usage.internet_users_m:
        return float(cfg.platform_usage.default_share.get(platform, 0.05))
    return float(min(max(users / usage.internet_users_m, 0.05), 1.0))


def is_public_author(post: SocialPost) -> bool:
    """Public figures, institutions and channels are named; ordinary people found by topic search are not."""
    return post.query_kind != "topic" or post.author_followers >= PUBLIC_FOLLOWERS


def to_social_scenario(
    posts: list[SocialPost],
    labels: list[dict[str, Any]],
    cfg: SocialConfig,
    per_country: int,
    roster_visibility: dict[str, float],
    language_code: Callable[[str], str],
    reachable: set[tuple[str, str]],
    news_day: dt.date,
    prompt: str = "",
    duration_days: int = 2,
) -> tuple[Scenario, dict[str, int]]:
    """Top posts per country as SocialPostEvents plus one SocialDigest, and counts of what was dropped.

    Selection (HEURISTIC): visibility x (0.5 + 0.5 engagement) x (0.5 + 0.5 model salience).  The model's
    salience only ranks; the event's reach is computed by the engine from visibility and engagement.  Posts in
    a language with no social channel in their country are dropped (nobody there would read them).
    """
    dropped = {"no_channel": 0, "private": 0}
    by_country: dict[str, list[tuple[float, SocialPost, dict[str, Any]]]] = {}
    for post, lab in zip(posts, labels, strict=True):
        if not is_public_author(post):
            dropped["private"] += 1  # ordinary people's words are not spread or shown; they count in trending
            continue
        code = language_code(post.lang)
        if (post.country, code) not in reachable:
            dropped["no_channel"] += 1
            continue
        vis = author_visibility(post, roster_visibility)
        score = vis * (0.5 + 0.5 * engagement(post)) * (0.5 + 0.5 * float(lab["salience"]))
        by_country.setdefault(post.country, []).append((score, post, lab))
    events: list[Any] = []
    tracked: set[str] = set()
    for iso in sorted(by_country):
        ranked = sorted(by_country[iso], key=lambda x: (-x[0], x[1].rank))[:per_country]
        for _, post, lab in ranked:
            tracked.add(post.key)
            events.append(
                SocialPostEvent(
                    platform=post.platform,
                    post_id=post.post_id,
                    author_handle=post.author_handle,
                    author_name=post.author_name,
                    author_key=post.query_key if post.query_kind == "figure" else None,
                    author_visibility=author_visibility(post, roster_visibility),
                    engagement=engagement(post),
                    headline=post.title,
                    topic=lab["topic"],
                    valence=float(lab["valence"]),
                    accuracy=0.5,
                    language=language_code(post.lang),
                    countries=[iso],
                    url=post.url,
                    mentions=post.mentions,
                    is_repost=post.is_repost,
                    duration_days=duration_days,
                    release_utc_hour=release_hour(post, news_day),
                    intensity=platform_share(cfg, iso, post.platform),
                )
            )
    digest = social_digest(posts, labels, tracked, news_day)
    n_posts = len(events)
    events.append(digest)
    scenario = Scenario(
        prompt=prompt or f"Live social ingest ({n_posts} posts)",
        events=events,
        horizon_days=max(30, duration_days),
    )
    return scenario, dropped


def _feed_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tracked posts first, then newest first."""
    newest = sorted(rows, key=lambda r: str(r["created_utc"]), reverse=True)
    return sorted(newest, key=lambda r: not r["tracked"])


def social_digest(
    posts: list[SocialPost], labels: list[dict[str, Any]], tracked: set[str], news_day: dt.date
) -> SocialDigest:
    figures: dict[str, list[dict[str, Any]]] = {}
    feed: dict[str, list[dict[str, Any]]] = {}
    topics: dict[str, dict[str, list[tuple[float, SocialPost]]]] = {}
    for post, lab in zip(posts, labels, strict=True):
        topics.setdefault(post.country, {}).setdefault(lab["topic"], []).append((float(lab["valence"]), post))
        if not is_public_author(post):
            continue  # counted in trending only
        row = {
            "platform": post.platform,
            "handle": post.author_handle,
            "name": post.author_name,
            "followers": post.author_followers,
            "kind": post.query_kind,
            "key": post.query_key,
            "text": post.title,
            "url": post.url,
            "lang": post.lang,
            "topic": lab["topic"],
            "valence": round(float(lab["valence"]), 2),
            "likes": post.likes,
            "reposts": post.reposts,
            "views": post.views,
            "created_utc": post.created_utc,
            "is_repost": post.is_repost,
            "tracked": post.key in tracked,
            "post_key": post.key,
        }
        if post.query_kind == "figure":
            figures.setdefault(post.query_key, []).append(row)
        feed.setdefault(post.country, []).append(row)
    trending: dict[str, list[dict[str, Any]]] = {}
    for iso, by_topic in topics.items():
        rows: list[dict[str, Any]] = []
        for topic, items in by_topic.items():
            top = sorted(items, key=lambda x: -engagement(x[1]))[:3]
            rows.append(
                {
                    "topic": topic,
                    "n": len(items),
                    "mean_valence": round(sum(v for v, _ in items) / len(items), 2),
                    "top_handles": [p.author_handle for _, p in top if is_public_author(p)],
                }
            )
        trending[iso] = sorted(rows, key=lambda r: (-r["n"], r["topic"]))[:6]
    return SocialDigest(
        fetch_day=news_day.isoformat(),
        seen=sorted(p.key for p in posts),
        figures={k: sorted(v, key=lambda r: str(r["created_utc"]))[-FIGURE_POSTS:] for k, v in figures.items()},
        trending=trending,
        feed={iso: _feed_order(rows)[:FEED_ROWS] for iso, rows in feed.items()},
    )
