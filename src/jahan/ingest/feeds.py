"""Fetch and parse RSS feeds into headline records (recorded so replays never refetch)."""

from __future__ import annotations

import base64
import datetime as dt
import email.utils
import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from jahan.llm.recordings import Recordings


class FeedSpec(BaseModel):
    name: str
    url: str
    lang: str
    channel_kinds: list[str]


@dataclass
class Headline:
    country: str
    feed: str
    lang: str
    title: str
    link: str
    published_utc: str
    rank: int  # position in the feed (top stories first)
    channel_kinds: list[str]

    @property
    def key(self) -> str:
        return hashlib.sha256(f"{self.country}|{self.title}".encode()).hexdigest()[:16]


def load_feeds(data_dir: Path) -> dict[str, list[FeedSpec]]:
    with (data_dir / "feeds" / "feeds.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)["feeds"]
    return {iso: [FeedSpec.model_validate(x) for x in lst] for iso, lst in raw.items()}


def _published_utc(pub: str) -> str:
    """ISO UTC time of an RSS pubDate; a date without a zone is taken as UTC (never the machine's zone)."""
    if not pub:
        return ""
    try:
        d = email.utils.parsedate_to_datetime(pub)
    except (TypeError, ValueError):
        return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    return d.astimezone(dt.UTC).isoformat()


def parse_rss(xml_bytes: bytes, country: str, feed: FeedSpec) -> list[Headline]:
    # News feeds have no reason to declare entities; refusing them rules out entity-expansion tricks.
    if re.search(rb"<!ENTITY", xml_bytes, re.IGNORECASE):
        raise ValueError("feed declares XML entities; refused")
    root = ET.fromstring(xml_bytes)
    out = []
    for rank, item in enumerate(root.iter("item")):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        when = _published_utc(item.findtext("pubDate") or "")
        out.append(
            Headline(
                country,
                feed.name,
                feed.lang,
                title,
                (item.findtext("link") or "").strip(),
                when,
                rank,
                list(feed.channel_kinds),
            )
        )
    return out


MAX_FEED_BYTES = 5_000_000  # a news RSS feed is tens of kB; anything this large is not a feed


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (jahan-simulator; research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (configured feed URLs)
        body: bytes = r.read(MAX_FEED_BYTES + 1)
    if len(body) > MAX_FEED_BYTES:
        raise ValueError(f"feed larger than {MAX_FEED_BYTES} bytes")
    return body


def fetch_feed(
    url: str, recordings: Recordings, fetch_day: str, fetcher: Callable[[str], bytes] | None = None
) -> bytes:
    """Fetch an RSS feed once per (url, real calendar day); later calls and replays use the recording."""
    return fetch_url(url, recordings, fetch_day, "rss", fetcher)


def fetch_url(
    url: str,
    recordings: Recordings,
    fetch_day: str,
    purpose: str,
    fetcher: Callable[[str], bytes] | None = None,
) -> bytes:
    """Any web fetch, recorded per (purpose, url, real calendar day)."""
    request = {"kind": "web", "purpose": purpose, "url": url, "fetch_day": fetch_day}
    cached = recordings.get(request)  # in replay mode a missing recording raises (no network)
    if cached is not None:
        if "body_b64" in cached:
            return base64.b64decode(cached["body_b64"])
        return bytes(cached["body"], "utf-8")  # recordings written before bytes were stored exactly
    body = (fetcher or http_get)(url)
    # store the exact bytes: the XML declaration may name a non-UTF-8 encoding
    recordings.put(request, {"body_b64": base64.b64encode(body).decode("ascii")}, {"url": url})
    return body


def collect(
    data_dir: Path,
    countries: list[str],
    recordings: Recordings,
    fetch_day: str | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> tuple[list[Headline], list[str]]:
    """All headlines for the given countries.  Returns (headlines, problems); a failing feed is skipped.

    ``fetcher`` replaces the network call (tests); its result is recorded like a real fetch.
    """
    day = fetch_day or dt.datetime.now(dt.UTC).date().isoformat()
    feeds = load_feeds(data_dir)
    out: list[Headline] = []
    problems: list[str] = []
    for iso in countries:
        for feed in feeds.get(iso, []):
            try:
                out.extend(parse_rss(fetch_feed(feed.url, recordings, day, fetcher), iso, feed))
            except Exception as e:  # noqa: BLE001 - one broken feed must not stop the others
                problems.append(f"{iso}/{feed.name}: {e}")
    return out, problems
