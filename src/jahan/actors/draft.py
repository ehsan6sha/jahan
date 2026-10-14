"""Draft a roster entry for a real public figure from public sources, for a person to review.

Sources: the English Wikipedia summary of the person and a Google News search for their name (both
recorded, so the draft can be reproduced).  The LLM may use only those sources; every stance must cite one
of the fetched URLs, and stances citing anything else are dropped.  No private information, no personality
diagnoses.  The result is written as a DRAFT and is never loaded by the simulator until a person has
verified it and copied it into ``data/notables/<ISO3>.yaml``.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import ValidationError

from jahan.ingest.feeds import FeedSpec, fetch_url, parse_rss
from jahan.llm.provider import LLMClient, extract_json
from jahan.llm.recordings import Recordings
from jahan.model.enums import TOPIC_NAMES, VALUE_NAMES
from jahan.model.notables import NotableEntry

DRAFT_SYSTEM = f"""You draft a roster entry for a REAL public figure in a research society simulator.
Use ONLY the numbered sources given.  Every stance must cite, as "source", one of the given URLs exactly.
Include only public roles and documented public statements and decisions: no private or personal
information, no health or personality claims, no rumours.  If the sources do not support a field, omit it.
Answer with JSON only:
{{"holder_name": str, "sex": "F" or "M" (only if the sources make it clear), "birth_year": int or null,
 "visibility": 0-10 (log10 of the audience they reach: 4 local figure, 6 national figure, 7-8 top global),
 "audience_note": "<why that visibility, from the sources>",
 "approval": {{"value": 0-1, "source": "<poll named in a source>", "url": "<its URL>"}} or null,
 "platforms": [str], "languages": [ISO 639-1 codes],
 "values": {{{", ".join(sorted(VALUE_NAMES))}}} as weights summing to 1, inferred from documented positions,
 "stances": [{{"topic": one of {sorted(TOPIC_NAMES)}, "position": -1..1, "summary": "<one sentence>",
              "source": "<one of the URLs>"}}],
 "communication_style": "<documented public communication habits, or empty>"}}"""


def gather_sources(
    name: str,
    recordings: Recordings,
    fetch_day: str,
    wiki_title: str | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """[(url, title, text)] from the encyclopedia summary and a news search; plus problems."""
    sources: list[dict[str, str]] = []
    problems: list[str] = []
    title = urllib.parse.quote((wiki_title or name).replace(" ", "_"))
    wiki = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        data = json.loads(fetch_url(wiki, recordings, fetch_day, "notable_wiki", fetcher))
        page = data.get("content_urls", {}).get("desktop", {}).get("page") or wiki
        sources.append(
            {
                "url": page,
                "title": str(data.get("title", name)),
                "text": f"{data.get('description', '')}. {data.get('extract', '')}",
            }
        )
    except Exception as e:  # noqa: BLE001 - a missing page is reported, not fatal
        problems.append(f"encyclopedia summary: {e}")
    q = urllib.parse.quote(f'"{name}"')
    news = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = FeedSpec(name="news_search", url=news, lang="en", channel_kinds=["online"])
        for h in parse_rss(fetch_url(news, recordings, fetch_day, "notable_news", fetcher), "", feed)[:15]:
            if h.link:
                sources.append({"url": h.link, "title": h.title, "text": f"{h.title} ({h.published_utc[:10]})"})
    except Exception as e:  # noqa: BLE001
        problems.append(f"news search: {e}")
    return sources, problems


def draft_entry(
    client: LLMClient,
    sources: list[dict[str, str]],
    key: str,
    role: str,
    office: str,
    name: str,
    sex: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """A roster entry dict for review, and the list of problems a reviewer must look at."""
    listing = "\n".join(f"[{i + 1}] {s['url']}\n{s['title']}\n{s['text'][:1500]}" for i, s in enumerate(sources))
    user = f"Person: {name}, {office}.\n\nSources:\n{listing}"
    raw = client.complete(
        "persona", DRAFT_SYSTEM, [{"role": "user", "content": user}], max_tokens=1500, purpose="notable_draft"
    )
    problems: list[str] = []
    try:
        data = extract_json(raw)
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
        problems.append("the model's answer was not a JSON object")
    urls = {s["url"] for s in sources}
    stances = []
    for s in data.get("stances") or []:
        if isinstance(s, dict) and s.get("source") in urls and s.get("topic") in TOPIC_NAMES:
            stances.append(s)
        else:
            problems.append(f"dropped a stance without a fetched source: {str(s)[:120]}")
    approval = data.get("approval")
    if not (isinstance(approval, dict) and approval.get("url") in urls):
        approval = {"value": 0.5, "source": "PLACEHOLDER: no sourced poll in the fetched material"}
        problems.append("approval is a placeholder; add a sourced poll")
    raw_values = data.get("values")
    values: dict[str, float] = {}
    if isinstance(raw_values, dict):
        for k, v in raw_values.items():
            try:
                if k in VALUE_NAMES:
                    values[k] = max(float(v), 0.0)
            except (TypeError, ValueError):
                continue
    total = sum(values.values())
    if total <= 0:
        values = {k: round(1 / len(VALUE_NAMES), 4) for k in sorted(VALUE_NAMES)}
        problems.append("values are uniform placeholders")
    else:
        values = {k: round(v / total, 4) for k, v in values.items()}
    final_sex = sex or data.get("sex")
    if final_sex not in ("F", "M"):
        problems.append("sex not stated in the sources; set it by hand")
        final_sex = "M" if final_sex is None else final_sex
    entry = {
        "key": key,
        "role": role,
        "office": office,
        "holder_name": str(data.get("holder_name") or name),
        "sex": final_sex,
        "birth_year": data.get("birth_year"),
        "verified": f"DRAFT {dt.date.today().isoformat()}: not verified; check every field against the sources",
        "sources": sorted(urls) or ["https://example.invalid/no-sources"],
        "visibility": float(data.get("visibility") or 5.0),
        "audience_note": str(data.get("audience_note") or ""),
        "approval": approval,
        "platforms": [str(p) for p in data.get("platforms") or []],
        "languages": [str(x) for x in data.get("languages") or []],
        "values": values,
        "stances": stances,
        "communication_style": str(data.get("communication_style") or ""),
        "confidence": "low",
    }
    try:
        NotableEntry.model_validate(entry)
    except ValidationError as e:
        problems.append(f"schema: {e}")
    if not urls:
        problems.append("no sources could be fetched")
    return entry, problems


def write_draft(path: Any, entry: dict[str, Any], problems: list[str]) -> None:
    header = [
        "# DRAFT roster entry generated by `jahan notables-draft` (LLM from the listed sources).",
        "# It is NOT used by the simulator.  A person must verify every field against the sources, fix the",
        "# problems below, and copy the entry into data/notables/<ISO3>.yaml with an updated `verified` line.",
        *[f"# PROBLEM: {p}" for p in problems],
    ]
    body = yaml.safe_dump({"entries": [entry]}, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
