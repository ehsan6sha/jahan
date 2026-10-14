"""Social ingest from fixture responses (no network): parsing, budget, replay, ranking and anonymity."""

import datetime as dt
import json
import re
import urllib.parse

from jahan import paths
from jahan.gen.registry import validate_data
from jahan.ingest.classify import classify
from jahan.ingest.social import (
    SocialPost,
    collect_social,
    parse_count,
    platform_share,
    to_social_scenario,
)
from jahan.llm.recordings import Recordings
from jahan.model.social import load_social, validate_social
from jahan.scenario.spec import SocialDigest, SocialPostEvent

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]
DAY = "2026-10-14"
LANG = {"LulaOficial": "pt", "FlavioBolsonaro": "pt", "_FriedrichMerz": "de", "drpezeshkian": "fa", "prabowo": "id"}


def x_item(i, handle, text, created="2026-10-13T12:00:00.000Z", followers=1_000_000, likes=100, reposts=20, lang="en"):
    return {
        "id": str(i),
        "text": text,
        "createdAt": created,
        "language": lang,
        "url": f"https://x.com/{handle}/status/{i}",
        "author": {"handle": handle, "name": handle, "followers": followers},
        "metrics": {"likes": likes, "reposts": reposts, "replies": 1, "quotes": 0, "views": 1000},
        "repostedBy": None,
    }


def x_search(query):
    handles = re.findall(r"from:(\w+)", query)
    items = []
    if not handles:  # a topic search: an ordinary user, a repost and a large account
        items = [
            x_item(900, "smalluser", "Prices are up again #tariffs", followers=300),
            x_item(901, "smalluser2", "RT @someone: tariffs thread", followers=50),
            x_item(902, "bigaccount", "Tariff analysis #tariffs", followers=500_000),
            {**x_item(903, "replier", "@bigaccount agreed", followers=10), "inReplyToPostId": "902"},
        ]
    for k, h in enumerate(handles):
        base = 100 * (k + 1)
        items.append(x_item(base, h, f"Statement by {h} on the economy @SecScottBessent", lang=LANG.get(h, "en")))
        if h == "POTUS":
            items.append(x_item(base + 1, h, "RT @WhiteHouse: A great day", likes=0, reposts=500))
            items.append(x_item(base + 2, h, "An old post", created="2026-10-01T09:00:00.000Z"))
    return {"query": query, "posts": {"items": items, "hasMore": False}}


def bsky_post(uri, handle, text, created="2026-10-13T10:00:00Z", lang="pt"):
    return {
        "uri": uri,
        "author": {"handle": handle, "displayName": handle},
        "record": {"text": text, "createdAt": created, "langs": [lang]},
        "likeCount": 50,
        "repostCount": 5,
        "replyCount": 1,
        "quoteCount": 0,
    }


TELEGRAM = """<html><div class="tgme_channel_info_counter"><span class="counter_value">770K</span>
<span class="counter_type">subscribers</span></div>
<div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="BBCPersian/10">
<div class="tgme_widget_message_text js-message_text" dir="auto">خبر مهم امروز<br/>درباره اقتصاد</div>
<span class="tgme_widget_message_views">44.5K</span><time datetime="2026-10-13T21:00:48+00:00" class="time">21:00</time>
</div>
<div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="BBCPersian/11">
<div class="tgme_widget_message_text js-message_text" dir="auto">خبر دوم</div>
<time datetime="2026-10-13T22:00:00+00:00" class="time">22:00</time></div></html>"""


def fetcher_factory(seen: list[str]):
    def fetch(url: str) -> bytes:
        seen.append(url)
        if "unifapi.com" in url:
            return json.dumps(x_search(urllib.parse.unquote(url.split("q=", 1)[1]))).encode()
        if "getAuthorFeed" in url:
            actor = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["actor"][0]
            feed = [
                {"post": bsky_post(f"at://{actor}/app.bsky.feed.post/a1", actor, f"Bluesky post by {actor}")},
                {
                    "post": bsky_post("at://other/app.bsky.feed.post/b2", "other.bsky.social", "Reposted"),
                    "reason": {"$type": "app.bsky.feed.defs#reasonRepost"},
                },
            ]
            return json.dumps({"feed": feed}).encode()
        if "searchPosts" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            lang = q["lang"][0]
            return json.dumps(
                {"posts": [bsky_post(f"at://s/{lang}/1", "someone.bsky.social", f"About {q['q'][0]}", lang=lang)]}
            ).encode()
        if "accounts/lookup" in url:
            return json.dumps({"id": "42", "acct": "Bundesregierung", "followers_count": 39743}).encode()
        if "/statuses" in url or "timelines/tag" in url:
            status = {
                "id": "7",
                "uri": f"https://social.example/{len(seen)}",
                "created_at": "2026-10-13T08:00:00.000Z",
                "language": "de",
                "url": "https://social.bund.de/@Bundesregierung/7",
                "content": "<p>Die Bundesregierung &amp; die Wirtschaft</p>",
                "reblogs_count": 3,
                "favourites_count": 10,
                "replies_count": 0,
                "reblog": None,
                "account": {"acct": "Bundesregierung", "display_name": "Bundesregierung", "followers_count": 39743},
            }
            return json.dumps([status]).encode()
        if url.endswith("/BBCPersian"):
            return TELEGRAM.encode()
        if url.endswith("/khamenei_ir"):
            return b"<html><div class='tgme_page_title'>Contact</div></html>"
        raise AssertionError(f"unexpected URL {url}")

    return fetch


def collect(tmp_path, countries=M1, disabled=None, rec=None, day=DAY):
    seen: list[str] = []
    rec = rec or Recordings(tmp_path / "rec")
    posts, problems, stats = collect_social(paths.data_dir(), countries, rec, day, fetcher_factory(seen), disabled)
    return posts, problems, stats, seen


def test_config_is_valid_and_cites_the_registry():
    assert validate_social(paths.data_dir(), M1) == []
    assert not [p for p in validate_data(paths.data_dir(), M1) if "social" in p]
    cfg = load_social(paths.data_dir())
    assert cfg.sources["x"].kind == "unofficial" and "may stop working" in cfg.sources["x"].note
    assert abs(platform_share(cfg, "USA", "x") - 99.0 / 324) < 1e-9
    assert platform_share(cfg, "IND", "x") == 0.05  # 2% of internet users, floored
    assert platform_share(cfg, "IRN", "telegram") == cfg.platform_usage.default_share["telegram"]


def test_every_source_is_parsed_in_a_fixed_order(tmp_path):
    posts, problems, stats, seen = collect(tmp_path)
    x_urls = [urllib.parse.unquote(u) for u in seen if "unifapi.com" in u]
    assert x_urls and all("since:2026-10-12" in u for u in x_urls)  # from the fetch day, not the clock
    assert all("include:nativeretweets" in u for u in x_urls if "from:" in u)
    assert {"x", "bluesky", "mastodon", "telegram"} <= {p.platform for p in posts}
    by_key = {p.key: p for p in posts}
    potus = [p for p in posts if p.author_handle == "POTUS"]
    assert {p.query_key for p in potus} == {"USA.head_of_state"}
    assert any(p.is_repost for p in potus)  # a figure's reposts are kept
    assert not any("old post" in p.text for p in posts)  # older than the look-back
    assert by_key["x:100"].mentions == ["@SecScottBessent"]
    wh = [p for p in posts if p.author_handle == "WhiteHouse"]
    assert wh and wh[0].query_kind == "channel"
    topic = [p for p in posts if p.query_kind == "topic" and p.platform == "x"]
    assert topic and not any(p.is_repost or p.is_reply for p in topic)  # topic searches drop reposts and replies
    bbc = [p for p in posts if p.platform == "telegram"]
    assert len(bbc) == 2 and bbc[0].author_followers == 770_000 and bbc[0].views == 44_500
    assert "\n" in bbc[0].text and bbc[0].url == "https://t.me/BBCPersian/10"
    masto = next(p for p in posts if p.platform == "mastodon")
    assert masto.text == "Die Bundesregierung & die Wirtschaft"
    assert any(p.startswith("IRN/telegram: khamenei_ir") for p in problems)
    assert stats["x"]["queries"] <= 10
    again, _, _, _ = collect(tmp_path / "second")
    assert [p.key for p in again] == [p.key for p in posts]
    assert [p.rank for p in posts] == list(range(len(posts)))


def test_x_budget_and_switch(tmp_path):
    posts, problems, stats, seen = collect(tmp_path)
    cfg = load_social(paths.data_dir())
    assert sum("unifapi.com" in u for u in seen) <= cfg.sources["x"].max_searches
    posts, problems, stats, seen = collect(tmp_path / "nox", disabled={"x"})
    assert not any("unifapi.com" in u for u in seen) and posts
    assert "x" not in stats


def test_replay_uses_recordings_and_fails_closed(tmp_path):
    rec = Recordings(tmp_path / "rec")
    posts, _, _, _ = collect(tmp_path, ["USA", "DEU"], rec=rec)
    replay = Recordings(tmp_path / "rec", replay_only=True)

    def offline(url):
        raise AssertionError("replay touched the network")

    again, problems, _ = collect_social(paths.data_dir(), ["USA", "DEU"], replay, DAY, offline)
    assert [p.key for p in again] == [p.key for p in posts] and not problems
    other_day, problems, _ = collect_social(paths.data_dir(), ["USA", "DEU"], replay, "2026-10-15", offline)
    assert other_day == [] and problems  # an unrecorded day is refused, query by query


def test_parse_count():
    assert parse_count("770K") == 770_000 and parse_count("1.2M") == 1_200_000 and parse_count("714") == 714
    assert parse_count("n/a") == -1


def post(handle, followers, kind, key, text="Prices rise", lang="en", country="USA", likes=10):
    return SocialPost(
        platform="x",
        post_id=f"{handle}-1",
        author_handle=handle,
        author_name=handle,
        author_followers=followers,
        text=text,
        created_utc="2026-10-13T12:00:00+00:00",
        url=f"https://x.com/{handle}/status/1",
        lang=lang,
        country=country,
        query_kind=kind,
        query_key=key,
        likes=likes,
    )


def test_ranking_language_filter_and_anonymity():
    cfg = load_social(paths.data_dir())
    posts = [
        post("smalluser", 300, "topic", "en:economy", likes=5000),
        post("POTUS", 4_270_643, "figure", "USA.head_of_state"),
        post("spanish", 2_000_000, "channel", "spanish", lang="es"),
    ]
    labels = [{"topic": "economy", "valence": -0.4, "salience": 0.6}] * 3
    code = {"en": "en", "es": "es"}.get
    sc, dropped = to_social_scenario(
        posts,
        labels,
        cfg,
        per_country=1,
        roster_visibility={"USA.head_of_state": 8.7},
        language_code=lambda s: code(s) or "other",
        reachable={("USA", "en")},
        news_day=dt.date(2026, 10, 14),
    )
    events = [e for e in sc.events if isinstance(e, SocialPostEvent)]
    assert [e.author_handle for e in events] == ["POTUS"]  # reach beats an ordinary account's likes
    assert events[0].author_key == "USA.head_of_state" and events[0].author_visibility == 8.7
    assert not hasattr(events[0], "salience")  # the model's salience only ranked; the engine sets reach
    # nobody in the US reads Spanish social media in this world; an ordinary person's words are not spread
    assert dropped == {"no_channel": 1, "private": 1}
    digest = next(e for e in sc.events if isinstance(e, SocialDigest))
    assert set(digest.seen) == {p.key for p in posts}
    assert all(r["handle"] != "smalluser" and "smalluser" not in str(r) for r in digest.feed["USA"])
    assert digest.figures["USA.head_of_state"][0]["handle"] == "POTUS"
    assert digest.trending["USA"][0] == {
        "topic": "economy",
        "n": 3,
        "mean_valence": -0.4,
        "top_handles": ["POTUS", "spanish"],
    }


def test_posts_classify_without_a_model_by_their_topic_search():
    labels = classify(
        [post("a", 5, "topic", "en:economy"), post("b", 5, "figure", "USA.head_of_state")], None, 0, "posts"
    )
    assert labels[0]["topic"] == "economy" and labels[1]["topic"] == "world"
    assert all(lab["valence"] == 0.0 for lab in labels)
