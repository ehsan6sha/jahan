"""Ingest from fixture RSS (no network): parsing, language-restricted pushes, recording, classification."""

import datetime as dt
import json

import numpy as np
import pytest

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.ingest.classify import classify, to_scenario
from jahan.ingest.feeds import collect, load_feeds, parse_rss
from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, ScriptedProvider
from jahan.llm.recordings import Recordings
from jahan.model.enums import EventKind
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from tests.conftest import cli

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>{a}</title><link>http://x/1</link><pubDate>Tue, 13 Oct 2026 08:00:00 GMT</pubDate></item>
<item><title>{b}</title><link>http://x/2</link><pubDate>Tue, 13 Oct 2026 09:00:00 GMT</pubDate></item>
<item><title>{c}</title><link>http://x/3</link></item>
</channel></rss>"""


def fixture_fetcher(url: str) -> bytes:
    if "IN:hi" in url:
        return RSS.format(a="रुपया गिरा", b="बारिश", c="क्रिकेट").encode()
    return RSS.format(a=f"Top story {url[-6:]}", b="Second story", c="Third story").encode()


def test_feeds_config_covers_all_countries():
    feeds = load_feeds(paths.data_dir())
    assert {"USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"} <= set(feeds)
    assert {f.lang for f in feeds["IRN"]} == {"fa"}


def test_parse_and_build_language_restricted_news(tmp_path):
    rec = Recordings(tmp_path / "rec")
    heads, problems = collect(paths.data_dir(), ["IND"], rec, fetcher=fixture_fetcher)
    assert not problems and len(heads) == 6  # two Indian feeds x three items
    sc = to_scenario(heads, classify(heads, None, 0), per_country=2)
    assert len(sc.events) == 2 and all(ev.countries == ["IND"] and ev.origin == "web" for ev in sc.events)
    w = generate_world(RunConfig(n_agents=3000, countries=["IND"], seed=3, start_date="2026-10-13"))
    log = EventLog()
    apply_scenario(w, sc, log, 1)
    for item in range(w.items.n):
        lang = int(w.items["lang"][item])
        pushed = np.flatnonzero(w.pushes["item_id"] == item)
        assert (w.channels["lang"][w.pushes["channel_id"][pushed]] == lang).all()
    rec = run_in_memory(w, 5)
    # media reach the Hindi headline only to Hindi speakers; others can hear it only by word of mouth
    hindi_item = next(i for i in range(w.items.n) if w.items["lang"][i] == w.lang_id("hi"))
    ev = rec.events.table().to_pydict()
    exposed = [
        (a, o)
        for k, a, o, it in zip(ev["kind"], ev["agent"], ev["other"], ev["item"], strict=True)
        if k == int(EventKind.INFO_EXPOSED) and it == hindi_item
    ]
    assert exposed
    speaks = (w.pop["lang"] == w.lang_id("hi")).any(axis=1)
    assert any(o < 0 for _, o in exposed)  # some exposures came from channels, so the checks below run
    for agent, other in exposed:
        if other < 0:  # a channel code -(channel + 2)
            assert speaks[w.agent_row(agent)]
            assert w.channels["lang"][-other - 2] == w.lang_id("hi")


def test_fetches_are_recorded_and_replayed(tmp_path):
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        return fixture_fetcher(url)

    def offline(url):
        raise AssertionError("network used during replay")

    rec = Recordings(tmp_path / "rec")
    heads, _ = collect(paths.data_dir(), ["USA"], rec, fetch_day="2026-10-13", fetcher=fetch)
    n_feeds = len(load_feeds(paths.data_dir())["USA"])
    assert len(calls) == n_feeds
    again, problems = collect(
        paths.data_dir(),
        ["USA"],
        Recordings(tmp_path / "rec", replay_only=True),
        fetch_day="2026-10-13",
        fetcher=offline,
    )
    assert not problems and [h.title for h in again] == [h.title for h in heads]
    # another real day has no recording: replay fails closed per feed (before any fetcher or network call)
    # and the failure is reported; no fetcher is passed, so a leak would hit the real network
    none, problems = collect(
        paths.data_dir(),
        ["USA"],
        Recordings(tmp_path / "rec", replay_only=True),
        fetch_day="2026-10-14",
    )
    assert not none and len(problems) == n_feeds
    assert all("refusing to call the network in replay mode" in p for p in problems)


def test_feed_bytes_and_dates_are_stable(tmp_path):
    """Non-UTF-8 feeds replay byte-identically; zone-less dates are UTC; entity declarations are refused."""
    from jahan.ingest.feeds import FeedSpec, fetch_feed

    xml = '<?xml version="1.0" encoding="windows-1256"?><rss><channel><item><title>خبر</title>'
    xml += "<pubDate>Tue, 13 Oct 2026 08:00:00 -0000</pubDate></item></channel></rss>"
    raw = xml.encode("cp1256")
    feed = FeedSpec(name="t", url="u", lang="fa", channel_kinds=["tv"])
    live = parse_rss(fetch_feed("u", Recordings(tmp_path / "r"), "2026-10-13", fetcher=lambda u: raw), "IRN", feed)
    replay = parse_rss(fetch_feed("u", Recordings(tmp_path / "r", replay_only=True), "2026-10-13"), "IRN", feed)
    assert [h.title for h in live] == [h.title for h in replay] == ["خبر"]
    assert live[0].published_utc == "2026-10-13T08:00:00+00:00"
    bomb = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss><channel><item><title>&a;</title></item>'
    with pytest.raises(ValueError, match="entities"):
        parse_rss(bomb + b"</channel></rss>", "USA", feed)


def test_classification_with_llm(tmp_path):
    rec = Recordings(tmp_path / "rec")
    heads, _ = collect(paths.data_dir(), ["USA"], rec, fetcher=fixture_fetcher)
    labels_json = json.dumps(
        [{"i": i, "topic": "economy", "valence": -0.5, "salience": 0.9 - 0.1 * i} for i in range(len(heads))]
    )
    client = LLMClient(ScriptedProvider(lambda s, m: labels_json), rec, Budget())
    labels = classify(heads, client, 0)
    assert labels[0] == {"topic": "economy", "valence": -0.5, "salience": 0.9}
    sc = to_scenario(heads, labels, per_country=1)
    assert sc.events[0].headline.startswith("Top story") and sc.events[0].valence == -0.5
    # malformed model output falls back to neutral labels instead of failing the ingest
    bad = LLMClient(ScriptedProvider(lambda s, m: "not json"), Recordings(tmp_path / "rec2"), Budget())
    assert all(lab["valence"] == 0.0 for lab in classify(heads, bad, 0))


def test_cli_ingest_offline(tmp_path):
    """The CLI injects recorded headlines as a ledger prompt; the date guard refuses anachronistic news."""
    run = tmp_path / "run"
    today = dt.datetime.now(dt.UTC).date().isoformat()
    cli("gen", "--run", str(run), "--agents", "1500", "--countries", "IND", "--start-date", today)
    rec = Recordings(run / "recordings")
    collect(paths.data_dir(), ["IND"], rec, fetch_day=today, fetcher=fixture_fetcher)
    out = cli("ingest", "--run", str(run), "--replay", "--provider", "none", "--fetch-day", today, "-y")
    assert "Prompt 1 injected" in out
    ledger = json.loads((run / "prompts.json").read_text(encoding="utf-8"))
    assert ledger[-1]["source"] == "ingest"
    # ingesting again the same day adds the next stories, never the ones already live
    cli("ingest", "--run", str(run), "--replay", "--provider", "none", "--fetch-day", today, "-y")
    first, second = (
        Scenario.load(run / led["scenario_file"])
        for led in json.loads((run / "prompts.json").read_text(encoding="utf-8"))
    )
    assert len(second.events) == 2
    assert not {e.headline for e in first.events} & {e.headline for e in second.events}
    # replaying a classification that was never recorded fails closed with a clear message, not a traceback
    with pytest.raises(AssertionError, match="classification stopped") as err:
        cli("ingest", "--run", str(run), "--replay", "--provider", "replay", "--fetch-day", today, "-y")
    assert "Traceback" not in str(err.value)
    cli("run", "--run", str(run), "--days", "2")
    with pytest.raises(AssertionError, match="anachronistic"):
        cli("ingest", "--run", str(run), "--replay", "--fetch-day", "2026-01-01", "-y")


def test_release_hour_uses_publication_day():
    from jahan.ingest.classify import release_hour
    from jahan.ingest.feeds import Headline

    def h(published: str) -> Headline:
        return Headline("USA", "f", "en", "t", "", published, 0, ["tv"])

    today = dt.date(2026, 10, 13)
    assert release_hour(h("2026-10-13T08:30:00+00:00"), today) == 8
    assert release_hour(h("2026-10-12T22:00:00+00:00"), today) == 0  # already out when the day starts
    assert release_hour(h(""), today) == 12
    assert release_hour(h("garbage"), today) == 12


def test_parse_rss_handles_missing_fields():
    from jahan.ingest.feeds import FeedSpec

    feed = FeedSpec(name="t", url="u", lang="en", channel_kinds=["online"])
    out = parse_rss(
        b"<rss><channel><item><title> A </title></item><item><title></title></item></channel></rss>", "USA", feed
    )
    assert len(out) == 1 and out[0].title == "A" and out[0].published_utc == ""
