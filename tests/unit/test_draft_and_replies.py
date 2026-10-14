"""Roster drafts cite only fetched sources; in-character replies are told how the message was taken."""

import json

import numpy as np
import yaml

from jahan.actors import engine
from jahan.actors.actions import Claim, Message, Post
from jahan.actors.briefing import briefing, pending_replies, reply
from jahan.actors.draft import draft_entry, gather_sources, write_draft
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.llm.budget import Budget
from jahan.llm.provider import LLMClient, ScriptedProvider
from jahan.llm.recordings import Recordings
from jahan.model.notables import NotableEntry
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog

WIKI = {
    "title": "Jane Example",
    "description": "Chief executive",
    "extract": "Jane Example is the chief executive of Example Corp.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Jane_Example"}},
}
NEWS = (
    '<?xml version="1.0"?><rss><channel><item><title>Example Corp cuts prices</title>'
    "<link>https://news.example.org/a</link><pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate></item>"
    "</channel></rss>"
)


def fetcher(url: str) -> bytes:
    return json.dumps(WIKI).encode() if "wikipedia" in url else NEWS.encode()


def test_draft_keeps_only_sourced_stances_and_records_sources(tmp_path):
    rec = Recordings(tmp_path / "rec")
    sources, problems = gather_sources("Jane Example", rec, "2026-10-13", fetcher=fetcher)
    assert not problems and {s["url"] for s in sources} == {
        "https://en.wikipedia.org/wiki/Jane_Example",
        "https://news.example.org/a",
    }
    again, _ = gather_sources("Jane Example", Recordings(tmp_path / "rec", replay_only=True), "2026-10-13")
    assert again == sources  # recorded: the draft can be reproduced without the network
    answer = {
        "holder_name": "Jane Example",
        "sex": "F",
        "visibility": 6.0,
        "values": {"money": 3, "freedom": 1},
        "stances": [
            {
                "topic": "economy",
                "position": 0.5,
                "summary": "Favours lower prices.",
                "source": "https://news.example.org/a",
            },
            {"topic": "politics", "position": -1, "summary": "Invented.", "source": "https://made-up.example/x"},
        ],
    }
    client = LLMClient(ScriptedProvider(lambda s, m: json.dumps(answer)), rec, Budget())
    entry, problems = draft_entry(
        client, sources, "USA.jane_example", "business_leader", "CEO, Example Corp", "Jane Example"
    )
    assert [s["source"] for s in entry["stances"]] == ["https://news.example.org/a"]
    assert any("dropped a stance" in p for p in problems) and any("approval is a placeholder" in p for p in problems)
    assert abs(sum(entry["values"].values()) - 1) < 1e-6
    NotableEntry.model_validate(entry)
    path = tmp_path / "draft.yaml"
    write_draft(path, entry, problems)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# DRAFT") and "PROBLEM: dropped a stance" in text
    assert yaml.safe_load(text)["entries"][0]["key"] == "USA.jane_example"


def test_reply_is_told_how_the_message_landed_and_only_messages_get_replies(tmp_path):
    w = generate_world(RunConfig(n_agents=1500, countries=["USA"], seed=3, start_date="2026-10-13"))
    pop = w.pop
    a = next(
        int(x)
        for x in np.flatnonzero(pop["alive"] & ~pop["is_notable"] & (w.ages() > 25))
        if engine.circle_edges(w, int(x)).size >= 3
    )
    t = int(w.edges["src"][engine.circle_edges(w, a)[0]])
    claim = Claim(headline="The clinic downtown is excellent", topic="health", stance=1.0, text="Go, it's great")
    log = EventLog()
    apply_scenario(
        w,
        Scenario(
            events=[
                Message(actor=a, target=t, claim=claim, utc_hour=0, session=1),
                Post(actor=a, claim=claim, utc_hour=0, session=1),
            ]
        ),
        log,
        1,
    )
    run_in_memory(w, 1)
    msgs = pending_replies(w, a)
    assert len(msgs) == 1 and msgs[0]["target"] == t  # the post gets no per-recipient replies
    text = briefing(w, a)
    assert "Your messages" in text
    seen = []
    client = LLMClient(
        ScriptedProvider(lambda s, m: seen.append(m[0]["content"]) or "Thanks, I'll go."),
        Recordings(tmp_path / "r"),
        Budget(),
    )
    out = reply(client, w, msgs[0])
    assert out == "Thanks, I'll go."
    prompt = seen[0]
    m = msgs[0]
    assert "How you took it:" in prompt and f"{m['trust']:.2f}" in prompt
    if m["accepted"]:
        assert f"{m['belief_before']:.2f} to {m['belief_after']:.2f}" in prompt
    else:
        assert "not convinced" in prompt
    assert "Go, it's great" in prompt and "Mind snapshot" in prompt
