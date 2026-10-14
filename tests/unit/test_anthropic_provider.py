"""The Anthropic request shape the current models accept, checked against a stand-in SDK (no network)."""

import sys
import types
from types import SimpleNamespace

import pytest

from jahan.llm.budget import Budget
from jahan.llm.provider import (
    THINKING_HEADROOM,
    AnthropicProvider,
    LLMClient,
    LLMConfig,
    LLMRefused,
    ScriptedProvider,
    make_client,
)
from jahan.llm.recordings import Recordings


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def fake_sdk(monkeypatch, reply):
    messages = FakeMessages(reply)
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda api_key: SimpleNamespace(messages=messages)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return messages


def reply(blocks, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
    )


def test_request_omits_sampling_and_sets_effort(monkeypatch):
    blocks = [SimpleNamespace(type="thinking", thinking="", signature="s"), SimpleNamespace(type="text", text="hi")]
    messages = fake_sdk(monkeypatch, reply(blocks))
    out = AnthropicProvider().complete("claude-haiku-5-5", "sys", [{"role": "user", "content": "q"}], 300, 0.0, "low")
    (call,) = messages.calls
    assert "temperature" not in call and "top_p" not in call and "top_k" not in call
    assert call["output_config"] == {"effort": "low"}
    assert call["max_tokens"] == 300 + THINKING_HEADROOM  # room for thinking before the visible answer
    assert call["messages"][-1]["role"] == "user"  # no assistant prefill
    assert out.text == "hi" and (out.input_tokens, out.output_tokens) == (12, 34)


def test_refusal_and_empty_answers_raise(monkeypatch):
    fake_sdk(monkeypatch, reply([], "refusal", SimpleNamespace(category="general_harms")))
    with pytest.raises(LLMRefused, match="general_harms"):
        AnthropicProvider().complete("claude-haiku-5-5", "s", [{"role": "user", "content": "q"}], 10, 0.0)
    fake_sdk(monkeypatch, reply([SimpleNamespace(type="thinking", thinking="", signature="s")], "max_tokens"))
    with pytest.raises(RuntimeError, match="no text"):
        AnthropicProvider().complete("claude-haiku-5-5", "s", [{"role": "user", "content": "q"}], 10, 0.0)


def test_roles_get_their_effort(monkeypatch, tmp_path):
    messages = fake_sdk(monkeypatch, reply([SimpleNamespace(type="text", text="{}")]))
    c = LLMClient(AnthropicProvider(), Recordings(tmp_path / "rec"), Budget(), LLMConfig())
    c.complete("scenario_parser", "s", [{"role": "user", "content": "a"}])
    c.complete("persona", "s", [{"role": "user", "content": "b"}])
    assert [m["output_config"]["effort"] for m in messages.calls] == ["medium", "low"]
    assert [m["model"] for m in messages.calls] == ["claude-sonnet-5-5", "claude-haiku-5-5"]


def test_live_recordings_replay_under_replay_provider(tmp_path):
    """An answer recorded from one provider is found by `--provider replay` (a different provider object)."""
    live = LLMClient(ScriptedProvider(lambda s, m: "answer"), Recordings(tmp_path / "recordings"), Budget())
    live.provider.name = "anthropic"
    assert live.complete("persona", "s", [{"role": "user", "content": "q"}]) == "answer"
    replay = make_client(tmp_path, "scripted", replay_only=True)
    assert replay.complete("persona", "s", [{"role": "user", "content": "q"}]) == "answer"


def test_env_overrides_models_and_usd_cap(monkeypatch):
    monkeypatch.setenv("JAHAN_MODEL_STRONG", "claude-opus-5-5")
    monkeypatch.setenv("JAHAN_MODEL_FAST", "claude-sonnet-5-5")
    monkeypatch.setenv("JAHAN_MAX_USD", "2.5")
    cfg = LLMConfig.from_env()
    assert cfg.models["scenario_parser"] == "claude-opus-5-5" and cfg.models["persona"] == "claude-sonnet-5-5"
    b = Budget.from_env()
    assert b.max_usd == 2.5
    b.charge("claude-haiku-5-5", "persona", 1_000_000, 1_000_000)
    assert b.used_usd == pytest.approx(0.60)
    b.charge("some-other-model", "persona", 10, 10)
    assert "some-other-model has no price" in b.summary()
