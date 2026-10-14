"""LLM providers behind one interface, with recording, budgeting and a global call guard.

* ``ScriptedProvider``: deterministic responses from a function or table (tests, offline demos).
* ``AnthropicProvider``: Claude via the official SDK (``uv sync --extra llm``; needs ANTHROPIC_API_KEY).
* ``OpenAICompatProvider``: any OpenAI-compatible chat-completions endpoint (base_url + key).

``LLMClient.complete`` consults the recordings first, enforces the budget, calls the provider, records
the response and returns it.  ``forbid_llm()`` makes any call raise (used by core tests).
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from jahan.llm.budget import Budget
from jahan.llm.recordings import Recordings

_CALLS = {"count": 0, "forbidden": False}


class LLMForbidden(RuntimeError):
    pass


@contextlib.contextmanager
def forbid_llm() -> Iterator[None]:
    prev = _CALLS["forbidden"]
    _CALLS["forbidden"] = True
    try:
        yield
    finally:
        _CALLS["forbidden"] = prev


def call_count() -> int:
    return int(_CALLS["count"])


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class Provider(Protocol):
    name: str

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        effort: str = "low",
    ) -> Completion: ...


class ScriptedProvider:
    """Answers from ``responder(system, messages) -> str``; counts tokens as words (approximate)."""

    name = "scripted"

    def __init__(self, responder: Callable[[str, list[dict[str, str]]], str]) -> None:
        self.responder = responder

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        effort: str = "low",
    ) -> Completion:
        text = self.responder(system, messages)
        words = len(system.split()) + sum(len(m["content"].split()) for m in messages)
        return Completion(text, words, len(text.split()), model)


class LLMRefused(RuntimeError):
    """The model declined the request (``stop_reason == "refusal"``)."""


# Current Claude models think before answering by default, and thinking counts toward max_tokens. Callers
# size max_tokens for the visible answer; this much extra room keeps a short JSON reply from being cut off.
# Unused headroom costs nothing (only generated tokens are billed), and both default models allow 128K.
THINKING_HEADROOM = 8192


class AnthropicProvider:
    """Claude via the official SDK.

    Current models (Haiku 5.5, Sonnet 5.5) reject non-default sampling parameters, so ``temperature`` is not
    sent; how much the model thinks is set by ``effort`` instead.  Reproducibility comes from the
    recordings, not from sampling settings.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("install the llm extra: uv sync --extra llm") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        effort: str = "low",
    ) -> Completion:
        request: dict[str, Any] = {  # plain dicts; the SDK's TypedDicts are only available with the extra
            "model": model,
            "max_tokens": max_tokens + THINKING_HEADROOM,
            "system": system,
            "messages": messages,
            "output_config": {"effort": effort},
        }
        msg = self._client.messages.create(**request)
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMRefused(f"{model} declined the request ({getattr(details, 'category', None) or 'no category'})")
        text = "".join(b.text for b in msg.content if b.type == "text")
        if not text.strip():
            raise RuntimeError(f"{model} returned no text (stop_reason={msg.stop_reason})")
        return Completion(text, int(msg.usage.input_tokens), int(msg.usage.output_tokens), model)


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, base_url: str, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        if not self.key:
            raise RuntimeError(f"{api_key_env} is not set")

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        effort: str = "low",
    ) -> Completion:  # pragma: no cover - needs network and a key
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *messages],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (configured https endpoint)
            data = json.loads(r.read())
        usage = data.get("usage", {})
        return Completion(
            data["choices"][0]["message"]["content"],
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            model,
        )


STRONG_ROLES = ("scenario_parser", "feed_parser")  # turn free text into a validated scenario


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    base_url: str | None = None
    models: dict[str, str] = field(
        default_factory=lambda: {
            "scenario_parser": "claude-sonnet-5-5",
            "feed_parser": "claude-sonnet-5-5",
            "classifier": "claude-haiku-5-5",
            "persona": "claude-haiku-5-5",
            "transcript": "claude-haiku-5-5",
            "notable_autonomy": "claude-haiku-5-5",
            "text_extraction": "claude-haiku-5-5",
        }
    )
    # Parsing a scenario is the one multi-step task; everything else is short classification, extraction
    # or a few in-character sentences, where low effort is the recommended setting.
    efforts: dict[str, str] = field(default_factory=lambda: dict.fromkeys(STRONG_ROLES, "medium"))

    @classmethod
    def from_env(cls, provider: str = "anthropic", base_url: str | None = None) -> LLMConfig:
        """Defaults, overridden by JAHAN_MODEL_STRONG (scenario and feed parsing) and JAHAN_MODEL_FAST (the rest)."""
        cfg = cls(provider=provider, base_url=base_url)
        strong, fast = os.environ.get("JAHAN_MODEL_STRONG"), os.environ.get("JAHAN_MODEL_FAST")
        for role in cfg.models:
            override = strong if role in STRONG_ROLES else fast
            if override:
                cfg.models[role] = override
        return cfg


class LLMClient:
    def __init__(self, provider: Provider, recordings: Recordings, budget: Budget, config: LLMConfig | None = None):
        self.provider = provider
        self.recordings = recordings
        self.budget = budget
        self.config = config or LLMConfig()

    def model_for(self, role: str) -> str:
        return self.config.models.get(role, self.config.models["persona"])

    def effort_for(self, role: str) -> str:
        return self.config.efforts.get(role, "low")

    def complete(
        self,
        role: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        purpose: str = "",
        sim_day: int | None = None,
    ) -> str:
        model = self.model_for(role)
        effort = self.effort_for(role)
        # The key describes what was asked, not who answered: `--provider replay` must find answers that
        # were recorded live (the provider name goes into the manifest row instead).
        request = {
            "kind": "llm",
            "model": model,
            "role": role,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "effort": effort,
            "purpose": purpose,
            "sim_day": sim_day,
        }
        cached = self.recordings.get(request)
        if cached is not None:
            return str(cached["text"])
        if _CALLS["forbidden"]:
            raise LLMForbidden("LLM calls are forbidden in this context")
        est_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4 + 1
        self.budget.check(model, role, est_in, max_tokens)
        _CALLS["count"] += 1
        out = self.provider.complete(model, system, messages, max_tokens, temperature, effort)
        self.budget.charge(model, role, out.input_tokens, out.output_tokens)
        self.recordings.put(
            request,
            {"text": out.text},
            {
                "provider": self.provider.name,
                "model": out.model,
                "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens,
            },
        )
        return out.text


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply (tolerates code fences and prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1 :] if "\n" in t else t
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in model reply")
    start = min(starts)
    depth = 0
    in_str = False
    esc = False
    opener = t[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unterminated JSON in model reply")


def make_client(
    run_root: os.PathLike[str] | str,
    provider: str = "anthropic",
    replay_only: bool = False,
    budget: Budget | None = None,
    scripted: Callable[[str, list[dict[str, str]]], str] | None = None,
    base_url: str | None = None,
) -> LLMClient:
    from pathlib import Path

    rec = Recordings(Path(run_root) / "recordings", replay_only=replay_only)
    if scripted is not None or provider == "scripted":
        prov: Provider = ScriptedProvider(scripted or (lambda s, m: "{}"))
    elif provider == "anthropic":
        prov = AnthropicProvider()
    elif provider == "openai_compat":
        if not base_url:
            raise RuntimeError("openai_compat needs base_url")
        prov = OpenAICompatProvider(base_url)
    else:
        raise RuntimeError(f"unknown provider {provider}")
    return LLMClient(prov, rec, budget or Budget.from_env(), LLMConfig.from_env(provider, base_url))
