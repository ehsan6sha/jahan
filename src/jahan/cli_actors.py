"""CLI commands for people acting in the world: create-agent, possess, release, say, post, decide, act,
brief, circle, play and session.

Every action becomes a ledger prompt tagged with the person's session (pre-action snapshot, so reset,
experiments and transcripts treat it like any prompt).  Only agents controlled by your session can act.
"""

from __future__ import annotations

import queue
import shlex
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, NoReturn

import numpy as np
import typer
from pydantic import ValidationError

from jahan.actors import engine, session
from jahan.actors.actions import Claim, Control, CreateAgent, Decision, Message, Post
from jahan.actors.briefing import briefing, circle_listing, pending_replies, reply
from jahan.actors.extract import claim_from_flags, extract_claim, profile_from_text
from jahan.actors.policy import PolicyError, compile_policy, default_utc_hour
from jahan.cli import (
    ProviderOpt,
    RunOpt,
    _advance,
    _autonomy_hook,
    _AutonomyHook,
    _client,
    _inject_spec,
    _load_head,
    _park_events,
    app,
)
from jahan.model.enums import EventKind
from jahan.scenario.spec import Scenario
from jahan.store.eventlog import EventLog
from jahan.store.rundir import RunDir
from jahan.store.snapshot import world_fingerprint

session_app = typer.Typer(help="Sessions: list, show and replay a person's turns.", no_args_is_help=True)
app.add_typer(session_app, name="session")

AsOpt = Annotated[
    str | None, typer.Option("--as", help="Which of your agents acts (id); needed if you control several")
]
HourOpt = Annotated[int | None, typer.Option(help="Your local hour (0-23) for the action; default 10")]
TopicOpt = Annotated[str | None, typer.Option(help="Topic of the claim (skips the LLM)")]
StanceOpt = Annotated[float | None, typer.Option(help="-1 bad news .. +1 good news (skips the LLM)")]
AboutOpt = Annotated[str | None, typer.Option(help="Product the claim is about")]


# ------------------------------------------------------------------ helpers
def _ref(world, ref: str) -> int:  # noqa: ANN001
    """An agent id, or role:<notable key> such as role:USA.head_of_state."""
    if ref.startswith("role:"):
        key = ref[5:]
        for e in world.notables:
            if e["key"] == key:
                return int(e["agent_id"])
        raise typer.BadParameter(f"no public figure {key!r}; known: {[e['key'] for e in world.notables]}")
    try:
        aid = int(ref)
    except ValueError as e:
        raise typer.BadParameter(f"{ref!r} is not an agent id or role:<key>") from e
    if not 0 <= aid < world.pop.n:
        raise typer.BadParameter(f"no agent {aid}")
    return aid


def _session_agents(rd: RunDir, world, sid: int | None) -> list[int]:  # noqa: ANN001
    """Living agents this session plays (controlled during your turns, on autopilot in between)."""
    if sid is None:
        return []
    pop = world.pop
    return [a for a in session.agents(rd, sid) if 0 <= a < pop.n and bool(pop["alive"][a])]


def _actor(rd: RunDir, world, as_ref: str | None) -> tuple[int, int]:  # noqa: ANN001
    sid = session.current(rd)
    mine = _session_agents(rd, world, sid)
    if sid is None or not mine:
        raise typer.BadParameter("you play no agent; use `jahan possess --agent ID` or `jahan create-agent --possess`")
    if as_ref is not None:
        aid = _ref(world, as_ref)
        if aid not in mine:
            raise typer.BadParameter(f"agent {aid} is not played by your session (you play {mine})")
        return sid, aid
    if len(mine) > 1:
        raise typer.BadParameter(f"you play several agents ({mine}); choose one with --as")
    return sid, mine[0]


def _take_control(world, sid: int, agents: list[int]) -> list[Control]:  # noqa: ANN001
    """Control events for agents of the session that are on autopilot right now."""
    return [Control(agent=a, session=sid, on=True) for a in agents if int(world.pop["controlled_by"][a]) != sid]


def _reject(rd: RunDir, world, actor: int, sid: int | None, reason: str) -> NoReturn:  # noqa: ANN001
    log = EventLog()
    log.add(world.day, EventKind.ACTION_REJECTED, actor, -1 if sid is None else sid, -1, 0.0, -1, reason[:300])
    _park_events(rd, log)
    raise typer.BadParameter(reason)


def _submit(rd: RunDir, world, sid: int | None, events: list[Any], prompt: str) -> int:  # noqa: ANN001
    if sid is not None:  # acting takes your agents off autopilot
        acting = {int(ev.actor) for ev in events if isinstance(ev, Message | Post | Decision) and ev.actor >= 0}
        acting -= {int(ev.agent) for ev in events if isinstance(ev, Control)}
        mine = [a for a in session.agents(rd, sid) if a in acting]
        events = [*_take_control(world, sid, mine), *events]
    spec = Scenario(prompt=prompt, events=events, horizon_days=30)
    extra = {"session": sid} if sid is not None else None
    source = f"session:{sid}" if sid is not None else "actor"
    before = world_fingerprint(world) if sid is not None else ""
    pid = _inject_spec(rd, world, spec, True, source=source, quiet=True, extra=extra)
    if sid is not None:
        session.record_turn(rd, sid, pid, world.day, prompt, before)
    return pid


def _utc(world, aid: int, hour: int | None) -> int:  # noqa: ANN001
    local = 10 if hour is None else hour
    if not 0 <= local <= 23:
        raise typer.BadParameter("hour must be 0-23")
    return default_utc_hour(world, world.agent_row(aid), local)


def _claim(client, world, text: str, topic: str | None, stance: float | None, about: str | None) -> Claim:  # noqa: ANN001
    if about is not None and about not in world.product_names:
        raise typer.BadParameter(f"no product named {about!r} (known: {world.product_names})")
    if topic is not None or stance is not None or client is None:
        try:
            return claim_from_flags(text, topic or "world", 0.0 if stance is None else stance, about)
        except ValidationError as e:
            raise typer.BadParameter(str(e.errors()[0]["msg"])) from e
    claim = extract_claim(client, world, text)
    if about:
        claim = Claim(headline=claim.headline, topic=claim.topic, stance=claim.stance, about_product=about, text=text)
    typer.echo(f"Understood as: [{claim.topic}, stance {claim.stance:+g}] {claim.headline}")
    return claim


def _maybe_client(rd: RunDir, provider: str, purpose: str):  # noqa: ANN202
    try:
        return _client(rd, provider)
    except typer.BadParameter as e:
        typer.echo(f"({purpose} without an LLM: {e.message})")
        return None


def _check(rd: RunDir, world, ev: Any, actor: int, sid: int | None) -> None:  # noqa: ANN001
    reason = engine.check(world, ev)
    if reason:
        _reject(rd, world, actor, sid, reason)


# ------------------------------------------------------------------ people
@app.command("create-agent")
def create_agent_cmd(
    run_dir: RunOpt,
    country: Annotated[str, typer.Option(help="ISO3 country of residence")],
    sex: Annotated[str | None, typer.Option(help="F or M")] = None,
    age: Annotated[int | None, typer.Option()] = None,
    first_name: Annotated[str | None, typer.Option()] = None,
    last_name: Annotated[str | None, typer.Option()] = None,
    region: Annotated[str | None, typer.Option(help="urban or rural")] = None,
    education: Annotated[int | None, typer.Option(help="ISCED level 0-8")] = None,
    income: Annotated[float | None, typer.Option(help="Monthly income, local currency")] = None,
    languages: Annotated[str | None, typer.Option(help="Comma-separated ISO 639-1 codes")] = None,
    visibility: Annotated[
        float | None, typer.Option(help="log10 of reachable audience (1 family .. 8 head of state)")
    ] = None,
    friends: Annotated[int, typer.Option()] = 5,
    acquaintances: Annotated[int, typer.Option()] = 5,
    household_of: Annotated[int | None, typer.Option(help="Join this agent's household")] = None,
    notable_role: Annotated[str | None, typer.Option(help="Make a public figure with this role (roles.yaml)")] = None,
    office: Annotated[str | None, typer.Option(help="Office title for a public figure")] = None,
    describe: Annotated[str | None, typer.Option(help="Free-text description (LLM fills supported fields)")] = None,
    provider: ProviderOpt = "anthropic",
    possess: Annotated[bool, typer.Option(help="Take control of the new person")] = False,
) -> None:
    """Add a person to the world (population weight 0); unspecified traits come from a resident's profile."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    fields: dict[str, Any] = {}
    if describe:
        client = _maybe_client(rd, provider, "description used as a note only")
        if client is not None:
            fields.update(profile_from_text(client, world, describe))
    explicit = {
        "sex": sex.upper() if sex else None,
        "age": age,
        "first_name": first_name,
        "last_name": last_name,
        "region": region,
        "education": education,
        "income_month": income,
        "languages": [x.strip() for x in languages.split(",")] if languages else None,
        "visibility": visibility,
        "household_of": household_of,
        "notable_role": notable_role,
        "office": office,
    }
    fields.update({k: v for k, v in explicit.items() if v is not None})
    fields.update(country=country.upper(), friends=friends, acquaintances=acquaintances, description=describe or "")
    try:
        ev = CreateAgent(**fields)
    except ValidationError as e:
        raise typer.BadParameter(str(e)) from e
    sid = session.current(rd, create=True) if possess else None
    _check(rd, world, ev, -1, sid)
    new_id = int(world.counters["agent"])
    events: list[Any] = [ev]
    if possess:
        assert sid is not None
        events.append(Control(agent=new_id, session=sid, on=True))
    _submit(rd, world, sid, events, f"create agent in {ev.country}")
    if sid is not None:
        session.set_agents(rd, sid, add=[new_id])
    from jahan.query.views import agent_card

    typer.echo(f"Created agent {new_id}: {engine.display_name(world, world.agent_row(new_id))}")
    typer.echo("\n".join(agent_card(world, new_id).splitlines()[:14]))
    if possess:
        typer.echo(f"Session {sid}: you control agent {new_id}.")


@app.command()
def possess(
    run_dir: RunOpt,
    agent: Annotated[str, typer.Option(help="Agent id or role:<key>, e.g. role:USA.head_of_state")],
    new: Annotated[bool, typer.Option("--new-session", help="Start a new session instead of the current one")] = False,
) -> None:
    """Take control of an agent: it stops sharing, buying, marrying, moving jobs and travelling on its own."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    aid = _ref(world, agent)
    sid = session.new_session(rd) if new else session.current(rd, create=True)
    assert sid is not None
    ev = Control(agent=aid, session=sid, on=True)
    _check(rd, world, ev, aid, sid)
    _submit(rd, world, sid, [ev], f"possess agent {aid}")
    session.set_agents(rd, sid, add=[aid])
    row = world.agent_row(aid)
    typer.echo(f"Session {sid}: you control {engine.display_name(world, row)} (agent {aid}).")
    typer.echo(f"Allowed audiences: {engine.allowed_audiences(world, row)}")
    from jahan.actors.policy import allowed_policies

    pols = allowed_policies(world, aid)
    if pols:
        typer.echo(f"Policies this role may use (`jahan act`): {pols}")
    typer.echo(briefing(world, aid))


@app.command()
def release(
    run_dir: RunOpt,
    agent: Annotated[str | None, typer.Option(help="Agent id; default every agent of your session")] = None,
) -> None:
    """Stop playing agents; they act on their own from now on."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid = session.current(rd)
    mine = _session_agents(rd, world, sid)
    targets = [_ref(world, agent)] if agent else mine
    if sid is None or not targets or any(a not in mine for a in targets):
        raise typer.BadParameter(f"your session plays {mine}")
    events = [Control(agent=a, session=sid, on=False) for a in targets if int(world.pop["controlled_by"][a]) == sid]
    if events:
        _submit(rd, world, sid, events, f"release {targets}")
    session.set_agents(rd, sid, remove=targets)
    typer.echo(f"Released {targets}; they act on their own again.")


def autopilot_for_run(rd: RunDir, world, keep_control: bool) -> None:  # noqa: ANN001
    """Before `run`: agents people play go on autopilot (the engine drives them) unless kept idle."""
    for sid_s in session.load(rd)["sessions"]:
        sid = int(sid_s)
        ctrl = [a for a in _session_agents(rd, world, sid) if int(world.pop["controlled_by"][a]) == sid]
        if not ctrl:
            continue
        if keep_control:
            typer.echo(f"Session {sid}: agents {ctrl} stay idle during this run (--keep-control).")
        else:
            _autopilot_now(rd, world, sid, ctrl, "run")
            typer.echo(
                f"Session {sid}: agents {ctrl} are on autopilot during this run; your next action takes them back."
            )


def _autopilot_now(rd: RunDir, world, sid: int, agents: list[int], why: str) -> None:  # noqa: ANN001
    """Hand the session's controlled agents to the engine (one prompt, so reset and replay see it)."""
    events = [Control(agent=a, session=sid, on=False) for a in agents if int(world.pop["controlled_by"][a]) == sid]
    if events:
        _submit(rd, world, sid, events, f"autopilot ({why})")


# ------------------------------------------------------------------ actions
@app.command()
def say(
    run_dir: RunOpt,
    text: Annotated[str, typer.Argument(help="What you tell them")],
    to: Annotated[str, typer.Option(help="Agent id of someone who listens to you (see `jahan circle`)")],
    as_: AsOpt = None,
    hour: HourOpt = None,
    topic: TopicOpt = None,
    stance: StanceOpt = None,
    about: AboutOpt = None,
    provider: ProviderOpt = "anthropic",
) -> None:
    """Tell one person in your circle something (delivered when they are awake; they may not believe you)."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid, aid = _actor(rd, world, as_)
    target = _ref(world, to)
    client = None if (topic is not None or stance is not None) else _maybe_client(rd, provider, "claim")
    ev = Message(
        actor=aid,
        target=target,
        claim=_claim(client, world, text, topic, stance, about),
        utc_hour=_utc(world, aid, hour),
        session=sid,
    )
    _check(rd, world, ev, aid, sid)
    _submit(rd, world, sid, [ev], f"say to {target}: {text}")
    typer.echo(
        f"Queued for {world.date()}: {engine.display_name(world, world.agent_row(target))} hears it when awake. "
        "Advance with `jahan run --days 1` or `jahan play`; `jahan brief` shows how they took it."
    )


@app.command()
def post(
    run_dir: RunOpt,
    text: Annotated[str, typer.Argument()],
    audience: Annotated[
        str, typer.Option(help="circle, community, national or global (capped by who you are)")
    ] = "circle",
    as_: AsOpt = None,
    hour: HourOpt = None,
    topic: TopicOpt = None,
    stance: StanceOpt = None,
    about: AboutOpt = None,
    provider: ProviderOpt = "anthropic",
) -> None:
    """Say something publicly.  Reach follows your visibility and role, not your wording."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid, aid = _actor(rd, world, as_)
    if audience not in ("circle", "community", "national", "global"):
        raise typer.BadParameter("audience must be circle, community, national or global")
    client = None if (topic is not None or stance is not None) else _maybe_client(rd, provider, "claim")
    ev = Post(
        actor=aid,
        audience=audience,
        claim=_claim(client, world, text, topic, stance, about),
        utc_hour=_utc(world, aid, hour),
        session=sid,
    )
    _check(rd, world, ev, aid, sid)
    eff = engine.effective_audience(world, world.agent_row(aid), audience)
    _submit(rd, world, sid, [ev], f"post ({audience}): {text}")
    note = "" if eff == audience else f" (you asked for {audience}; your reach allows {eff})"
    typer.echo(f"Post queued for {world.date()}, audience {eff}{note}.")


@app.command()
def decide(
    run_dir: RunOpt,
    product: Annotated[str, typer.Argument()],
    choice: Annotated[str, typer.Argument(help="adopt, reject or delay")],
    as_: AsOpt = None,
) -> None:
    """Decide about a product for your agent (adopting counts as buying it)."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid, aid = _actor(rd, world, as_)
    if choice not in ("adopt", "reject", "delay"):
        raise typer.BadParameter("choice must be adopt, reject or delay")
    ev = Decision(actor=aid, product=product, choice=choice, session=sid)
    _check(rd, world, ev, aid, sid)
    _submit(rd, world, sid, [ev], f"{choice} {product}")
    typer.echo(f"Decision recorded: {choice} {product}.")


def _parse_params(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise typer.BadParameter(f"policy parameters look like key=value (got {it!r})")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@app.command()
def act(
    run_dir: RunOpt,
    kind: Annotated[str, typer.Argument(help="Policy from roles.yaml, e.g. tariff, set_interest_rate")],
    params: Annotated[
        list[str] | None, typer.Argument(help="key=value parameters, e.g. rate_pct=25 target=CHN")
    ] = None,
    text: Annotated[str | None, typer.Option(help="Text of a statement or directive")] = None,
    as_: AsOpt = None,
    hour: HourOpt = None,
    topic: TopicOpt = None,
    stance: StanceOpt = None,
    provider: ProviderOpt = "anthropic",
) -> None:
    """Use a power of your agent's office (checked against the role; effect sizes come from roles.yaml)."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid, aid = _actor(rd, world, as_)
    p = _parse_params(params or [])
    claim = None
    if text:
        client = None if (topic is not None or stance is not None) else _maybe_client(rd, provider, "claim")
        claim = _claim(client, world, text, topic, stance, None)
    try:
        events = compile_policy(world, aid, kind, p, claim=claim, session=sid, utc_hour=_utc(world, aid, hour))
    except (PolicyError, ValidationError) as e:
        _reject(rd, world, aid, sid, f"{kind}: {e}")
    for ev in events:
        if hasattr(ev, "session"):
            _check(rd, world, ev, aid, sid)
    _submit(rd, world, sid, events, f"act {kind} {p}" + (f": {text}" if text else ""))
    typer.echo(f"{kind} queued for {world.date()}: {len(events)} engine events.")


@app.command()
def brief(
    run_dir: RunOpt,
    as_: AsOpt = None,
    replies: Annotated[bool, typer.Option(help="In-character replies to yesterday's messages (LLM)")] = True,
    provider: ProviderOpt = "anthropic",
) -> None:
    """What your agent learned yesterday and how people took your messages."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    _, aid = _actor(rd, world, as_)
    typer.echo(briefing(world, aid))
    if replies:
        _print_replies(rd, world, aid, _maybe_client(rd, provider, "replies") if pending_replies(world, aid) else None)


def _print_replies(rd: RunDir, world, aid: int, client) -> None:  # noqa: ANN001
    msgs = pending_replies(world, aid)
    if not msgs or client is None:
        return
    for m in msgs:
        who = engine.display_name(world, world.agent_row(int(m["target"])))
        try:
            typer.echo(f"  {who} replies: {reply(client, world, m)}")
        except RuntimeError as e:  # budget, missing recording, provider error: the world is unaffected
            typer.echo(f"  ({who}'s reply unavailable: {e})")


@app.command()
def circle(run_dir: RunOpt, as_: AsOpt = None) -> None:
    """Who listens to your agent, and how much they trust it."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    _, aid = _actor(rd, world, as_)
    typer.echo(circle_listing(world, aid))


# ------------------------------------------------------------------ the turn-based loop
class _LineReader:
    """stdin lines with an optional timeout (a background thread, so it works on Windows too)."""

    EOF = object()

    def __init__(self) -> None:
        self.q: queue.Queue[Any] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in sys.stdin:
            self.q.put(line.rstrip("\r\n").lstrip("﻿"))  # Windows PowerShell pipes add a byte-order mark
        self.q.put(self.EOF)

    def get(self, prompt: str, timeout: float | None) -> Any:
        typer.echo(prompt, nl=False)
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            typer.echo("")
            return None
        if item is not self.EOF and not sys.stdin.isatty():
            typer.echo(item)  # echo scripted input so transcripts read naturally
        return item


HELP = (
    "Commands: say <id> <text> | post <circle|community|national|global> <text> | decide <product> "
    "<adopt|reject|delay> | act <policy> key=value ... [text=...] | as <id> | circle | who <id> | brief | help | "
    "next | auto <days> | quit.  Options for say/post: topic= stance= about= hour="
)
CLAIM_OPTS = {"topic", "stance", "about", "hour"}


def _split(tokens: list[str], keys: set[str]) -> tuple[list[str], dict[str, str]]:
    words, opts = [], {}
    for t in tokens:
        k, sep, v = t.partition("=")
        if sep and k in keys:
            opts[k] = v
        else:
            words.append(t)
    return words, opts


def _turn(
    rd: RunDir,
    world,
    sid: int,
    agents: list[int],
    client,
    reader: _LineReader,  # noqa: ANN001
    timeout: float | None,
) -> tuple[str, int, list[Any]]:
    """Read commands for one day.  Returns (outcome, auto_days, events); outcome is next|auto|quit|timeout."""
    actor = agents[0]
    events: list[Any] = []
    counts = {"message": 0, "post": 0}
    while True:
        line = reader.get(f"[{engine.display_name(world, world.agent_row(actor))}] > ", timeout)
        if line is None:
            return ("next" if events else "timeout"), 0, events
        if line is _LineReader.EOF:
            return "quit", 0, events
        try:
            tokens = shlex.split(line)
        except ValueError as e:
            typer.echo(f"  ! {e}")
            continue
        if not tokens:
            continue
        cmd, rest = tokens[0].lower(), tokens[1:]
        try:
            if cmd == "next":
                return "next", 0, events
            if cmd == "auto":
                return "auto", max(1, int(rest[0]) if rest else 1), events
            if cmd == "quit":
                return "quit", 0, events
            if cmd == "help":
                typer.echo(HELP)
            elif cmd == "as":
                aid = int(rest[0])
                if aid not in agents:
                    raise ValueError(f"you control {agents}")
                actor = aid
            elif cmd == "circle":
                typer.echo(circle_listing(world, actor))
            elif cmd == "brief":
                typer.echo(briefing(world, actor))
            elif cmd == "who":
                from jahan.query.views import agent_card

                typer.echo(agent_card(world, _ref(world, rest[0])))
            elif cmd in ("say", "post"):
                words, opts = _split(rest, CLAIM_OPTS)
                if len(words) < 2:
                    raise ValueError(f"usage: {cmd} <{'id' if cmd == 'say' else 'audience'}> <text>")
                text = " ".join(words[1:])
                stance = float(opts["stance"]) if "stance" in opts else None
                claim = _claim(client, world, text, opts.get("topic"), stance, opts.get("about"))
                hour = _utc(world, actor, int(opts["hour"]) if "hour" in opts else None)
                ev: Any
                if cmd == "say":
                    ev = Message(actor=actor, target=_ref(world, words[0]), claim=claim, utc_hour=hour, session=sid)
                else:
                    ev = Post(actor=actor, audience=words[0], claim=claim, utc_hour=hour, session=sid)
                kind = "message" if cmd == "say" else "post"
                limit = engine.MAX_MESSAGES_PER_DAY if cmd == "say" else engine.MAX_POSTS_PER_DAY
                if counts[kind] >= limit:
                    raise ValueError(f"daily limit of {limit} {kind}s reached")
                reason = engine.check(world, ev)
                if reason:
                    raise ValueError(reason)
                counts[kind] += 1
                events.append(ev)
                extra = ""
                if cmd == "post":
                    extra = f" (audience {engine.effective_audience(world, world.agent_row(actor), words[0])})"
                typer.echo(f"  queued{extra}")
            elif cmd == "decide":
                ev = Decision(actor=actor, product=rest[0], choice=rest[1], session=sid)
                reason = engine.check(world, ev)
                if reason:
                    raise ValueError(reason)
                events.append(ev)
                typer.echo("  queued")
            elif cmd == "act":
                words, opts = _split(rest[1:], {"text", "topic", "stance", "hour"})
                params = _parse_params(words)
                claim = None
                if "text" in opts:
                    st = float(opts["stance"]) if "stance" in opts else None
                    claim = _claim(client, world, opts["text"], opts.get("topic"), st, None)
                hour = _utc(world, actor, int(opts["hour"]) if "hour" in opts else None)
                evs = compile_policy(world, actor, rest[0], params, claim=claim, session=sid, utc_hour=hour)
                events.extend(evs)
                typer.echo(f"  queued {rest[0]} ({len(evs)} engine events)")
            else:
                typer.echo(f"  unknown command {cmd!r}. {HELP}")
        except (ValueError, IndexError, PolicyError, ValidationError, typer.BadParameter) as e:
            msg = e.message if isinstance(e, typer.BadParameter) else str(e)
            log = EventLog()
            log.add(world.day, EventKind.ACTION_REJECTED, actor, sid, -1, 0.0, -1, f"{cmd}: {msg}"[:300])
            _park_events(rd, log)
            typer.echo(f"  ! {msg}")


def _autopilot(rd: RunDir, world, sid: int, agents: list[int]) -> None:  # noqa: ANN001
    """A day the person lets pass: the engine drives their agents (released) and the day is marked."""
    _autopilot_now(rd, world, sid, agents, "no input")
    log = EventLog()
    for a in agents:
        log.add(world.day, EventKind.AUTOPILOT_TURN, a, sid, -1, 0.0, int(world.pop["location_country"][a]))
    _park_events(rd, log)
    session.record_autopilot(rd, sid, world.day, agents, world_fingerprint(world))


@app.command()
def play(
    run_dir: RunOpt,
    days: Annotated[int, typer.Option(help="Days to play")] = 7,
    timeout: Annotated[float | None, typer.Option(help="Seconds to wait for input; then autopilot for the day")] = None,
    provider: ProviderOpt = "anthropic",
    replies: Annotated[bool, typer.Option(help="In-character replies to your messages (LLM)")] = True,
    autonomy: Annotated[bool, typer.Option(help="Public figures decide on salient days (LLM, capped)")] = True,
) -> None:
    """Turn-based play: each day you get a briefing, give your agents' actions, and the world moves one day."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    sid = session.current(rd)
    if sid is None or not _session_agents(rd, world, sid):
        raise typer.BadParameter("you play no agent; use `jahan possess --agent ID` first")
    client = _maybe_client(rd, provider, "claims need topic= and stance=; no replies")
    hook: _AutonomyHook | None = None
    if autonomy and client is not None and world.notables:
        client.budget.role_share.setdefault("notable_autonomy", 0.3)
        hook = _AutonomyHook(client)
    elif autonomy:
        hook = _autonomy_hook(rd, world, provider)
    session.note_autonomy(rd, sid, hook is not None)
    reader = _LineReader()
    typer.echo(HELP)
    auto_left = 0
    for _ in range(days):
        agents = _session_agents(rd, world, sid)
        if not agents:
            typer.echo("Your session plays no living agent; stopping.")
            break
        typer.echo(f"\n--- Day {world.day} ({world.date().isoformat()}) ---")
        if auto_left > 0:
            auto_left -= 1
            _autopilot(rd, world, sid, agents)
        else:
            for a in agents:
                typer.echo(briefing(world, a))
                if replies:
                    _print_replies(rd, world, a, client)
            outcome, auto_days, events = _turn(rd, world, sid, agents, client, reader, timeout)
            if events or outcome == "next":  # you took the turn: your agents are under your control today
                turn = [*_take_control(world, sid, agents), *events]
                if turn:
                    _submit(rd, world, sid, turn, f"turn on day {world.day}")
            if outcome in ("timeout", "auto") and not events:
                _autopilot(rd, world, sid, agents)
            if outcome == "auto":
                auto_left = auto_days - 1
            if outcome == "quit":
                typer.echo("Stopped; your queued actions apply when the world next runs.")
                break
        _advance(rd, world, 1, before_day=hook)
    typer.echo(f"World is at day {world.day} ({world.date().isoformat()}).")
    if client is not None:
        typer.echo(client.budget.summary())


# ------------------------------------------------------------------ public-figure roster drafts
@app.command("notables-draft")
def notables_draft(
    country: Annotated[str, typer.Option(help="ISO3 country of the office")],
    name: Annotated[str, typer.Option(help="The person's public name")],
    role: Annotated[str, typer.Option(help="Role from data/roles.yaml, e.g. business_leader")],
    office: Annotated[str, typer.Option(help="Office or title, e.g. 'Chief executive, Example Corp'")],
    key: Annotated[str | None, typer.Option(help="Roster key; default <ISO3>.<name>")] = None,
    sex: Annotated[str | None, typer.Option(help="F or M, if the sources do not say")] = None,
    wiki_title: Annotated[str | None, typer.Option(help="Encyclopedia page title if it differs from the name")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output YAML; default data/notables/drafts/<ISO3>_<name>.yaml")
    ] = None,
    provider: ProviderOpt = "anthropic",
    replay: Annotated[bool, typer.Option(help="Use recorded sources and answers only")] = False,
    fetch_day: Annotated[str | None, typer.Option(help="Real UTC date of the recorded sources")] = None,
) -> None:
    """Draft a sourced profile of a real public figure for a person to review (never loaded automatically)."""
    import datetime as dt
    import re

    from jahan import paths
    from jahan.actors.draft import draft_entry, gather_sources, write_draft
    from jahan.model.notables import load_roles

    iso = country.upper()
    roles = load_roles(paths.data_dir())
    if role not in roles:
        raise typer.BadParameter(f"unknown role {role}; known: {sorted(roles)}")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    rd = RunDir(paths.data_dir() / "raw" / "notable_drafts")  # recordings live under data/raw (not committed)
    client = _client(rd, "replay" if replay else provider)
    if replay:
        client.recordings.replay_only = True
    day = fetch_day or dt.datetime.now(dt.UTC).date().isoformat()
    sources, problems = gather_sources(name, client.recordings, day, wiki_title)
    entry, more = draft_entry(client, sources, key or f"{iso}.{slug}", role, office, name, sex.upper() if sex else None)
    path = out or paths.data_dir() / "notables" / "drafts" / f"{iso}_{slug}.yaml"
    write_draft(path, entry, problems + more)
    typer.echo(f"Draft written to {path} ({len(sources)} sources, {len(entry['stances'])} sourced stances).")
    for p in problems + more:
        typer.echo(f"  review: {p}")
    typer.echo(client.budget.summary())


# ------------------------------------------------------------------ sessions
@session_app.command("list")
def session_list(run_dir: RunOpt) -> None:
    rd = RunDir(run_dir)
    world = _load_head(rd)
    data = session.load(rd)
    if not data["sessions"]:
        typer.echo("No sessions yet.")
        return
    for sid, s in sorted(data["sessions"].items(), key=lambda kv: int(kv[0])):
        live = [t for t in s["turns"] if not t["undone"]]
        mark = " (current)" if data.get("current") == int(sid) else ""
        typer.echo(
            f"session {sid}{mark}: {len(live)} turns in the timeline, {len(s['turns']) - len(live)} undone; "
            f"plays {_session_agents(rd, world, int(sid))} "
            f"(under control now: {[int(x) for x in np.flatnonzero(world.pop['controlled_by'] == int(sid))]})"
        )


@session_app.command("show")
def session_show(run_dir: RunOpt, session_id: Annotated[int, typer.Option("--session")]) -> None:
    rd = RunDir(run_dir)
    for t in session.turns(rd, session_id):
        what = (
            f"autopilot {t['autopilot']}" if t.get("autopilot") else f"prompt {t['prompt_id']}: {t.get('summary', '')}"
        )
        state = " [undone]" if t["undone"] else ""
        state += " [replayed]" if t.get("replayed") else ""
        typer.echo(f"turn {t['turn']}, day {t['day']}: {what}{state}")


@session_app.command("replay")
def session_replay(
    run_dir: RunOpt,
    session_id: Annotated[int, typer.Option("--session")],
    to_day: Annotated[int | None, typer.Option(help="Keep running until this world day after the last turn")] = None,
    provider: Annotated[str, typer.Option(help="LLM for public figures; replay = recorded answers only")] = "replay",
    autonomy: Annotated[
        bool | None,
        typer.Option(help="Public figures' turns between yours; default: as when the session was played"),
    ] = None,
    force: Annotated[bool, typer.Option(help="Replay even if the world differs from when a turn was recorded")] = False,
) -> None:
    """Re-apply a session's undone turns (after a reset) at their original days.

    Each turn is checked first: its recorded actions file must be unedited and the world must be in exactly
    the state the turn was recorded on; otherwise the replay stops (unless --force).
    """
    rd = RunDir(run_dir)
    world = _load_head(rd)
    todo = [t for t in session.turns(rd, session_id) if t["undone"] and not t.get("replayed")]
    if not todo:
        typer.echo("Nothing to replay: no undone turns.")
        return
    use_autonomy = session.autonomy_used(rd, session_id) if autonomy is None else autonomy
    hook = _autonomy_hook(rd, world, provider) if use_autonomy else None
    typer.echo(f"Public figures' own turns: {'replayed from recordings' if hook else 'off (as when played)'}.")
    data = session.load(rd)
    data["current"] = session_id
    session.save(rd, data)
    for t in sorted(todo, key=lambda x: (x["day"], x["turn"])):
        if t["day"] < world.day:
            raise typer.BadParameter(
                f"turn {t['turn']} is for day {t['day']} but the world is at day {world.day}; reset further back"
            )
        if t["day"] > world.day:
            _advance(rd, world, t["day"] - world.day, before_day=hook)
        problem = session.verify_turn(rd, t, world_fingerprint(world))
        if problem and not force:
            raise typer.BadParameter(f"{problem}; nothing more was replayed (use --force to replay anyway)")
        if problem:
            typer.echo(f"warning: {problem}")
        if t.get("prompt_id") is None:
            _autopilot(rd, world, session_id, [int(a) for a in t["autopilot"]])
        else:
            spec = Scenario.load(rd.root / t["file"])
            pid = _inject_spec(
                rd, world, spec, True, source=f"session:{session_id}", quiet=True, extra={"session": session_id}
            )
            session.record_turn(rd, session_id, pid, world.day, t.get("summary", ""), t.get("world_before", ""))
        _mark_replayed(rd, session_id, int(t["turn"]))
        typer.echo(f"Replayed turn {t['turn']} on day {world.day}.")
    if to_day is not None and to_day > world.day:
        _advance(rd, world, to_day - world.day, before_day=hook)
    typer.echo(f"World is at day {world.day}.")


def _mark_replayed(rd: RunDir, sid: int, turn: int) -> None:
    data = session.load(rd)
    for t in data["sessions"][str(sid)]["turns"]:
        if t["turn"] == turn:
            t["replayed"] = True
    session.save(rd, data)
