"""Command-line interface.

World: gen, run, reset, validate-data, bench, doctor.  Scenarios: inject (YAML or free text), feed,
ingest (real headlines), experiment, report.  People: agents, agent, ask, transcript.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pyarrow.parquet as pq
import typer
from pydantic import ValidationError

from jahan import paths
from jahan.config import RunConfig
from jahan.gen.registry import validate_data
from jahan.gen.world_gen import generate_world
from jahan.model.enums import EventKind
from jahan.model.notables import is_later_roster
from jahan.scenario.experiment import run_experiment
from jahan.scenario.inject import apply_scenario, describe_moves, preview_moves
from jahan.scenario.report import build_report
from jahan.scenario.spec import Scenario
from jahan.sim.runner import Recorder, run
from jahan.store.eventlog import EVENT_SCHEMA, EventLog, content_hash, read_all, truncate_from
from jahan.store.rundir import RunDir
from jahan.store.snapshot import load_snapshot, save_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Jahan: a real-world society simulator.")

RunOpt = Annotated[Path, typer.Option("--run", "-r", help="Run directory")]


def _load_head(rd: RunDir):  # noqa: ANN202
    from jahan.store.snapshot import SnapshotError

    if not rd.exists():
        raise typer.BadParameter(f"{rd.root} has no world; run `jahan gen` first")
    try:
        return load_snapshot(rd.head)
    except SnapshotError as e:
        raise typer.BadParameter(
            f"{rd.root} cannot be opened by this version of Jahan ({e}). Create a new world with `jahan gen`."
        ) from e


@app.command()
def gen(
    run_dir: RunOpt,
    config: Annotated[Path | None, typer.Option(help="RunConfig YAML")] = None,
    agents: Annotated[int | None, typer.Option(help="Override number of agents")] = None,
    countries: Annotated[str | None, typer.Option(help="Comma-separated ISO3 codes")] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    start_date: Annotated[str | None, typer.Option(help="YYYY-MM-DD; default today")] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing run directory")] = False,
    allow_later_roster: Annotated[
        bool,
        typer.Option(
            "--allow-later-roster",
            help="Allow a start date before the public-figure rosters' date (for backtests); the figures' "
            "approval and stances may then include later information",
        ),
    ] = False,
) -> None:
    """Generate a new world."""
    rd = RunDir(run_dir)
    if rd.root.exists() and any(rd.root.iterdir()):
        if not force:
            raise typer.BadParameter(f"{rd.root} is not empty (use --force)")
        shutil.rmtree(rd.root)
    cfg = RunConfig.load(config) if config else RunConfig()
    if agents is not None:
        cfg.n_agents = agents
    if countries:
        cfg.countries = [c.strip().upper() for c in countries.split(",")]
    if seed is not None:
        cfg.seed = seed
    if start_date:
        cfg.start_date = dt.date.fromisoformat(start_date)
    problems = validate_data(paths.data_dir(), cfg.countries, start_date=cfg.start_date if cfg.notables else None)
    if allow_later_roster:
        later = [p for p in problems if is_later_roster(p)]
        problems = [p for p in problems if not is_later_roster(p)]
        for p in later:
            typer.echo(f"Warning: {p}; public figures' approval and stances may include later information.")
    if problems:
        raise typer.BadParameter("data problems:\n" + "\n".join(problems))
    t0 = time.perf_counter()
    world = generate_world(cfg)
    rd.init()
    save_snapshot(world, rd.head, label="generated")
    save_snapshot(world, rd.periodic(0), label="day 0")
    typer.echo(
        f"Generated {world.pop.n} agents, {world.households.n} households, {world.edges.n // 2} ties, "
        f"{world.channels.n} channels in {time.perf_counter() - t0:.1f}s -> {rd.root}"
    )
    typer.echo(f"World day 0 = {cfg.start_date.isoformat()}")


def future_window(world_date: dt.date, days: int, real_today: dt.date) -> tuple[dt.date, dt.date] | None:
    """First and last simulated dates of this run that lie after today's real date (no real news), if any."""
    last = world_date + dt.timedelta(days=days - 1)
    first = max(world_date, real_today + dt.timedelta(days=1))
    return (first, last) if days > 0 and first <= last else None


def _offer_feed(rd: RunDir, world, window: tuple[dt.date, dt.date], provider: str, ask: bool) -> None:  # noqa: ANN001
    first, last = window
    note = (
        f"Note: {first} to {last} are after today's real date, so no real news exists for them. "
        "Add hypothetical news or market moves with `jahan feed` if you want any."
    )
    if not ask or not (sys.stdin.isatty() and sys.stdout.isatty()):  # scripts and pipes never block
        typer.echo(note)
        return
    try:
        answer = typer.prompt(
            f"Simulated days {first} to {last} have no real news. Describe events to feed (e.g. 'gold +1%/day "
            "for a week, then -10%'), give a file (.yaml scenario or text), or press Enter to skip",
            default="",
            show_default=False,
        ).strip()
    except typer.Abort:  # end of input
        typer.echo("\n" + note)
        return
    if not answer:
        return
    path = Path(answer)
    # A failed parse or a "no" at the confirmation skips the feed; the run itself goes ahead.
    try:
        if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
            spec, source = Scenario.load(path), str(path)
        else:
            text = path.read_text(encoding="utf-8") if path.is_file() else answer
            offset = (first - world.date()).days
            spec = _parse_interactive(rd, world, text, provider, feed=True, first_future_offset=offset)
            source = "feed"
    except (typer.Abort, typer.Exit, typer.BadParameter, ValidationError, RuntimeError, OSError) as e:
        typer.echo(f"Nothing fed ({e or 'cancelled'}); continuing the run. Use `jahan feed` to try again.")
        return
    try:
        _inject_spec(rd, world, spec, False, source=source)
    except typer.Abort:  # declined at the confirmation, before anything was written
        typer.echo("Nothing fed; continuing the run.")


@app.command("run")
def run_cmd(
    run_dir: RunOpt,
    days: Annotated[int, typer.Option(help="Days to simulate")] = 30,
    provider: Annotated[str, typer.Option(help="LLM for fed events and public figures' decisions")] = "anthropic",
    ask_feed: Annotated[
        bool, typer.Option(help="When the run passes today's real date, offer to feed events (terminal only)")
    ] = True,
    autonomy: Annotated[bool, typer.Option(help="Public figures decide on salient days (LLM, capped)")] = True,
    keep_control: Annotated[
        bool, typer.Option(help="Agents you play stay idle instead of going on autopilot during this run")
    ] = False,
) -> None:
    """Advance the world.  Agents you play are on autopilot meanwhile (your next action takes them back)."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    window = future_window(world.date(), days, dt.datetime.now(dt.UTC).date())
    if window is not None:
        _offer_feed(rd, world, window, provider, ask_feed)
    from jahan.cli_actors import autopilot_for_run

    autopilot_for_run(rd, world, keep_control)
    t0 = time.perf_counter()
    hook = _autonomy_hook(rd, world, provider) if autonomy else None
    from jahan.actors import session

    sid = session.current(rd)
    if sid is not None:
        session.note_autonomy(rd, sid, hook is not None)
    last = _advance(rd, world, days, before_day=hook)
    typer.echo(
        f"Simulated to day {world.day} ({world.date().isoformat()}) in {time.perf_counter() - t0:.1f}s. "
        f"Last day: {last}"
    )
    if hook is not None:
        typer.echo(hook.summary())


class _AutonomyHook:
    def __init__(self, client) -> None:  # noqa: ANN001
        self.client = client
        self.turns = 0

    def __call__(self, world, log: EventLog) -> None:  # noqa: ANN001
        from jahan.actors.autonomy import run_turns

        self.turns += run_turns(self.client, world, log)

    def summary(self) -> str:
        return f"Public figures took {self.turns} decision turns. {self.client.budget.summary()}"


def _autonomy_hook(rd: RunDir, world, provider: str) -> _AutonomyHook | None:  # noqa: ANN001
    if not world.notables:
        return None
    try:
        client = _client(rd, provider)
    except typer.BadParameter as e:
        typer.echo(f"Public figures will not act on their own this run ({e.message}).")
        return None
    client.budget.role_share.setdefault("notable_autonomy", 0.3)
    return _AutonomyHook(client)


def _advance(rd: RunDir, world, days: int, before_day=None) -> dict[str, int]:  # noqa: ANN001
    """Simulate ``days`` days on the head world (parked events go into the first chunk); save the head.

    ``before_day(world, log)`` runs at the start of each day (public-figure autonomy).
    """
    last: dict[str, int] = {}

    def progress(day: int, stats: dict[str, int]) -> None:
        last.update(stats)

    rec = Recorder(world.day)
    parked = rd.pending_files()
    for p in parked:
        rec.events.extend_table(pq.read_table(p))
    run(world, rd, days, progress, recorder=rec, before_day=before_day)
    for p in parked:
        p.unlink(missing_ok=True)
    return last


def _park_events(rd: RunDir, log: EventLog, tag: int | None = None) -> None:
    """Events for the current day that the next `run` writes into its first chunk.

    ``tag``: the prompt these events belong to; loose events (rejections, markers) take the last prompt id.
    """
    if len(log) == 0:
        return
    if tag is None:
        led = rd.ledger()
        tag = max((int(e["prompt_id"]) for e in led), default=0)
    table = log.table()
    day = int(table.column("day")[0].as_py())
    archive = rd.parked_archive(day)
    rd.pending_dir.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    seqs = [int(p.stem.split("_")[0]) for d in (rd.pending_dir, archive) for p in d.glob("*.parquet")]
    name = f"{max(seqs, default=0) + 1:06d}_{tag:04d}.parquet"
    pq.write_table(table, rd.pending_dir / name)
    pq.write_table(table, archive / name)


def _inject_spec(  # noqa: ANN001
    rd: RunDir, world, spec: Scenario, yes: bool, source: str, quiet: bool = False, extra: dict | None = None
) -> int:
    """Snapshot, apply, park day-D events, save head and ledger.  Returns the prompt id."""
    if not quiet:
        typer.echo(spec.dump_yaml())
        for line in describe_moves(preview_moves(world, spec)):
            typer.echo(line)
    if not yes and not typer.confirm(f"Inject this scenario at day {world.day}?"):
        raise typer.Abort()
    pid = rd.next_prompt_id()
    pre = rd.pre_prompt(pid, world.day)
    save_snapshot(world, pre, label=f"before prompt {pid}")
    rd.scenario_file(pid).parent.mkdir(parents=True, exist_ok=True)
    rd.scenario_file(pid).write_text(spec.dump_yaml(), encoding="utf-8")
    rec = Recorder(world.day)
    stats = apply_scenario(world, spec, rec.events, pid)
    # Injection events belong to day D, whose chunk is written by the next `run`; park them until then
    # (appending to events parked by an earlier prompt on the same day).
    _park_events(rd, rec.events, tag=pid)
    save_snapshot(world, rd.head, label="head")
    led = rd.ledger()
    led.append(
        {
            "prompt_id": pid,
            "day": world.day,
            "date": world.date().isoformat(),
            "prompt": spec.prompt,
            "source": source,
            "scenario_file": str(rd.scenario_file(pid).relative_to(rd.root)).replace("\\", "/"),
            "scenario_sha256": hashlib.sha256(rd.scenario_file(pid).read_bytes()).hexdigest(),
            "snapshot_before": str(pre.relative_to(rd.root)).replace("\\", "/"),
            "world_seed": world.sim_seed,
            "injected_at": dt.datetime.now(dt.UTC).isoformat(),
            **(extra or {}),
        }
    )
    rd.save_ledger(led)
    if not quiet:
        typer.echo(f"Prompt {pid} injected at day {world.day}: {stats}")
    return pid


ProviderOpt = Annotated[
    str, typer.Option(help="anthropic (default), openai_compat, or replay (recorded answers only, no network)")
]


def _client(rd: RunDir, provider: str, base_url: str | None = None):  # noqa: ANN202
    from jahan.llm.provider import make_client

    if provider == "replay":
        return make_client(rd.root, "scripted", replay_only=True)
    if provider == "none":
        raise typer.BadParameter("no LLM (provider none)")
    if os.environ.get("JAHAN_OFFLINE"):  # tests and scripts that must never reach a model
        raise typer.BadParameter("LLM provider unavailable: JAHAN_OFFLINE is set (use --provider replay)")
    try:
        return make_client(rd.root, provider, base_url=base_url)
    except RuntimeError as e:
        raise typer.BadParameter(f"LLM provider unavailable: {e}") from e


def _parse_interactive(  # noqa: ANN202
    rd: RunDir,
    world,  # noqa: ANN001
    text: str,
    provider: str,
    feed: bool,
    first_future_offset: int | None = None,
):
    from jahan.scenario.parser import parse_prompt

    client = _client(rd, provider)
    result = parse_prompt(client, world, text, feed=feed, first_future_offset=first_future_offset)
    answers: dict[str, str] = {}
    rounds = 0
    while result.questions and rounds < 3:
        if not sys.stdin.isatty():
            typer.echo("The model needs more detail:\n" + "\n".join(f"- {q}" for q in result.questions))
            raise typer.Exit(2)
        for q in result.questions:
            answers[q] = typer.prompt(q)
        result = parse_prompt(client, world, text, feed=feed, answers=answers, first_future_offset=first_future_offset)
        rounds += 1
    if result.scenario is None:
        raise typer.BadParameter(f"could not turn the prompt into a scenario: {result.error or 'no answer'}")
    typer.echo(client.budget.summary())
    return result.scenario


PreviewOpt = Annotated[
    bool,
    typer.Option(
        "--preview", help="Show the scenario the prompt becomes and inject nothing (re-run with --yes to inject)"
    ),
]


JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output (used by `jahan ui`)")]


def _show_preview(world, spec: Scenario, as_json: bool = False) -> None:  # noqa: ANN001
    moves = preview_moves(world, spec)
    if as_json:
        import json

        typer.echo(
            json.dumps(
                {"day": world.day, "scenario": spec.model_dump(mode="json"), "market_moves": moves}, ensure_ascii=False
            )
        )
        return
    typer.echo(spec.dump_yaml())
    for line in describe_moves(moves):
        typer.echo(line)
    typer.echo(f"Preview only: nothing was injected. The world is on day {world.day} ({world.date().isoformat()}).")


@app.command()
def inject(
    run_dir: RunOpt,
    scenario: Annotated[Path | None, typer.Option(help="Scenario YAML")] = None,
    prompt: Annotated[str | None, typer.Option(help="Free-text what-if prompt (parsed by the LLM)")] = None,
    provider: ProviderOpt = "anthropic",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    preview: PreviewOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Inject a scenario (YAML or free text) at the current day; writes a pre-prompt snapshot first."""
    if (scenario is None) == (prompt is None):
        raise typer.BadParameter("give exactly one of --scenario or --prompt")
    rd = RunDir(run_dir)
    world = _load_head(rd)
    if scenario is not None:
        spec, source = Scenario.load(scenario), str(scenario)
    else:
        spec, source = _parse_interactive(rd, world, str(prompt), provider, feed=False), "llm"
    if preview:
        _show_preview(world, spec, as_json)
        return
    _inject_spec(rd, world, spec, yes, source=source)


@app.command()
def feed(
    run_dir: RunOpt,
    text: Annotated[str, typer.Argument(help='Future events, e.g. "gold +1%/day for a week then -10%"')],
    provider: ProviderOpt = "anthropic",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    preview: PreviewOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Feed news and market moves for simulated days beyond today's real date (there is no real news yet)."""
    rd = RunDir(run_dir)
    world = _load_head(rd)
    real_today = dt.datetime.now(dt.UTC).date()
    offset = (real_today - world.date()).days + 1 if world.date() <= real_today else None
    spec = _parse_interactive(rd, world, text, provider, feed=True, first_future_offset=offset)
    if preview:
        _show_preview(world, spec, as_json)
        return
    _inject_spec(rd, world, spec, yes, source="feed")


@app.command()
def ingest(
    run_dir: RunOpt,
    per_country: Annotated[int, typer.Option(help="Headlines per country that become tracked items")] = 2,
    per_country_social: Annotated[int, typer.Option(help="Social posts per country that become tracked items")] = 2,
    news: Annotated[bool, typer.Option("--news/--no-news", help="Fetch news headlines (RSS feeds)")] = True,
    social: Annotated[
        bool, typer.Option("--social/--no-social", help="Fetch posts from social networks (data/feeds/social.yaml)")
    ] = True,
    x: Annotated[
        bool, typer.Option("--x/--no-x", help="Include X through an unofficial reader that may stop working")
    ] = True,
    provider: Annotated[
        str, typer.Option(help="Headline classifier LLM: anthropic, openai_compat, replay, none")
    ] = "anthropic",
    force: Annotated[bool, typer.Option(help="Ingest even if the world date is far from the real date")] = False,
    fetch_day: Annotated[str | None, typer.Option(help="Real UTC date of the feeds to use (default today)")] = None,
    replay: Annotated[
        bool, typer.Option(help="Use recorded feeds and model answers only; never touch the network")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Fetch today's real headlines and social posts (per country and language) and inject them as news.

    Everything fetched and every model answer is recorded, so `--replay` reproduces the same prompt offline.
    """
    from jahan.ingest.classify import classify, to_scenario
    from jahan.ingest.feeds import Headline, collect
    from jahan.ingest.social import SocialPost, collect_social, to_social_scenario
    from jahan.llm.budget import BudgetExceeded
    from jahan.llm.recordings import MissingRecording, Recordings
    from jahan.model.social import PLATFORMS, load_social

    if not (news or social):
        raise typer.BadParameter("nothing to fetch: --no-news and --no-social together")
    rd = RunDir(run_dir)
    world = _load_head(rd)
    real = dt.date.fromisoformat(fetch_day) if fetch_day else dt.datetime.now(dt.UTC).date()
    if abs((world.date() - real).days) > 3 and not force:
        raise typer.BadParameter(
            f"the world is on {world.date()} but the news is from {real}; real news would be anachronistic "
            "(use --force, or feed hypothetical events with `jahan feed`)"
        )
    client = None
    if provider != "none":
        try:
            client = _client(rd, provider)
        except typer.BadParameter as e:
            typer.echo(
                f"{e.message}\nHeadlines and posts will be labelled neutral: they raise awareness but do not "
                "move mood.  Set the provider's API key (see `jahan doctor`) for topic and tone labels."
            )
    rec = client.recordings if client else Recordings(rd.root / "recordings")
    if replay:
        rec.replay_only = True
    codes = [c.iso3 for c in world.countries]
    problems: list[str] = []
    heads: list[Headline] = []
    if news:
        heads, news_problems = collect(paths.data_dir(), codes, rec, real.isoformat())
        problems += news_problems
    posts: list[SocialPost] = []
    stats: dict[str, dict[str, int]] = {}
    if social:
        posts, social_problems, stats = collect_social(
            paths.data_dir(), codes, rec, real.isoformat(), disabled=set() if x else {"x"}
        )
        problems += social_problems
        if x and load_social(paths.data_dir()).sources["x"].enabled:
            typer.echo("X is read through an unofficial service that may stop working; use --no-x to skip it.")
    if problems:
        failed = EventLog()
        for p in problems:
            typer.echo(f"source problem (skipped): {p}")
            iso = p.split("/", 1)[0]
            failed.add(
                world.day,
                EventKind.WEB_INGEST_FAILED,
                -1,
                -1,
                -1,
                0.0,
                codes.index(iso) if iso in codes else -1,
                p[:300],
            )
        _park_events(rd, failed)  # the gap is part of the record, not just a console message
    if not heads and not posts:
        raise typer.BadParameter("nothing fetched (see the source problems above)")
    item_labels = world.macro.get("item_labels", {})
    pu = world.pushes
    live: set[tuple[str, str]] = set()  # (country, headline) of news still live there
    for r in range(pu.n):
        i = int(pu["item_id"][r])
        if int(world.items["active_until"][i]) >= world.day:
            live.add((codes[int(world.channels["country"][int(pu["channel_id"][r])])], str(item_labels.get(str(i)))))
    fresh = [h for h in heads if (h.country, h.title[:300]) not in live]
    if len(fresh) < len(heads):
        typer.echo(f"{len(heads) - len(fresh)} headlines are already live in this world; skipped")
    seen = set(world.macro.get("social", {}).get("seen", []))
    new_posts = [p for p in posts if p.key not in seen]
    if len(new_posts) < len(posts):
        typer.echo(f"{len(posts) - len(new_posts)} social posts were ingested before; skipped")
    if not fresh and not new_posts:
        typer.echo("Nothing new to ingest.")
        return
    new_posts = _cap_per_country(new_posts, SOCIAL_CLASSIFY_PER_COUNTRY)
    try:
        labels = classify(fresh, client, world.day)
        post_labels = classify(new_posts, client, world.day, kind="posts")
    except (MissingRecording, BudgetExceeded) as e:
        raise typer.BadParameter(f"classification stopped: {e} (use --provider none for neutral labels)") from e
    events: list[Any] = []
    if fresh:
        events += to_scenario(fresh, labels, per_country, news_day=real).events
    n_posts = 0
    if new_posts:
        social_spec, dropped = to_social_scenario(
            new_posts,
            post_labels,
            load_social(paths.data_dir()),
            per_country_social,
            roster_visibility={str(e["key"]): float(e["visibility"]) for e in world.notables},
            language_code=lambda code: world.languages[world.lang_id(code)],
            reachable=_social_reach(world),
            news_day=real,
            stats=stats,
            problems=[p for p in problems if "/" in p and p.split("/", 1)[1].split(":", 1)[0] in PLATFORMS],
        )
        n_posts = sum(1 for ev in social_spec.events if getattr(ev, "type", "") == "social")
        events += social_spec.events
        counts = ", ".join(f"{k} {v['posts']} posts from {v['queries']} queries" for k, v in sorted(stats.items()))
        typer.echo(f"Social posts fetched: {counts or 'none'}.")
        if dropped["no_channel"]:
            typer.echo(
                f"{dropped['no_channel']} posts are in a language nobody in their country reads in this world "
                "(no social channel in that language); they were not used."
            )
        if dropped["private"]:
            typer.echo(f"{dropped['private']} posts by ordinary people count towards trending topics only.")
    n_heads = sum(1 for ev in events if getattr(ev, "type", "") == "news")
    spec = Scenario(
        prompt=f"Live ingest ({n_heads} headlines, {n_posts} social posts)",
        events=events,
        horizon_days=30,
    )
    _inject_spec(rd, world, spec, yes, source="ingest")
    if client is not None:
        typer.echo(client.budget.summary())


SOCIAL_CLASSIFY_PER_COUNTRY = 60  # posts per country sent to the classifier (cost bound)


def _cap_per_country(posts: list[Any], cap: int) -> list[Any]:
    """At most ``cap`` posts per country: public figures first, then channels, then topic searches."""
    order = {"figure": 0, "channel": 1, "topic": 2}
    ranked = sorted(posts, key=lambda p: (order.get(p.query_kind, 3), p.rank))
    kept: list[Any] = []
    count: dict[str, int] = {}
    for p in ranked:
        if count.get(p.country, 0) < cap:
            kept.append(p)
            count[p.country] = count.get(p.country, 0) + 1
    return sorted(kept, key=lambda p: p.rank)  # back to collection order


def _social_reach(world: Any) -> set[tuple[str, str]]:
    """(country, language) pairs that have a social channel in this world."""
    from jahan.model.enums import CHANNEL_KIND_NAMES

    ch = world.channels
    social = int(CHANNEL_KIND_NAMES["social"])
    return {
        (world.countries[int(ch["country"][i])].iso3, world.languages[int(ch["lang"][i])])
        for i in range(ch.n)
        if int(ch["kind"][i]) == social
    }


@app.command()
def ask(
    run_dir: RunOpt,
    question: Annotated[str, typer.Argument()],
    scope: Annotated[str, typer.Option(help="agent:<id> | group:<filter> | world")] = "world",
    sample: Annotated[int, typer.Option(help="Agents to interview for group scope")] = 5,
    provider: ProviderOpt = "anthropic",
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Show snapshots and numbers only")] = False,
) -> None:
    """Ask how one agent, a group or the world sees something (persona interviews grounded in state)."""
    from jahan.query.persona import group_views, interview, sample_group

    rd = RunDir(run_dir)
    world = _load_head(rd)
    client = None if no_llm else _client(rd, provider)
    if scope.startswith("agent:"):
        res = interview(client, world, int(scope.split(":", 1)[1]), question)
        typer.echo(f"{res.name}: {res.answer}" + ("" if res.used_llm else "\n--- snapshot ---\n" + res.snapshot))
    elif scope.startswith("group:"):
        expr = scope.split(":", 1)[1]
        typer.echo(group_views(world, expr, None))
        for aid in sample_group(world, expr, sample):
            res = interview(client, world, aid, question)
            typer.echo(f"- {res.name}: {res.answer}")
    else:
        typer.echo(group_views(world, "alive", None))
        typer.echo("For world-level effects of a scenario, run `jahan experiment` and `jahan report`.")
    if client is not None:
        typer.echo(client.budget.summary())


@app.command()
def transcript(
    run_dir: RunOpt,
    agent_id: Annotated[int, typer.Option("--agent")],
    day: Annotated[int, typer.Option(help="World day to replay")],
    llm: Annotated[bool, typer.Option(help="Let the LLM rewrite the trace as prose")] = False,
    provider: ProviderOpt = "anthropic",
) -> None:
    """What one agent heard, said and believed on one day (deterministic re-simulation with a trace)."""
    from jahan.query.transcript import render, trace_day

    rd = RunDir(run_dir)
    world, rows, _ = trace_day(rd, agent_id, day)
    typer.echo(render(world, agent_id, rows, _client(rd, provider) if llm else None))


@app.command()
def doctor(
    provider: ProviderOpt = "anthropic",
    check_models: Annotated[bool, typer.Option(help="Send one tiny request per configured model")] = False,
) -> None:
    """Check the environment: data, optional LLM dependencies, API key and model IDs."""
    import importlib.util
    import os

    from jahan.llm.budget import Budget
    from jahan.llm.provider import LLMConfig

    problems = validate_data(paths.data_dir(), ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"])
    typer.echo("data: " + ("OK" if not problems else f"{len(problems)} problems"))
    typer.echo(
        "anthropic SDK: " + ("installed" if importlib.util.find_spec("anthropic") else "missing (uv sync --extra llm)")
    )
    typer.echo("ANTHROPIC_API_KEY: " + ("set" if os.environ.get("ANTHROPIC_API_KEY") else "not set"))
    cfg = LLMConfig.from_env()
    typer.echo("configured models: " + ", ".join(f"{k}={v}" for k, v in cfg.models.items()))
    cap = Budget.from_env().max_usd
    typer.echo("USD cap per command: " + (f"${cap:.2f} (JAHAN_MAX_USD)" if cap else "none (token caps only)"))
    failed = False
    if check_models:
        from jahan.llm.provider import AnthropicProvider

        try:
            prov = AnthropicProvider()
        except RuntimeError as e:
            typer.echo(f"cannot check models: {e}")
            raise typer.Exit(1) from e
        for model in sorted(set(cfg.models.values())):
            try:
                out = prov.complete(model, "Reply with OK.", [{"role": "user", "content": "OK?"}], 16, 1.0, "low")
                typer.echo(f"  {model}: OK ({out.input_tokens} in / {out.output_tokens} out tokens)")
            except Exception as e:  # noqa: BLE001
                failed = True
                typer.echo(f"  {model}: FAILED ({type(e).__name__}: {e})")
    if failed or problems:
        raise typer.Exit(1)


@app.command()
def experiment(
    run_dir: RunOpt,
    prompt: Annotated[int | None, typer.Option(help="Prompt id")] = None,
    session: Annotated[int | None, typer.Option(help="Session id: your actions versus autopilot")] = None,
    seeds: Annotated[int, typer.Option(help="Replicate seeds")] = 5,
    horizon: Annotated[int | None, typer.Option(help="Days; default scenario horizon")] = None,
    workers: Annotated[int, typer.Option(help="Parallel processes")] = 4,
) -> None:
    """Run baseline vs treated counterfactuals (a prompt, or a person's whole session)."""
    from jahan.scenario.experiment import run_session_experiment

    if (prompt is None) == (session is None):
        raise typer.BadParameter("give exactly one of --prompt or --session")
    rd = RunDir(run_dir)
    t0 = time.perf_counter()
    if session is not None:
        try:
            out = run_session_experiment(rd, session, seeds, horizon, workers)
        except KeyError as e:
            raise typer.BadParameter(str(e)) from e
    else:
        assert prompt is not None
        out = run_experiment(rd, prompt, seeds, horizon, workers)
    typer.echo(f"Experiment written to {out} in {time.perf_counter() - t0:.1f}s")


@app.command()
def report(
    run_dir: RunOpt,
    prompt: Annotated[int | None, typer.Option(help="Prompt id")] = None,
    session: Annotated[int | None, typer.Option(help="Session id")] = None,
) -> None:
    """Build the Markdown report for an experiment."""
    if (prompt is None) == (session is None):
        raise typer.BadParameter("give exactly one of --prompt or --session")
    if session is not None:
        from jahan.scenario.session_report import build_session_report

        path = build_session_report(RunDir(run_dir), session)
    else:
        assert prompt is not None
        path = build_report(RunDir(run_dir), prompt)
    typer.echo(f"Report: {path}")


@app.command()
def reset(
    run_dir: RunOpt,
    before_prompt: Annotated[int, typer.Option(help="Restore the world as it was before this prompt")],
) -> None:
    """Time travel: restore the world to just before a prompt and discard everything after it."""
    rd = RunDir(run_dir)
    entry = rd.prompt(before_prompt)
    day = int(entry["day"])
    snap = rd.root / entry["snapshot_before"]
    world = load_snapshot(snap)
    removed = truncate_from(rd.events, "events", day) + truncate_from(rd.metrics, "metrics", day)
    for p in list(rd.snapshots.glob("day_*.npz")):
        if int(p.stem.split("_")[1]) > day:
            p.unlink()
            p.with_suffix(".json").unlink(missing_ok=True)
    keep = []
    for e in rd.ledger():
        if e["prompt_id"] >= before_prompt:
            for suffix in (".npz", ".json"):
                (rd.root / e["snapshot_before"]).with_suffix(suffix).unlink(missing_ok=True)
        else:
            keep.append(e)
    rd.save_ledger(keep)
    # Parked rows: drop everything pending, then bring back the day's rows from before this prompt (earlier
    # prompts on the same day, rejections, markers); their chunk was truncated above.
    for p in rd.pending_files():
        p.unlink()
    if (rd.root / "parked").exists():
        for d in (rd.root / "parked").glob("day_*"):
            if int(d.name.split("_")[1]) > day:
                shutil.rmtree(d)
    archive = rd.parked_archive(day)
    if archive.exists():
        rd.pending_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(archive.glob("*.parquet")):
            if int(f.stem.split("_")[1]) >= before_prompt:
                f.unlink()
            else:
                shutil.copyfile(f, rd.pending_dir / f.name)
    save_snapshot(world, rd.head, label=f"reset before prompt {before_prompt}")
    from jahan.actors import session

    undone = session.mark_undone(rd, before_prompt, day)
    if undone:
        typer.echo(f"{undone} session turns undone (kept for `jahan session replay`).")
    meta = rd.meta()
    meta.setdefault("reset_history", []).append(
        {"before_prompt": before_prompt, "day": day, "at": dt.datetime.now(dt.UTC).isoformat()}
    )
    rd.save_meta(meta)
    typer.echo(f"Reset to day {day} (before prompt {before_prompt}); removed {len(removed)} log chunks.")


@app.command()
def status(run_dir: RunOpt, as_json: JsonOpt = False) -> None:
    """Where the world is: date, prompts so far, who you play, reports, and what Claude has cost."""
    import json

    from jahan.query.status import status_text, world_status

    rd = RunDir(run_dir)
    st = world_status(rd, _load_head(rd))
    typer.echo(json.dumps(st, ensure_ascii=False) if as_json else status_text(st))


@app.command("social-feed")
def social_feed(
    run_dir: RunOpt,
    country: Annotated[str | None, typer.Option(help="ISO3 country (default: all)")] = None,
    kind: Annotated[str, typer.Option(help="real (fetched posts), virtual (posts made in the world) or all")] = "all",
    limit: Annotated[int, typer.Option(help="Most recent rows to show")] = 50,
    as_json: JsonOpt = False,
) -> None:
    """Real posts from the last fetches and posts made in the world, newest first, with how far they got."""
    import json

    from jahan.query.social_feed import feed_rows, feed_text

    if kind not in ("real", "virtual", "all"):
        raise typer.BadParameter("kind is real, virtual or all")
    rd = RunDir(run_dir)
    world = _load_head(rd)
    rows = feed_rows(world, country.upper() if country else None, kind, limit)
    typer.echo(json.dumps({"rows": rows}, ensure_ascii=False) if as_json else feed_text(rows))


@app.command()
def agents(
    run_dir: RunOpt,
    filter: Annotated[str | None, typer.Option("--filter", "-f", help='e.g. "age > 30 and info_tier >= 2"')] = None,
    limit: Annotated[int, typer.Option()] = 20,
    as_json: JsonOpt = False,
) -> None:
    """List agents, optionally filtered."""
    from jahan.query.scope import FilterError, evaluate
    from jahan.query.views import list_row

    world = _load_head(RunDir(run_dir))
    try:
        mask = evaluate(world, filter) if filter else np.ones(world.pop.n, dtype=bool)
    except FilterError as e:
        raise typer.BadParameter(str(e)) from e
    mask &= world.pop["alive"]
    idx = np.flatnonzero(mask)
    if as_json:
        import json

        from jahan.query.status import agent_summary

        rows = [agent_summary(world, int(i)) for i in idx[:limit]]
        typer.echo(json.dumps({"matches": int(idx.size), "agents": rows}, ensure_ascii=False))
        return
    typer.echo(f"{idx.size} agents match.")
    for i in idx[:limit]:
        typer.echo(list_row(world, int(i)))


@app.command()
def agent(run_dir: RunOpt, agent_id: Annotated[int, typer.Argument()]) -> None:
    """Show one agent's profile and the world from their perspective."""
    from jahan.query.views import agent_card

    world = _load_head(RunDir(run_dir))
    typer.echo(agent_card(world, agent_id))


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="First port to try (the next free one is used if taken)")] = 8765,
    runs_dir: Annotated[Path, typer.Option(help="Folder holding your worlds")] = Path("runs"),
    browser: Annotated[bool, typer.Option(help="Open the page in your browser")] = True,
    key_file: Annotated[
        Path | None, typer.Option(help="Read the Anthropic API key from this file (instead of ANTHROPIC_API_KEY)")
    ] = None,
) -> None:
    """Open the browser interface: every button runs a jahan command on this computer and shows its output."""
    from jahan.ui.server import serve

    if key_file is not None and not key_file.expanduser().is_file():
        raise typer.BadParameter(f"no key file at {key_file} (a text file holding only your Anthropic API key)")
    if runs_dir.exists() and not runs_dir.is_dir():
        raise typer.BadParameter(f"{runs_dir} is a file, not a folder of worlds")
    project = paths.data_dir().parent
    serve(project, runs_dir if runs_dir.is_absolute() else Path.cwd() / runs_dir, port, browser, key_file)


@app.command("validate-data")
def validate_data_cmd(
    countries: Annotated[str, typer.Option(help="Comma-separated ISO3")] = "USA",
    strict: Annotated[bool, typer.Option(help="Fail on provisional or placeholder sources")] = False,
) -> None:
    """Check country files against the registry."""
    problems = validate_data(paths.data_dir(), [c.strip().upper() for c in countries.split(",")], strict)
    if problems:
        for p in problems:
            typer.echo(f"PROBLEM: {p}")
        raise typer.Exit(1)
    typer.echo("Data OK.")


@app.command("events-hash")
def events_hash(run_dir: RunOpt) -> None:
    """Content hash of the event log (determinism checks)."""
    typer.echo(content_hash(read_all(RunDir(run_dir).events, "events", EVENT_SCHEMA)))


@app.command()
def backtest(
    events: Annotated[Path | None, typer.Option(help="Events YAML (default data/backtests/events_2026.yaml)")] = None,
    config: Annotated[Path | None, typer.Option(help="RunConfig YAML (default configs/m1.yaml)")] = None,
    agents_n: Annotated[int | None, typer.Option("--agents", help="Override the number of agents")] = None,
    seeds: Annotated[int, typer.Option(help="Worlds per event with a news-reach check")] = 5,
    out: Annotated[Path | None, typer.Option(help="Also write the Markdown tables to this file")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Replay real announcements in worlds dated the day before and compare with what happened (offline)."""
    import json

    from jahan.scenario.backtest import BacktestFile, default_events_file, render_markdown, run_backtests, to_json

    cfg_path = config or paths.data_dir().parent / "configs" / "m1.yaml"
    base = RunConfig.load(cfg_path) if cfg_path.exists() else RunConfig(countries=["USA", "DEU"], n_agents=10000)
    if agents_n is not None:
        base.n_agents = agents_n
    file = BacktestFile.load(events or default_events_file())
    say = (lambda _m: None) if as_json else (lambda m: typer.echo(f"  {m}", err=True))
    if not as_json:
        typer.echo(
            "Educational and testing use only; not financial advice; no warranty. Market moves come from published "
            "estimates times the surprise, not from the simulated people.",
            err=True,
        )
    try:
        report = run_backtests(file, base, seeds=seeds, progress=say)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    md = render_markdown(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    if as_json:
        typer.echo(json.dumps(to_json(report), ensure_ascii=False, default=float))
    else:
        typer.echo(md)


@app.command()
def bench(
    agents_n: Annotated[int, typer.Option("--agents")] = 10000,
    days: Annotated[int, typer.Option()] = 30,
    countries: Annotated[str, typer.Option()] = "USA",
    scenario: Annotated[Path | None, typer.Option(help="Inject this scenario on day 0")] = None,
) -> None:
    """Time generation and simulation without writing to disk."""
    from jahan.sim.runner import run_in_memory
    from jahan.store.eventlog import EventLog

    cfg = RunConfig(n_agents=agents_n, countries=[c.strip() for c in countries.split(",")])
    t0 = time.perf_counter()
    world = generate_world(cfg)
    if scenario:
        apply_scenario(world, Scenario.load(scenario), EventLog(), 1)
    t1 = time.perf_counter()
    run_in_memory(world, days)
    t2 = time.perf_counter()
    typer.echo(
        f"generate: {t1 - t0:.2f}s; simulate {days} days: {t2 - t1:.2f}s "
        f"({1000 * (t2 - t1) / max(days, 1):.1f} ms/day; {world.edges.n} directed edges)"
    )


from jahan import cli_actors  # noqa: E402,F401  (registers create-agent, possess, say, post, act, play, ...)

if __name__ == "__main__":
    # `python -m jahan.cli` runs this file as __main__; the actor commands register on the importable
    # module's app, so run that one.
    from jahan.cli import app as _registered_app

    _registered_app()
