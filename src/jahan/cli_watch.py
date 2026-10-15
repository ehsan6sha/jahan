"""`jahan watch`: follow markets, moods and event odds while time moves (registered on jahan.cli.app)."""

from __future__ import annotations

import datetime as dt
import json
from typing import Annotated

import typer

from jahan import paths
from jahan.cli import JsonOpt, RunOpt, _inject_spec, _load_head, app
from jahan.store.rundir import RunDir

watch_app = typer.Typer(
    help="Watch list: follow US stocks, US tech stocks, gold, currencies, moods and event odds as time moves.",
    no_args_is_help=True,
)
app.add_typer(watch_app, name="watch")

DaysOpt = Annotated[int, typer.Option(help="Days of history to show")]


def _fmt(value: float | None, unit: str) -> str:
    if value is None or value != value:  # NaN
        return "n/a"
    if unit == "probability":
        return f"{100 * value:.1f}%"
    if unit == "mood":
        return f"{value:+.3f}"
    if unit == "rate":
        return f"{value:,.4g}"
    return f"{value:.2f}"


def _since(s: dict) -> str:
    if s["unit"] == "probability":
        return f"{100 * s['since_watched_abs']:+.1f} points"
    if s["unit"] == "mood":
        return f"{s['since_watched_abs']:+.3f}"
    pct = s["since_watched_pct"]
    return "n/a" if pct is None else f"{pct:+.2f}%"


def _seed_rows() -> list[dict]:
    from jahan.model.questions import load_question_seeds

    seeds = load_question_seeds(paths.data_dir())
    if seeds is None:
        return []
    return [
        {
            "series": f"odds:seed:{q.id}",
            "id": q.id,
            "question": q.question,
            "yes_price": q.price,
            "fetch_date": seeds.fetch_date,
            "resolves_by": q.resolves_by,
            "url": f"https://polymarket.com/event/{q.event_slug}" if q.event_slug else "",
            "countries": q.countries,
            "offline": True,
        }
        for q in seeds.questions
    ]


@watch_app.command("catalog")
def watch_catalog(run_dir: RunOpt, as_json: JsonOpt = False) -> None:
    """Everything this world can watch, and questions you can add (offline seeds read from Polymarket)."""
    from jahan.query.watch import show

    rd = RunDir(run_dir)
    data = show(rd, _load_head(rd), days=0)
    seeds = _seed_rows()
    if as_json:
        typer.echo(json.dumps({"day": data["day"], "catalog": data["catalog"], "seeds": seeds}, ensure_ascii=False))
        return
    for c in data["catalog"]:
        typer.echo(
            f"{'*' if c['watched'] else ' '} {c['id']:<26} {c['label']}" + (f"  ({c['note']})" if c["note"] else "")
        )
    if seeds:
        typer.echo(
            f"Questions you can add (Polymarket prices on {seeds[0]['fetch_date']}; `watch add odds:seed:<id>`):"
        )
        for s in seeds:
            typer.echo(f"  {100 * s['yes_price']:5.1f}%  {s['question']}  [{s['id']}]")


QUESTION_HELP = (
    "Series ids: market:us_stocks, market:us_tech_stocks, market:gold, market:dollar, market:fx:IRN, mood:USA, "
    "...; or a question: odds:seed:<id> (data/global/questions.yaml), odds:custom:<id> with --question"
)


def _seed(qid: str):  # noqa: ANN202
    from jahan.model.questions import load_question_seeds

    seeds = load_question_seeds(paths.data_dir())
    if seeds is not None:
        for q in seeds.questions:
            if q.id == qid:
                return q, seeds.fetch_date
    known = ", ".join(q.id for q in seeds.questions) if seeds else "none"
    raise typer.BadParameter(f"no seed question {qid!r} in data/global/questions.yaml (known: {known})")


def _question_event(world, sid: str, opts: dict):  # noqa: ANN001, ANN202
    """A QuestionEvent for ``odds:seed:<id>`` or ``odds:custom:<id>``, with the trader settings of odds.yaml."""
    from jahan.model.questions import load_odds_config
    from jahan.query.status import INGEST_WINDOW_DAYS
    from jahan.scenario.spec import QuestionEvent, QuestionSource

    cfg = load_odds_config(paths.data_dir())
    if cfg is None:
        raise typer.BadParameter("data/global/odds.yaml is missing")
    traders = {
        "trader_share": cfg.trader_share.value,
        "adult_age": cfg.trader_share.adult_age,
        "skew": cfg.skew.model_dump(),
        "sources": cfg.source_ids(),
    }
    kind, _, qid = sid[len("odds:") :].partition(":")
    try:
        if kind == "seed":
            q, fetched = _seed(qid)
            gap = abs((world.date() - dt.date.fromisoformat(fetched)).days)
            if gap > INGEST_WINDOW_DAYS and not opts["force"]:
                raise typer.BadParameter(
                    f"the seed price of {qid!r} is from {fetched}, but the world is on {world.date().isoformat()}: "
                    "add --force to start from it anyway, or use odds:custom:<id> with your own --prior"
                )
            url = f"https://polymarket.com/event/{q.event_slug}" if q.event_slug else ""
            return QuestionEvent(
                id=q.id,
                question=q.question,
                description=q.note,
                resolves_by=q.resolves_by,
                source=QuestionSource(name="polymarket", url=url, slug=q.id, event_slug=q.event_slug),
                price=q.price,
                fetch_date=fetched,
                countries=opts["countries"] or q.countries,
                partisan=opts["partisan"],
                topic=q.topic,
                **traders,
            )
        if kind == "custom":
            if not opts["question"]:
                raise typer.BadParameter('odds:custom:<id> needs --question "Will ... by ...?"')
            return QuestionEvent(
                id=qid,
                question=opts["question"],
                resolves_by=opts["resolves_by"],
                source=QuestionSource(name="manual"),
                prior=opts["prior"],
                countries=opts["countries"],
                partisan=opts["partisan"],
                topic=opts["topic"],
                **traders,
            )
    except ValueError as e:  # pydantic: a bad id, date or range
        raise typer.BadParameter(str(e)) from e
    raise typer.BadParameter(
        f"{sid!r}: use odds:seed:<id> (offline seeds, see `jahan watch catalog`) or odds:custom:<id> --question ..."
    )


@watch_app.command("add")
def watch_add(
    run_dir: RunOpt,
    series: Annotated[list[str], typer.Argument(help=QUESTION_HELP)],
    question: Annotated[str | None, typer.Option(help="odds:custom: the yes/no question")] = None,
    resolves_by: Annotated[str | None, typer.Option(help="odds:custom: the date it resolves, YYYY-MM-DD")] = None,
    prior: Annotated[float, typer.Option(help="odds:custom: the traders' starting probability of Yes")] = 0.5,
    countries: Annotated[str | None, typer.Option(help="Questions: whose residents trade on it, e.g. USA,IRN")] = None,
    partisan: Annotated[
        float, typer.Option(help="Questions: -1..1, how far the governing camp leans Yes (heuristic)")
    ] = 0.0,
    topic: Annotated[str, typer.Option(help="odds:custom: topic of the question")] = "world",
    force: Annotated[bool, typer.Option(help="Start from a seed price far from the world's date")] = False,
) -> None:
    """Start watching series from today.  A question becomes a numbered prompt (undo it like any other)."""
    from jahan.query.odds import append_real
    from jahan.query.watch import add, resolve, show
    from jahan.scenario.inject import InjectError
    from jahan.scenario.spec import Scenario
    from jahan.sim.questions import current_odds

    rd = RunDir(run_dir)
    world = _load_head(rd)
    opts = {
        "question": question,
        "resolves_by": resolves_by,
        "prior": prior,
        "countries": [c.strip() for c in countries.split(",")] if countries else None,
        "partisan": partisan,
        "topic": topic,
        "force": force,
    }
    new_q = [s for s in series if s.startswith(("odds:seed:", "odds:custom:"))]
    plain = [s for s in series if s not in new_q]  # markets, moods, and questions already open in the world
    try:
        for s in plain:
            resolve(world, s)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    for sid in new_q:
        ev = _question_event(world, sid, opts)
        try:
            _inject_spec(rd, world, Scenario(prompt=f"Watch: {ev.question}", events=[ev]), True, "watch", quiet=True)
        except InjectError as e:
            raise typer.BadParameter(str(e)) from e
        odds = current_odds(world)[ev.id]
        add(rd, world, [f"odds:{ev.id}"], {f"odds:{ev.id}": odds["odds"]})
        if ev.price is not None and ev.source is not None:
            row = {"fetch_date": ev.fetch_date, "price": ev.price, "source_url": ev.source.url, "closed": False}
            append_real(rd, {ev.id: row}, [], str(ev.fetch_date))
        typer.echo(
            f'Watching "{ev.question}" (odds:{ev.id}) from day {world.day}: {odds["n_traders"]} simulated traders, '
            f"odds {100 * odds['odds']:.1f}%"
            + ("" if ev.price is None else f" (market {100 * ev.price:.1f}% on {ev.fetch_date})")
        )
    if plain:
        added = add(rd, world, plain)
        data = show(rd, world, days=0, only=plain)
        for s in data["series"]:
            typer.echo(f"Watching {s['label']} from day {world.day}: now {_fmt(s['now'], s['unit'])}")
        if not added:
            typer.echo("Already watched.")


@watch_app.command("remove")
def watch_remove(
    run_dir: RunOpt,
    series: Annotated[list[str], typer.Argument(help="Series ids to stop watching")],
) -> None:
    """Stop watching series.  A question stops being simulated too (a numbered prompt you can undo)."""
    from jahan.query.watch import remove
    from jahan.scenario.inject import InjectError
    from jahan.scenario.spec import QuestionEvent, Scenario
    from jahan.sim.questions import open_questions

    rd = RunDir(run_dir)
    open_ids = None
    for sid in series:
        if not sid.startswith("odds:"):
            continue
        qid = sid[len("odds:") :]
        if open_ids is None:
            world = _load_head(rd)
            open_ids = {q for q, _ in open_questions(world)}
        if qid in open_ids:
            try:
                _inject_spec(
                    rd,
                    world,
                    Scenario(prompt=f"Stop watching {qid}", events=[QuestionEvent(id=qid, action="close")]),
                    True,
                    "watch",
                    quiet=True,
                )
            except InjectError as e:
                raise typer.BadParameter(str(e)) from e
    gone = remove(rd, series)
    typer.echo(f"Stopped watching {', '.join(gone)}." if gone else "None of these were watched.")


@watch_app.command("list")
def watch_list(run_dir: RunOpt, as_json: JsonOpt = False) -> None:
    """The series you watch and since when."""
    from jahan.query.watch import load_list

    watched = load_list(RunDir(run_dir))
    if as_json:
        typer.echo(json.dumps(watched, ensure_ascii=False))
        return
    if not watched:
        typer.echo("Nothing watched yet; see `jahan watch catalog`.")
    for k, v in watched.items():
        typer.echo(f"{k:<26} since day {v.get('since_day')}")


@watch_app.command("show")
def watch_show(run_dir: RunOpt, days: DaysOpt = 60, as_json: JsonOpt = False) -> None:
    """How each watched series moved since you started watching it, with its recent history."""
    from jahan.query.watch import show

    rd = RunDir(run_dir)
    data = show(rd, _load_head(rd), days=days)
    if as_json:
        typer.echo(json.dumps(data, ensure_ascii=False, default=float))
        return
    typer.echo(f"Day {data['day']} ({data['date']}). {data['disclaimer']}")
    if not data["series"]:
        typer.echo("Nothing watched yet; see `jahan watch catalog`.")
    for s in data["series"]:
        line = f"{s['label']}: {_fmt(s['now'], s['unit'])}, {_since(s)} since day {s['since_day']}" + (
            " (rebased after an undo)" if s["rebased"] else ""
        )
        c = s["components"]
        if c is not None:
            line += f"; people's trading {c['people_pct']:+.2f}%, announcements and scripted {c['scripted_pct']:+.2f}%"
        typer.echo(line)
