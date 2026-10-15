"""`jahan watch`: follow markets, moods and event odds while time moves (registered on jahan.cli.app)."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from jahan.cli import JsonOpt, RunOpt, _load_head, app
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


@watch_app.command("catalog")
def watch_catalog(run_dir: RunOpt, as_json: JsonOpt = False) -> None:
    """Everything this world can watch."""
    from jahan.query.watch import show

    rd = RunDir(run_dir)
    data = show(rd, _load_head(rd), days=0)
    if as_json:
        typer.echo(json.dumps({"day": data["day"], "catalog": data["catalog"]}, ensure_ascii=False))
        return
    for c in data["catalog"]:
        typer.echo(
            f"{'*' if c['watched'] else ' '} {c['id']:<26} {c['label']}" + (f"  ({c['note']})" if c["note"] else "")
        )


@watch_app.command("add")
def watch_add(
    run_dir: RunOpt,
    series: Annotated[list[str], typer.Argument(help="Series ids, e.g. market:us_stocks market:us_tech_stocks")],
) -> None:
    """Start watching series from today (see `jahan watch catalog`)."""
    from jahan.query.watch import add, resolve, show

    rd = RunDir(run_dir)
    world = _load_head(rd)
    try:
        for s in series:
            resolve(world, s)
        added = add(rd, world, series)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    data = show(rd, world, days=0, only=series)
    for s in data["series"]:
        typer.echo(f"Watching {s['label']} from day {world.day}: now {_fmt(s['now'], s['unit'])}")
    if not added:
        typer.echo("Already watched.")


@watch_app.command("remove")
def watch_remove(
    run_dir: RunOpt,
    series: Annotated[list[str], typer.Argument(help="Series ids to stop watching")],
) -> None:
    """Stop watching series."""
    from jahan.query.watch import remove

    gone = remove(RunDir(run_dir), series)
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
