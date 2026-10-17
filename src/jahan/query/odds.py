"""Watched questions for the watch list and the Watch page (read-only), and the real prices fetched for them.

Real prices are observations, not world state: they live in ``runs/<name>/odds/real.json`` (append-only, not
undone by a reset) and never pull the simulated odds.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from jahan.model.questions import KIND_LABELS
from jahan.model.world import World
from jahan.query.watch import SeriesSpec
from jahan.sim.questions import current_odds
from jahan.store.rundir import RunDir

REAL_FILE = "odds/real.json"


def load_real(rd: RunDir) -> dict[str, Any]:
    data = rd._read(REAL_FILE, {"questions": {}, "problems": []})
    data.setdefault("questions", {})
    data.setdefault("problems", [])
    return dict(data)


def append_real(rd: RunDir, rows: dict[str, dict[str, Any]], problems: list[str], fetch_date: str) -> None:
    """Add one observation per question (same fetch date: replaced) and the fetch's problems."""
    (rd.root / "odds").mkdir(parents=True, exist_ok=True)
    data = load_real(rd)
    for qid, row in rows.items():
        series = [r for r in data["questions"].get(qid, []) if r.get("fetch_date") != row["fetch_date"]]
        data["questions"][qid] = sorted(series + [row], key=lambda r: str(r["fetch_date"]))
    data["problems"] = (data["problems"] + [{"fetch_date": fetch_date, "problem": p} for p in problems])[-100:]
    rd._write(REAL_FILE, data)


def _movers(world: World, q: dict[str, Any], top: int = 3) -> list[dict[str, Any]]:
    labels = world.macro.get("item_labels", {})
    claims = [(int(k), c) for k, c in q["claims"].items() if abs(float(c.get("moved", 0.0))) > 0]
    claims.sort(key=lambda kc: -abs(float(kc[1]["moved"])))
    return [
        {
            "item": k,
            "label": labels.get(str(k), f"item {k}"),
            "day": c.get("day"),
            "direction": c.get("direction"),
            "by": c.get("by"),
            "why": c.get("why", ""),
            "moved": c.get("moved"),
        }
        for k, c in claims[:top]
    ]


def question_rows(rd: RunDir, world: World) -> list[dict[str, Any]]:
    st = world.macro.get("odds")
    if st is None:
        return []
    real = load_real(rd)["questions"]
    now = current_odds(world)
    rows = []
    for qid, q in st["questions"].items():
        series = real.get(qid, [])
        sim = now.get(qid)
        rows.append(
            {
                "series": f"odds:{qid}",
                "id": qid,
                "question": q["question"],
                "resolves_by": q.get("resolves_by"),
                "status": q["status"],
                "source": q.get("source"),
                "seed": q.get("seed"),
                "real": series[-1] if series else None,
                "real_series": series,
                "simulated": None
                if sim is None
                else {"odds": sim["odds"], "n_traders": sim["n_traders"], "represents": sim["represents"]},
                "by_country": None if sim is None else sim["by_country"],
                "added_day": q["added_day"],
                "closed_day": q.get("closed_day"),
                "final_odds": q.get("final_odds"),
                "movers": _movers(world, q),
                "claims": len(q["claims"]),
                "kind": q.get("kind", "other"),
                "kind_by": q.get("kind_by"),
                "kind_why": q.get("kind_why", ""),
                "kind_label": KIND_LABELS[q.get("kind", "other")],
                "days_left": None
                if not q.get("resolves_by")
                else (dt.date.fromisoformat(str(q["resolves_by"])) - world.date()).days + 1,
            }
        )
    return rows


def question_specs(world: World) -> list[SeriesSpec]:
    """Each watched question as a watch-list series (open and closed: a closed one keeps its history)."""
    st = world.macro.get("odds")
    if st is None:
        return []
    now = current_odds(world)
    out = []
    for qid, q in st["questions"].items():
        sim = now.get(qid)
        out.append(
            SeriesSpec(
                f"odds:{qid}",
                "odds",
                q["question"],
                f"odds.{qid}",
                item=int(q["item"]),
                unit="probability",
                components=("time", "news", "talk"),
                note=("resolves " + str(q["resolves_by"]) if q.get("resolves_by") else "")
                + ("" if q["status"] == "open" else f"; closed on day {q.get('closed_day')}"),
                extra={"now": float("nan") if sim is None else sim["odds"], "qid": qid, "added_day": q["added_day"]},
            )
        )
    return out
