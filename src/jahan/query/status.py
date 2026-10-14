"""Machine-readable summaries of a run directory and of agents, for `jahan status --json` and the browser UI.

Everything here is read-only: it describes the world, the prompt ledger, the person's session, what has
been computed (experiments, reports) and what the model layer has cost so far.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from typing import Any

import numpy as np

from jahan.actors import session
from jahan.actors.engine import allowed_audiences, display_name, notable_entry
from jahan.llm.budget import DEFAULT_PRICES
from jahan.llm.provider import LLMConfig
from jahan.model.enums import TOPIC_NAMES, EmpStatus, Marital, Sex
from jahan.model.world import World
from jahan.query.views import ISCO_LABELS, age_of
from jahan.store.rundir import RunDir

INGEST_WINDOW_DAYS = 3  # `ingest` refuses when the world date is further than this from the real date
INFO_TIERS = ["minimal", "major news", "well informed", "specialist"]  # info_tier 0..3


def agent_summary(world: World, i: int) -> dict[str, Any]:
    """One row of an agent list (index ``i`` into the population table)."""
    pop = world.pop
    employed = int(pop["emp_status"][i]) == EmpStatus.EMPLOYED
    entry = notable_entry(world, i)
    return {
        "id": int(pop["id"][i]),
        "name": display_name(world, i),
        "sex": "F" if int(pop["sex"][i]) == Sex.F else "M",
        "age": age_of(world, i),
        "country": world.countries[int(pop["country"][i])].iso3,
        "occupation": ISCO_LABELS.get(int(pop["occupation"][i]), "?")
        if employed
        else EmpStatus(int(pop["emp_status"][i])).name.lower(),
        "info_tier": int(pop["info_tier"][i]),
        "alive": bool(pop["alive"][i]),
        "public_figure": None if entry is None else {"key": entry["key"], "office": entry["office"]},
        "controlled_by": int(pop["controlled_by"][i]),
        "alignment": round(float(pop["alignment"][i]), 3),
    }


def _policies(world: World, row: int) -> list[dict[str, Any]]:
    entry = notable_entry(world, row)
    if entry is None or entry["role"] not in world.roles:
        return []
    out = []
    for kind, spec in sorted(world.roles[entry["role"]].policies.items()):
        param = None if spec.param is None else {"name": spec.param.name, "min": spec.param.min, "max": spec.param.max}
        out.append({"kind": kind, "effect": spec.effect, "scope": spec.scope, "param": param, "note": spec.note})
    return out


def _llm_spend(rd: RunDir) -> dict[str, Any]:
    """Tokens and approximate dollars of every model answer recorded in this run (list prices, see budget)."""
    tin = tout = 0
    usd = 0.0
    unpriced: set[str] = set()
    manifest = rd.root / "recordings" / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "llm":
                continue
            i, o = int(row.get("input_tokens", 0)), int(row.get("output_tokens", 0))
            tin, tout = tin + i, tout + o
            price = DEFAULT_PRICES.get(str(row.get("model")))
            if price is None or price.usd_per_mtok_input is None or price.usd_per_mtok_output is None:
                unpriced.add(str(row.get("model")))
            else:
                usd += i / 1e6 * price.usd_per_mtok_input + o / 1e6 * price.usd_per_mtok_output
    return {"input_tokens": tin, "output_tokens": tout, "usd": round(usd, 4), "unpriced_models": sorted(unpriced)}


def world_status(rd: RunDir, world: World) -> dict[str, Any]:
    pop = world.pop
    real_today = dt.datetime.now(dt.UTC).date()
    ahead = (world.date() - real_today).days
    alive = pop["alive"]
    sessions = session.load(rd)
    sid = session.current(rd)
    played = []
    if sid is not None:
        for a in session.agents(rd, sid):
            if not 0 <= a < pop.n:
                continue
            row = world.agent_row(a)
            entry = notable_entry(world, row)
            played.append(
                {
                    **agent_summary(world, row),
                    "controlled_now": int(pop["controlled_by"][row]) == sid,
                    "office": None if entry is None else entry["office"],
                    "audiences": allowed_audiences(world, row),
                    "policies": _policies(world, row),
                }
            )
    reports = sorted(str(p.relative_to(rd.root)).replace("\\", "/") for p in (rd.root / "reports").glob("*/report.md"))
    experiments = sorted(p.name for p in (rd.root / "experiments").glob("*") if p.is_dir())
    cfg = LLMConfig.from_env()
    sdk = importlib.util.find_spec("anthropic") is not None
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "run": str(rd.root),
        "day": int(world.day),
        "date": world.date().isoformat(),
        "start_date": world.config.start_date.isoformat(),
        "real_today": real_today.isoformat(),
        "days_ahead_of_today": ahead,
        "ingest_allowed": abs(ahead) <= INGEST_WINDOW_DAYS,
        "agents_alive": int(alive.sum()),
        "countries": [
            {
                "iso3": c.iso3,
                "name": c.name,
                "agents": int((alive & (pop["country"] == k)).sum()),
                "languages": sorted(
                    {world.languages[int(x)] for x in np.unique(pop["lang"][alive & (pop["country"] == k)]) if x >= 0}
                ),
                "politics": None
                if c.politics is None
                else {
                    "pos": c.politics.axis.pos,
                    "neg": c.politics.axis.neg,
                    "camps": c.politics.camps,
                    "placeholder": c.politics.placeholder,
                },
            }
            for k, c in enumerate(world.countries)
        ],
        "products": list(world.product_names),
        "topics": list(TOPIC_NAMES),
        "prompts": [
            {
                "prompt_id": int(e["prompt_id"]),
                "day": int(e["day"]),
                "date": e.get("date"),
                "prompt": e.get("prompt", ""),
                "source": e.get("source", ""),
                "session": e.get("session"),
                "scenario_file": e.get("scenario_file"),
            }
            for e in rd.ledger()
        ],
        "vocab": {
            "status": [s.name.lower() for s in EmpStatus],
            "marital": [m.name.lower() for m in Marital],
            "info_tiers": INFO_TIERS,
            "audiences": ["circle", "community", "national", "global"],
            "roles": [{"key": k, "label": r.label} for k, r in sorted(world.roles.items())],
        },
        "session": {"current": sid, "agents": played},
        "sessions": [
            {
                "id": int(k),
                "agents": [int(a) for a in v.get("agents", [])],
                "turns": len(v.get("turns", [])),
                "undone": sum(1 for t in v.get("turns", []) if t.get("undone")),
            }
            for k, v in sorted(sessions["sessions"].items(), key=lambda kv: int(kv[0]))
        ],
        "public_figures": [
            {
                "key": e["key"],
                "name": e["holder_name"],
                "office": e["office"],
                "role": e["role"],
                "country": world.countries[int(e["country"])].iso3,
                "agent_id": int(e["agent_id"]),
                "alignment": float(e.get("alignment", 0.0)),
                "camp_status": e.get("camp_status", "placeholder"),
            }
            for e in world.notables
        ],
        "experiments": experiments,
        "reports": reports,
        "llm": {
            "sdk_installed": sdk,
            "api_key_set": key,
            "ready": sdk and key and not os.environ.get("JAHAN_OFFLINE"),
            "models": cfg.models,
            "spent_in_this_world": _llm_spend(rd),
        },
    }


def status_text(st: dict[str, Any]) -> str:
    """The same summary for people reading a terminal."""
    lines = [
        f"World {st['run']}: day {st['day']} ({st['date']}); {st['agents_alive']:,} people alive in "
        + ", ".join(c["iso3"] for c in st["countries"]),
    ]
    ahead = st["days_ahead_of_today"]
    when = "today" if ahead == 0 else f"{abs(ahead)} days {'after' if ahead > 0 else 'before'} today"
    lines.append(f"World date is {when}; real news can {'' if st['ingest_allowed'] else 'not '}be ingested.")
    lines.append(f"Prompts: {len(st['prompts'])}")
    for p in st["prompts"]:
        lines.append(f"  {p['prompt_id']:>3}  day {p['day']:>4}  {p['source']:<12} {str(p['prompt'])[:70]}")
    sess = st["session"]
    if sess["current"] is not None and sess["agents"]:
        who = ", ".join(f"{a['name']} ({a['id']})" for a in sess["agents"])
        lines.append(f"Session {sess['current']}: you play {who}")
    if st["products"]:
        lines.append("Products: " + ", ".join(st["products"]))
    if st["reports"]:
        lines.append("Reports: " + ", ".join(st["reports"]))
    llm = st["llm"]
    spent = llm["spent_in_this_world"]
    lines.append(
        f"Claude: {'ready' if llm['ready'] else 'not configured (see `jahan doctor`)'}; spent in this world "
        f"about ${spent['usd']:.4f} ({spent['input_tokens']:,} input / {spent['output_tokens']:,} output tokens)"
    )
    return "\n".join(lines)
