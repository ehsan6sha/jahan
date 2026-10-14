"""Sessions: a person controlling one or more agents over several turns.

``sessions.json`` in the run directory lists sessions and their turns.  Every turn is a ledger prompt
(pre-turn snapshot, so `reset --before-prompt` undoes it) and a copy of its actions is kept under
``sessions/<id>/`` so the session can be replayed after a reset (`jahan session replay`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import shutil
from typing import Any

from jahan.store.rundir import RunDir

FILE = "sessions.json"


def load(rd: RunDir) -> dict[str, Any]:
    data: dict[str, Any] = rd._read(FILE, {"next_id": 1, "current": None, "sessions": {}})
    return data


def save(rd: RunDir, data: dict[str, Any]) -> None:
    rd._write(FILE, data)


def current(rd: RunDir, create: bool = False) -> int | None:
    data = load(rd)
    sid = data.get("current")
    if sid is not None and str(sid) in data["sessions"]:
        return int(sid)
    if not create:
        return None
    return new_session(rd)


def new_session(rd: RunDir) -> int:
    data = load(rd)
    sid = int(data["next_id"])
    data["next_id"] = sid + 1
    data["current"] = sid
    data["sessions"][str(sid)] = {
        "id": sid,
        "created": dt.datetime.now(dt.UTC).isoformat(),
        "agents": [],
        "autonomy": False,
        "turns": [],
    }
    save(rd, data)
    return sid


def agents(rd: RunDir, sid: int) -> list[int]:
    """The agents this person plays (controlled at their turns, on autopilot otherwise)."""
    return [int(a) for a in load(rd)["sessions"][str(sid)].get("agents", [])]


def set_agents(rd: RunDir, sid: int, add: list[int] = (), remove: list[int] = ()) -> None:  # type: ignore[assignment]
    data = load(rd)
    sess = data["sessions"][str(sid)]
    cur = [int(a) for a in sess.get("agents", []) if int(a) not in remove]
    sess["agents"] = cur + [int(a) for a in add if int(a) not in cur]
    save(rd, data)


def note_autonomy(rd: RunDir, sid: int, active: bool) -> None:
    """Remember that public figures acted on their own while this session ran (replay must match)."""
    if not active:
        return
    data = load(rd)
    data["sessions"][str(sid)]["autonomy"] = True
    save(rd, data)


def autonomy_used(rd: RunDir, sid: int) -> bool:
    return bool(load(rd)["sessions"][str(sid)].get("autonomy", False))


def record_turn(rd: RunDir, sid: int, prompt_id: int, day: int, summary: str, fingerprint: str) -> None:
    """``fingerprint``: world state the turn was applied to (store.snapshot.world_fingerprint)."""
    data = load(rd)
    sess = data["sessions"][str(sid)]
    k = len(sess["turns"]) + 1
    dest = rd.root / "sessions" / f"{sid:04d}" / f"turn_{k:04d}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rd.scenario_file(prompt_id), dest)
    sess["turns"].append(
        {
            "turn": k,
            "prompt_id": prompt_id,
            "day": day,
            "file": str(dest.relative_to(rd.root)).replace("\\", "/"),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "world_before": fingerprint,
            "summary": summary[:200],
            "undone": False,
        }
    )
    save(rd, data)


def record_autopilot(rd: RunDir, sid: int, day: int, agents: list[int], fingerprint: str) -> None:
    """A day the person let pass (timeout or `auto`): recorded so a replay reproduces the AUTOPILOT_TURN log."""
    data = load(rd)
    sess = data["sessions"][str(sid)]
    sess["turns"].append(
        {
            "turn": len(sess["turns"]) + 1,
            "prompt_id": None,
            "day": day,
            "autopilot": agents,
            "world_before": fingerprint,
            "undone": False,
        }
    )
    save(rd, data)


def verify_turn(rd: RunDir, turn: dict[str, Any], fingerprint: str) -> str | None:
    """Why a recorded turn must not be replayed onto the current world, or None."""
    if turn.get("prompt_id") is not None:
        path = rd.root / turn["file"]
        if not path.exists():
            return f"turn {turn['turn']}: its recorded actions file {turn['file']} is missing"
        if turn.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != turn["sha256"]:
            return f"turn {turn['turn']}: {turn['file']} was edited after it was recorded"
    if turn.get("world_before") and turn["world_before"] != fingerprint:
        return (
            f"turn {turn['turn']}: the world on day {turn['day']} is not the state the turn was recorded on "
            "(something earlier differs: another prompt, a different seed, or edited data)"
        )
    return None


def mark_undone(rd: RunDir, from_prompt: int, from_day: int) -> int:
    """After a reset: turns from ``from_prompt`` (autopilot days from ``from_day``) are no longer in the
    timeline; they are kept for replay."""
    data = load(rd)
    n = 0
    for sess in data["sessions"].values():
        for t in sess["turns"]:
            pid = t.get("prompt_id")
            gone = (pid is not None and pid >= from_prompt) or (pid is None and t["day"] >= from_day)
            if not t["undone"] and gone:
                t["undone"] = True
                n += 1
    save(rd, data)
    return n


def turns(rd: RunDir, sid: int) -> list[dict[str, Any]]:
    data = load(rd)
    sess = data["sessions"].get(str(sid))
    if sess is None:
        raise KeyError(f"no session {sid}")
    return list(sess["turns"])
