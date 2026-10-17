"""Worlds without everyday activity simulate exactly as before it existed.

`fixtures/unswitched_fingerprints.json` holds `world_fingerprint` values of three small worlds (one with people's
trading, one made before people traded, one with a watched question) on days 0, 10 and 30, written by the code
as it was before everyday activity (`macro["economy"]`, per-person arrays `ps.*`) existed.  The same worlds made
with `economy=False` must reproduce them, so old saved worlds and recorded sessions keep their fingerprints.

Simulated floats can differ in their last bits between machines (BLAS kernels), so the fixture is keyed by
platform and the check is skipped where no fixture was written; the frozen copy of the fingerprint function
below is checked everywhere.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import Bearing, MacroPath, MacroStep, NewsEvent, QuestionEvent, Scenario
from jahan.sim.runner import run_in_memory
from jahan.store.eventlog import EventLog
from jahan.store.snapshot import TABLES, world_fingerprint

FIXTURE = Path(__file__).parent / "fixtures" / "unswitched_fingerprints.json"
DAYS = (0, 10, 30)


def _platform() -> str:
    return f"{sys.platform}-{platform.machine()}-numpy{np.__version__}"


def _config(countries: list[str], seed: int) -> RunConfig:
    extra = {"economy": False} if "economy" in RunConfig.model_fields else {}
    return RunConfig(n_agents=1200, countries=countries, seed=seed, start_date="2026-10-13", **extra)


def _news() -> NewsEvent:
    return NewsEvent(
        headline="US economy shrinks as layoffs spread",
        topic="economy",
        valence=-0.7,
        salience=0.9,
        countries=["USA"],
        duration_days=7,
    )


def _with_market():  # noqa: ANN202
    w = generate_world(_config(["USA", "DEU"], 7))
    gold = MacroPath(variable="gold", steps=[MacroStep(days=3, pct_per_day=1.0)])
    apply_scenario(w, Scenario(prompt="slump", events=[_news(), gold]), EventLog(), 1)
    return w


def _before_market():  # noqa: ANN202
    w = generate_world(_config(["USA", "DEU"], 7))
    del w.macro["market"]  # a world made before people traded
    apply_scenario(w, Scenario(prompt="slump", events=[_news()]), EventLog(), 1)
    return w


def _with_question():  # noqa: ANN202
    w = generate_world(_config(["USA", "IRN"], 9))
    q = QuestionEvent(
        id="us-x-iran-ceasefire-fixture",
        question="US x Iran ceasefire continues through December 31?",
        price=0.7,
        fetch_date="2026-10-13",
        resolves_by="2026-12-31",
        countries=["USA", "IRN"],
    )
    story = NewsEvent(
        headline="US and Iran trade fire in the Gulf",
        topic="world",
        valence=-0.6,
        salience=0.8,
        countries=["USA", "IRN"],
        bears_on=[Bearing(question=q.id, direction=-0.6)],
    )
    apply_scenario(w, Scenario(prompt="watch", events=[q, story]), EventLog(), 1)
    return w


WORLDS = {"with_market": _with_market, "before_market": _before_market, "with_question": _with_question}


def fingerprints(make) -> list[str]:  # noqa: ANN001
    w = make()
    out, day = [], 0
    for target in DAYS:
        if target > day:
            run_in_memory(w, target - day)
            day = target
        out.append(world_fingerprint(w))
    return out


def write_fixture() -> None:
    """Run once on the code before everyday activity existed: `uv run python -c "from
    tests.determinism.test_unswitched_worlds import write_fixture; write_fixture()"`."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}
    data[_platform()] = {name: fingerprints(make) for name, make in WORLDS.items()}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_unswitched_worlds_keep_their_fingerprints(name):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if _platform() not in data:
        pytest.skip(f"no fingerprints recorded for {_platform()} (recorded: {sorted(data)})")
    assert fingerprints(WORLDS[name]) == data[_platform()][name]


def _fingerprint_before_everyday_activity(world) -> str:  # noqa: ANN001
    """A frozen copy of `store.snapshot.world_fingerprint` as it was before per-person arrays existed."""
    h = hashlib.sha256()
    for prefix, (attr, _spec) in TABLES.items():
        for name, arr in sorted(getattr(world, attr).to_arrays(prefix).items()):
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
    for name, arr in sorted(world.beliefs.to_arrays(world.pop.n).items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.ascontiguousarray(world.ptrust.alpha[: world.pop.n]).tobytes())
    h.update(np.ascontiguousarray(world.ptrust.beta[: world.pop.n]).tobytes())
    for a in (world.product_price, world.product_available, world.product_price_usd):
        h.update(np.ascontiguousarray(a).tobytes())
    meta = {
        "day": world.day,
        "seed": world.sim_seed,
        "counters": world.counters,
        "scheduled": world.scheduled,
        "macro": world.macro,
        "notables": world.notables,
        "products": world.product_names,
    }
    h.update(json.dumps(json.loads(json.dumps(meta, default=str)), sort_keys=True).encode())
    return h.hexdigest()


def test_the_fingerprint_of_an_unswitched_world_is_unchanged():
    w = _with_market()
    run_in_memory(w, 3)
    assert world_fingerprint(w) == _fingerprint_before_everyday_activity(w)
