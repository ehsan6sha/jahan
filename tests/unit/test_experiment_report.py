import pyarrow.parquet as pq

from jahan.model.enums import EventKind
from jahan.store.eventlog import EVENT_SCHEMA, read_all
from tests.conftest import SCENARIO, cli


def _adoptions(table):
    d = table.to_pydict()
    return sorted((day, a) for day, k, a in zip(d["day"], d["kind"], d["agent"], strict=True) if k == EventKind.ADOPTED)


def test_seed_zero_replays_the_main_timeline(tmp_path):
    run = tmp_path / "main"
    cli("gen", "--run", str(run), "--agents", "700", "--seed", "8", "--start-date", "2026-10-13")
    cli("run", "--run", str(run), "--days", "5")
    cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes")
    cli("run", "--run", str(run), "--days", "60")
    cli("experiment", "--run", str(run), "--prompt", "1", "--seeds", "2", "--horizon", "60", "--workers", "1")
    main = _adoptions(read_all(run / "events", "events", EVENT_SCHEMA))
    seed0 = _adoptions(pq.read_table(run / "experiments" / "0001" / "seed_00" / "treated" / "events.parquet"))
    seed1 = _adoptions(pq.read_table(run / "experiments" / "0001" / "seed_01" / "treated" / "events.parquet"))
    assert main and main == seed0
    assert seed1 != seed0


def test_end_to_end_experiment_and_report(tmp_path):
    run = tmp_path / "e2e"
    cli("gen", "--run", str(run), "--agents", "600", "--seed", "5", "--start-date", "2026-10-13")
    cli("run", "--run", str(run), "--days", "10")
    cli("inject", "--run", str(run), "--scenario", str(SCENARIO), "--yes")
    cli("experiment", "--run", str(run), "--prompt", "1", "--seeds", "2", "--horizon", "40", "--workers", "1")
    out = cli("report", "--run", str(run), "--prompt", "1")
    report = (run / "reports" / "0001" / "report.md").read_text(encoding="utf-8")
    assert "Report:" in out
    for heading in (
        "# Scenario report",
        "## Smart Ring X: adoption and awareness",
        "### Who adopted Smart Ring X",
        "### Trust (Smart Ring X)",
        "### Sampled stories",
        "## How first exposures arrived",
        "## Caveats",
    ):
        assert heading in report, heading
    table = report.split("## How first exposures arrived", 1)[1].split("##", 1)[0]
    assert "| Smart Ring X |" in table and "people" in table  # word of mouth next to the channel kinds
    assert (run / "reports" / "0001" / "curves.csv").exists()
    listing = cli("agents", "--run", str(run), "--filter", "age > 30", "--limit", "3")
    assert "agents match" in listing
    card = cli("agent", "--run", str(run), "5")
    assert "agent 5" in card
