"""`jahan benchmark` pieces on canned data (the real runs are slow and read FRED and Polymarket)."""

import datetime as dt
import json

import pytest

from jahan.ingest.polymarket import price_history
from jahan.llm.recordings import Recordings
from jahan.scenario import benchmark as bm
from jahan.scenario.backtest import BacktestEvent, Moves
from jahan.scenario.spec import Announcement


def _event(i, day):
    ann = Announcement(indicator="us_payrolls", actual=200, expected=150, headline="Jobs beat", valence=0.3)
    return BacktestEvent(id=f"jobs-{i}", date=day, title="Jobs", announcement=ann, actual=Moves(us_stocks=0.5))


def _basis(est0, people, real):
    return bm.Basis(_event(0, dt.date(2024, 1, 5)), est0, people, real, False)


def test_score_is_linear_in_kappa_and_compares_with_and_without_people():
    rows = [_basis(0.2, [1.0, 0.5, 0.0, -0.5, 0.0], [0.4, 0.3, 0.0, -0.2, 0.1])]
    zero = bm.b1_score(rows, 0.0)
    assert zero["five_day"] == pytest.approx(zero["five_day_without_people"])
    assert zero["day0"] == pytest.approx(abs(0.2 - 0.4) / bm.SD)
    some = bm.b1_score(rows, 0.2)
    sim = [100 * ((1 + 0.002) * (1 + 0.002) - 1), 0.1, 0.0, -0.1, 0.0]  # day 0 combines estimate and people
    expect = sum(abs(s - r) for s, r in zip(sim, [0.4, 0.3, 0.0, -0.2, 0.1], strict=True)) / 5 / bm.SD
    assert some["five_day"] == pytest.approx(expect)
    # a basis run at kappa 2 scores the same as one at kappa 1 with double the people's part
    doubled = [_basis(0.2, [2 * p for p in rows[0].people], rows[0].real)]
    assert bm.b1_score(doubled, 0.2, 2.0)["five_day"] == pytest.approx(some["five_day"])


def test_optimum_prefers_the_smallest_kappa_on_ties():
    rows = [_basis(0.0, [0.0, 0.0], [0.3, -0.1])]  # people never move: every kappa scores the same
    k, losses = bm.optimum(rows)
    assert k == 0.0 and len(losses) == len(bm.GRID)
    rows = [_basis(0.0, [1.0, 1.0], [0.1, 0.1])]  # the best kappa is 0.1
    assert bm.optimum(rows)[0] == pytest.approx(0.1, abs=1e-3)


def test_windows_stop_before_the_next_event():
    a, b = _event(1, dt.date(2024, 1, 5)), _event(2, dt.date(2024, 1, 10))
    actuals = {
        "b1_windows": {
            "jobs-1": [{"date": f"2024-01-{d:02d}", "us_stocks": 0.1} for d in (8, 9, 10, 11)],
            "jobs-2": [{"date": f"2024-01-{d:02d}", "us_stocks": 0.2} for d in (11, 12, 16, 17)],
        }
    }
    win = bm._windows([a, b], actuals)
    assert win["jobs-1"][0] == [dt.date(2024, 1, 8), dt.date(2024, 1, 9)] and win["jobs-1"][2] is True
    assert len(win["jobs-2"][0]) == 4 and win["jobs-2"][2] is False


def test_price_history_keeps_one_price_a_day(tmp_path):
    day = 86400
    body = json.dumps({"history": [{"t": 1767225600 + k * day // 2, "p": 0.5 + 0.01 * k} for k in range(6)]}).encode()
    rows = price_history("123", Recordings(tmp_path / "rec"), "2026-10-17", fetcher=lambda _url: body)
    assert [d for d, _ in rows] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert rows[0][1] == pytest.approx(0.51)  # the last price of a day wins


def test_odds_case_runs_a_quiet_world():
    start, end = dt.date(2026, 7, 1), dt.date(2026, 8, 31)
    left0 = (end - start).days + 1
    # real prices on the path a constant chance per day implies: a quiet stretch where nothing happened
    prices = [((start + dt.timedelta(days=k)).isoformat(), 0.6 ** ((left0 - k) / left0)) for k in range(40)]
    case = bm.OddsCase(
        "us-x-iran-ceasefire-test",
        "US x Iran ceasefire continues through August 31?",
        "holds_through",
        "2026-08-31",
        prices,
        False,
    )
    r = bm.odds_case_run(case, talk=False)
    assert r is not None and r["days"] == 39 and r["kind"] == "holds_through"
    assert r["sim"] < r["flat"]  # on such a path, time passing beats a flat price
    assert (
        bm.odds_case_run(
            bm.OddsCase("x-y-z", "Will it happen by May?", "by_date", "2026-05-31", [("2026-04-01", 0.99)] * 31, True),
            False,
        )
        is None
    )


def test_registration_is_known_and_report_renders():
    rec = bm.registration_record()
    assert rec["registered_in"] and rec["changed_since"] is False
    text = bm.render_markdown(
        {
            "date": "2026-10-17",
            "command": "jahan benchmark",
            "registered_in": rec["registered_in"],
            "checks": {"B3": {"summary": "flat", "passed": True}},
        }
    )
    assert "| B3 | flat | yes |" in text and "B7" in text and "No benchmark exists outside the US" in text
