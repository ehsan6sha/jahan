"""Real data for `jahan benchmark` (data/backtests/benchmarks.yaml), from FRED.

    uv run python -I -X utf8 scripts/benchmark_actuals.py [--sp500 FILE.csv --unrate FILE.csv --umcsent FILE.csv]

Writes data/backtests/benchmark_actuals.yaml:
- B1 windows: for every backtest event with a real S&P 500 move, the S&P 500's percent moves on the next 4 FRED
  observation dates after the release date (as the events files keep the release day's move);
- B2: the S&P 500's daily moves from the last close before the US and Israeli strikes on Iran (2026-02-27) to the
  20th trading day after 2026-03-02, its largest one-day fall and its largest fall within 20 trading days;
- B6: only the fitted coefficient of 12-month changes in the University of Michigan sentiment index (UMCSENT) on
  12-month changes in the unemployment rate (UNRATE), monthly 1978-2025, and the standard deviation of the
  sentiment changes. UMCSENT is copyrighted by the University of Michigan: it is read in memory and never stored.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import statistics
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
OUT = ROOT / "data" / "backtests" / "benchmark_actuals.yaml"
WINDOW = 4
B2_BEFORE, B2_FIRST, B2_DAYS = "2026-02-27", "2026-03-02", 20


def series(name: str, path: str | None, start: str) -> list[tuple[str, float]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        url = FRED.format(series=name, start=start)
        req = urllib.request.Request(url, headers={"User-Agent": "jahan-benchmark"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URL
            text = r.read().decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text)) if r][1:]
    return [(d, float(v)) for d, v in rows if v not in ("", ".")]


def windows(sp: list[tuple[str, float]], events: list[dict]) -> dict[str, list[dict]]:
    idx = {d: i for i, (d, _) in enumerate(sp)}
    out: dict[str, list[dict]] = {}
    for ev in events:
        day = str(ev["date"])
        if day not in idx or ev.get("actual", {}).get("us_stocks") is None:
            continue
        i = idx[day]
        rows = []
        for k in range(1, WINDOW + 1):
            if i + k >= len(sp):
                break
            (_, a), (d, b) = sp[i + k - 1], sp[i + k]
            rows.append({"date": d, "us_stocks": round(100 * (b / a - 1), 4)})
        out[str(ev["id"])] = rows
    return out


def iran_window(sp: list[tuple[str, float]]) -> dict:
    days = [d for d, _ in sp]
    i0 = days.index(B2_BEFORE)
    i1 = days.index(B2_FIRST)
    rows = sp[i0 : i1 + B2_DAYS]  # the last close before, then the first 20 trading days from 2026-03-02
    moves = [
        {"date": d, "us_stocks": round(100 * (b / a - 1), 4)} for (_, a), (d, b) in zip(rows, rows[1:], strict=False)
    ]
    levels = [v for _, v in rows]
    worst = 0.0
    for a in range(len(levels)):
        for b in range(a + 1, min(len(levels), a + B2_DAYS + 1)):
            worst = min(worst, 100 * (levels[b] / levels[a] - 1))
    return {
        "from": rows[0][0],
        "to": rows[-1][0],
        "moves": moves,
        "largest_daily_fall_pct": round(min(m["us_stocks"] for m in moves), 4),
        "largest_20_day_fall_pct": round(worst, 4),
    }


def sentiment_slope(unrate: list[tuple[str, float]], umc: list[tuple[str, float]]) -> dict:
    u = {d: v for d, v in unrate if "1978-01-01" <= d <= "2025-12-01"}
    s = {d: v for d, v in umc if "1978-01-01" <= d <= "2025-12-01"}
    months = sorted(set(u) & set(s))

    def back12(d: str) -> str:
        y, m, _ = (int(x) for x in d.split("-"))
        return dt.date(y - 1, m, 1).isoformat()

    pairs = [(u[d] - u[back12(d)], s[d] - s[back12(d)]) for d in months if back12(d) in u and back12(d) in s]
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    slope = sum((a - mx) * (b - my) for a, b in pairs) / sum((a - mx) ** 2 for a in x)
    sd = statistics.stdev(y)
    level = [s[d] for d in months]
    return {
        "months": len(pairs),
        "range": f"{months[0]}..{months[-1]}",
        "index_min": round(min(level), 2),
        "index_max": round(max(level), 2),
        "slope_share_of_range": round(slope / (max(level) - min(level)), 5),
        "slope_points_per_unemployment_point": round(slope, 3),
        "sd_of_12_month_changes": round(sd, 3),
        "slope_in_sd": round(slope / sd, 3),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500")
    ap.add_argument("--unrate")
    ap.add_argument("--umcsent")
    args = ap.parse_args(argv)
    try:
        sp = series("SP500", args.sp500, "2022-12-01")
        unrate = series("UNRATE", args.unrate, "1977-01-01")
        umc = series("UMCSENT", args.umcsent, "1977-01-01")
    except OSError as e:
        print(f"Could not download from FRED ({e}); save the CSVs by hand and pass them.")
        return 2
    files = sorted((ROOT / "data" / "backtests").glob("events_*.yaml"))
    events = [ev for f in files for ev in yaml.safe_load(f.read_text(encoding="utf-8"))["events"]]
    data = {
        "retrieved": dt.date.today().isoformat(),
        "source": "FRED (SP500, UNRATE, UMCSENT); only percent moves and fitted coefficients are kept",
        "b1_windows": windows(sp, events),
        "b2_iran_2026": iran_window(sp),
        "b6_sentiment_unemployment": sentiment_slope(unrate, umc),
    }
    header = (
        "# Real data for `jahan benchmark` (data/backtests/benchmarks.yaml), written by\n"
        "# scripts/benchmark_actuals.py from FRED. B1: S&P 500 moves on the 4 FRED dates after each release; B2:\n"
        "# the S&P 500 around the 2026 US and Israeli strikes on Iran; B6: only a fitted coefficient (UMCSENT is\n"
        "# copyrighted by the University of Michigan and never stored).\n"
    )
    OUT.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    b2, b6 = data["b2_iran_2026"], data["b6_sentiment_unemployment"]
    print(f"B1: windows for {len(data['b1_windows'])} events")
    print(
        f"B2: {b2['from']}..{b2['to']}: largest daily fall {b2['largest_daily_fall_pct']}%, "
        f"20-day {b2['largest_20_day_fall_pct']}%"
    )
    print(f"B6: {b6}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
