"""Check the backtest events' S&P 500 moves and 2-year yield changes against FRED.

    uv run python -I -X utf8 scripts/check_market_actuals.py [--events FILE.yaml ...]
        [--sp500 FILE.csv --dgs2 FILE.csv]
    uv run python -I -X utf8 scripts/check_market_actuals.py --beta [--sp500 FILE.csv --nasdaq100 FILE.csv]

Downloads FRED's SP500 and DGS2 daily series (or reads saved CSVs), recomputes each event's close-to-close
S&P 500 change and 2-year yield change, and reports any difference from the events file. Also prints the
standard deviation of daily S&P 500 changes since 2023 (the benchmark's "typical daily move").
Gold and dollar-index closes have no free official series and are not checked here.

With --beta it instead computes the slope of Nasdaq-100 daily percent changes on S&P 500 daily percent changes
since 2023 (OLS with an intercept), the coefficient of Jahan's derived US tech index, and prints a block to
paste into data/global/investors.yaml. FRED marks NASDAQ100 "Copyrighted: Pre-Approval Required": the series
is read in memory and never written to disk; only the coefficient is kept.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2022-12-01"


def series(name: str, path: str | None) -> list[tuple[str, float]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        req = urllib.request.Request(FRED.format(series=name), headers={"User-Agent": "jahan-backtest-check"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URL
            text = r.read().decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text)) if r][1:]
    return [(d, float(v)) for d, v in rows if v not in ("", ".")]


def daily_pct(rows: list[tuple[str, float]], since: str) -> dict[str, float]:
    """Percent change from the previous observation, for days on or after ``since``."""
    return {d: 100 * (b / a - 1) for (_, a), (d, b) in zip(rows, rows[1:], strict=False) if d >= since}


def beta(sp: list[tuple[str, float]], ndx: list[tuple[str, float]], since: str = "2023-01-03") -> dict[str, object]:
    x_all, y_all = daily_pct(sp, since), daily_pct(ndx, since)
    days = sorted(set(x_all) & set(y_all))
    x = [x_all[d] for d in days]
    y = [y_all[d] for d in days]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    return {
        "beta": round(slope, 3),
        "r2": round(sxy * sxy / (sxx * syy), 3),
        "n": len(days),
        "range": f"{days[0]}..{days[-1]}",
    }


def print_beta(sp_path: str | None, ndx_path: str | None) -> int:
    try:
        sp = series("SP500", sp_path)
        ndx = series("NASDAQ100", ndx_path)
    except OSError as e:
        print(f"Could not download from FRED ({e}); save the CSVs by hand and pass --sp500 and --nasdaq100.")
        return 2
    b = beta(sp, ndx)
    print(f"Nasdaq-100 on S&P 500, daily % changes: beta {b['beta']}, R2 {b['r2']}, N {b['n']}, {b['range']}")
    print("Paste under derived.us_tech_stocks in data/global/investors.yaml:")
    print(f"    beta: {b['beta']}\n    r2: {b['r2']}\n    n: {b['n']}\n    range: {b['range']}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", help="default: every data/backtests/events_*.yaml")
    ap.add_argument("--sp500")
    ap.add_argument("--dgs2")
    ap.add_argument("--nasdaq100")
    ap.add_argument("--beta", action="store_true", help="compute the derived US tech index coefficient")
    args = ap.parse_args(argv)
    if args.beta:
        return print_beta(args.sp500, args.nasdaq100)
    try:
        sp = series("SP500", args.sp500)
        two = series("DGS2", args.dgs2)
    except OSError as e:
        print(f"Could not download from FRED ({e}); save the CSVs by hand and pass --sp500 and --dgs2.")
        return 2
    files = (
        [Path(p) for p in args.events] if args.events else sorted((ROOT / "data" / "backtests").glob("events_*.yaml"))
    )
    events = [ev for f in files for ev in yaml.safe_load(f.read_text(encoding="utf-8"))["events"]]
    sp_i = {d: i for i, (d, _) in enumerate(sp)}
    two_i = {d: i for i, (d, _) in enumerate(two)}
    bad = 0
    for ev in events:
        day = str(ev["date"])
        if day in sp_i and sp_i[day] > 0 and ev["actual"].get("us_stocks") is not None:
            (_, a), (_, b) = sp[sp_i[day] - 1], sp[sp_i[day]]
            pct = 100 * (b / a - 1)
            ok = abs(pct - ev["actual"]["us_stocks"]) < 0.015
            bad += not ok
            mark = "" if ok else "  MISMATCH"
            print(f"{day} {ev['id']}: S&P 500 {pct:+.2f}% (file {ev['actual']['us_stocks']:+.2f}%){mark}")
        if ev.get("two_year_bp") is not None and day in two_i and two_i[day] > 0:
            (_, a), (_, b) = two[two_i[day] - 1], two[two_i[day]]
            bp = round(100 * (b - a))
            ok = bp == ev["two_year_bp"]
            bad += not ok
            print(
                f"{day} {ev['id']}: 2-year yield {bp:+d} bp (file {ev['two_year_bp']:+g}){'' if ok else '  MISMATCH'}"
            )
    since = [(d, v) for d, v in sp if d >= "2023-01-01" or d == max(x for x, _ in sp if x < "2023-01-01")]
    moves = [100 * (b / a - 1) for (_, a), (_, b) in zip(since, since[1:], strict=False)]
    print(f"S&P 500 daily s.d. since 2023: {statistics.stdev(moves):.2f}% over {len(moves)} days")
    print("All checked values match." if bad == 0 else f"{bad} mismatches.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
