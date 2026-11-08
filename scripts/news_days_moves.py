"""Real moves for B9 (data/backtests/news_days.yaml), from FRED.

    uv run python -I -X utf8 scripts/news_days_moves.py --sp500 FILE.csv [FILE2.csv ...] [--nasdaq100 FILE.csv]

Fills each day's `actual.us_stocks`: the S&P 500's percent move on the trading day (against the previous FRED
observation) and on the next 4 FRED observation dates (fewer when they do not exist yet; null when FRED has no
value for the trading day). The files are FRED CSV downloads (fredgraph refuses some machines; several files are
merged). With --nasdaq100 the same moves of the Nasdaq-100 go to runs/_benchmark/news_days_tech.yaml (git-ignored:
FRED marks that series "Copyrighted: Pre-Approval Required", so it is never committed).
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DAYS = ROOT / "data" / "backtests" / "news_days.yaml"
TECH = ROOT / "runs" / "_benchmark" / "news_days_tech.yaml"
WINDOW = 4


def load(paths: list[str]) -> list[tuple[str, float]]:
    out: dict[str, float] = {}
    for p in paths:
        rows = [r for r in csv.reader(io.StringIO(Path(p).read_text(encoding="utf-8"))) if r][1:]
        out.update({d: float(v) for d, v in rows if v not in ("", ".")})
    return sorted(out.items())


def moves(series: list[tuple[str, float]], day: str) -> list[float] | None:
    idx = {d: i for i, (d, _) in enumerate(series)}
    if day not in idx or idx[day] == 0:
        return None
    i = idx[day]
    return [
        round(100.0 * (series[j][1] / series[j - 1][1] - 1.0), 4) for j in range(i, min(i + WINDOW + 1, len(series)))
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500", nargs="+", required=True)
    ap.add_argument("--nasdaq100", nargs="*")
    args = ap.parse_args()
    text = DAYS.read_text(encoding="utf-8")
    head, _, body = text.partition("\nregistered_in:")
    doc = yaml.safe_load("registered_in:" + body)
    sp = load(args.sp500)
    missing = []
    for day in doc["days"]:
        m = moves(sp, day["trading_day"])
        day["actual"] = {"us_stocks": m}
        if m is None:
            missing.append(day["trading_day"])
    out = head + "\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=115)
    DAYS.write_text(out, encoding="utf-8", newline="\n")
    print(f"{len(doc['days'])} days; no FRED value for: {', '.join(missing) or 'none'}")
    if args.nasdaq100:
        ndx = load(args.nasdaq100)
        TECH.parent.mkdir(parents=True, exist_ok=True)
        tech = {d["trading_day"]: moves(ndx, d["trading_day"]) for d in doc["days"]}
        TECH.write_text(yaml.safe_dump({"source": "FRED NASDAQ100", "moves": tech}, sort_keys=True), encoding="utf-8")
        print(f"tech moves written to {TECH.relative_to(ROOT)} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
