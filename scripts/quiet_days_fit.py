"""Fit the ordinary-day movement of everyday activity version 2 (investors.yaml other_investors.quiet_days).

    uv run python -I -X utf8 scripts/quiet_days_fit.py [--write] [--sp500 FILE.csv --dexuseu FILE.csv]

Real days 2023-01-01..2025-12-31 without a Fed decision or jobs report (dates from data/backtests/events_*.yaml):
FRED SP500 and DEXUSEU, and Yahoo Finance GC=F (gold futures; contract rolls may inflate its volatility). The S&P
500's demeaned daily percent moves get a GARCH(1,1) variance with Student-t shocks by maximum likelihood (event
days are dropped and the rest treated as consecutive, an approximation); gold and the euro's dollar rate get their
own standard deviations and their correlations with the S&P 500's standardised shocks. Only coefficients are kept
(prices are read in memory and never stored). Also prints the real statistics B10 compares against. --write puts
the record under other_investors.quiet in data/global/investors.yaml.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
START, END = "2023-01-01", "2025-12-31"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd={a}&coed={b}"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?period1={a}&period2={b}&interval=1d"
UA = {"User-Agent": "jahan-benchmark"}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URLs
        return r.read().decode("utf-8")


def fred(series: str, path: str | None = None) -> dict[str, float]:
    text = Path(path).read_text(encoding="utf-8") if path else _get(FRED.format(s=series, a=START, b=END))
    rows = [r for r in csv.reader(io.StringIO(text)) if r][1:]
    return {d: float(v) for d, v in rows if v not in ("", ".")}


def yahoo_gold() -> dict[str, float]:
    a = int(dt.datetime(2022, 12, 30, tzinfo=dt.UTC).timestamp())
    b = int(dt.datetime(2026, 1, 2, tzinfo=dt.UTC).timestamp())
    res = json.loads(_get(YAHOO.format(a=a, b=b)))["chart"]["result"][0]
    out = {}
    for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"], strict=True):
        if c is not None:
            out[dt.datetime.fromtimestamp(t, dt.UTC).date().isoformat()] = float(c)
    return out


def event_days() -> set[str]:
    days = set()
    for f in (ROOT / "data" / "backtests").glob("events_*.yaml"):
        for ev in yaml.safe_load(f.read_text(encoding="utf-8")).get("events", []):
            if ev.get("announcement"):
                days.add(str(ev["date"]))
    return days


def returns(levels: dict[str, float]) -> dict[str, float]:
    ds = sorted(levels)
    return {d1: 100.0 * (levels[d1] / levels[d0] - 1.0) for d0, d1 in zip(ds, ds[1:], strict=False)}


def _params(p: np.ndarray) -> tuple[float, float, float, float]:
    e1, e2 = math.exp(p[1]), math.exp(p[2])
    return math.exp(p[0]), e1 / (1 + e1 + e2), e2 / (1 + e1 + e2), 2.1 + math.exp(p[3])


def garch_t_negll(p: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta, nu = _params(p)
    h = float(np.var(r))
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    ll = 0.0
    for x in r:
        ll += c - 0.5 * math.log(h) - (nu + 1) / 2 * math.log(1 + x * x / (h * (nu - 2)))
        h = omega + alpha * x * x + beta * h
    return -ll


def nelder_mead(f, x0: np.ndarray, iters: int = 4000, tol: float = 1e-10) -> np.ndarray:
    n = x0.size
    pts = [x0] + [x0 + np.eye(n)[i] * 0.5 for i in range(n)]
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centre = np.mean(pts[:-1], axis=0)
        refl = centre + (centre - pts[-1])
        fr = f(refl)
        if fr < vals[0]:
            exp_ = centre + 2 * (centre - pts[-1])
            fe = f(exp_)
            pts[-1], vals[-1] = (exp_, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = centre + 0.5 * (pts[-1] - centre)
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                pts = [pts[0] + 0.5 * (x - pts[0]) for x in pts]
                vals = [f(x) for x in pts]
    return pts[int(np.argmin(vals))]


def stats(moves: list[float], levels_moves: list[float]) -> dict[str, float]:
    a = np.abs(np.asarray(moves))
    lv = np.cumprod(1 + np.asarray(levels_moves) / 100.0)
    m21 = np.abs(lv[21:] / lv[:-21] - 1.0) * 100.0
    return {
        "mean_abs_daily_pct": round(float(a.mean()), 4),
        "p95_abs_daily_pct": round(float(np.percentile(a, 95)), 4),
        "median_abs_21_day_pct": round(float(np.median(m21)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sp500", help="a FRED SP500 CSV already downloaded (when fredgraph refuses this machine)")
    ap.add_argument("--dexuseu", help="a FRED DEXUSEU CSV already downloaded")
    args = ap.parse_args()
    skip = event_days()
    sp, eur = returns(fred("SP500", args.sp500)), returns(fred("DEXUSEU", args.dexuseu))
    gold = returns(yahoo_gold())
    days = [d for d in sorted(sp) if START <= d <= END and d not in skip and d in eur and d in gold]
    r = np.array([sp[d] for d in days])
    mean = float(r.mean())
    r = r - mean
    best = nelder_mead(lambda p: garch_t_negll(p, r), np.array([math.log(0.05), 0.0, 2.0, math.log(4.0)]))
    omega, alpha, beta, nu_hat = _params(best)
    nu = max(5, round(nu_hat))  # an integer for the simulation's chi-square draw; at least 5 (finite kurtosis)
    var = omega / (1 - alpha - beta)
    h, hs = float(np.var(r)), []
    for x in r:
        hs.append(h)
        h = omega + alpha * x * x + beta * h
    ratio = np.sqrt(np.asarray(hs) / var)
    g = np.array([gold[d] for d in days])
    e = np.array([eur[d] for d in days])
    g, e = g - g.mean(), e - e.mean()
    gsd, esd = float(g.std()), float(e.std())
    z = np.vstack([r / np.sqrt(hs), g / (gsd * ratio), e / (esd * ratio)])
    corr = np.corrcoef(z)
    chol = np.linalg.cholesky(corr)
    all_sp = [sp[d] for d in sorted(sp) if START <= d <= END and d not in skip]
    record = {
        "sp": {
            "omega": round(omega, 6),
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "var": round(var, 6),
            "mean_removed_pct": round(mean, 6),
        },
        "nu": int(nu),
        "nu_fitted": round(nu_hat, 3),
        "gold_sd": round(gsd, 6),
        "eur_sd": round(esd, 6),
        "corr": [[round(float(x), 4) for x in row] for row in corr],
        "chol": [[round(float(x), 6) for x in row] for row in chol],
        "fit": {
            "date": dt.date.today().isoformat(),
            "command": "uv run python -I -X utf8 scripts/quiet_days_fit.py --write",
            "range": f"{days[0]}..{days[-1]}",
            "days": len(days),
            "event_days_dropped": sum(1 for d in sp if START <= d <= END and d in skip),
            "sources": [
                "FRED SP500 (S&P Dow Jones Indices)",
                "FRED DEXUSEU (Federal Reserve H.10)",
                "Yahoo Finance GC=F (COMEX gold futures)",
            ],
            "neg_log_likelihood": round(garch_t_negll(best, r), 3),
            "note": (
                "event days are dropped and the rest treated as consecutive; gold futures roll between contracts; "
                "prices were read in memory only"
            ),
        },
        "real_b10": {
            "us_stocks": stats([sp[d] for d in days], all_sp),
            "gold": stats([gold[d] for d in days], [gold[d] for d in sorted(gold) if START <= d <= END]),
            "euro": stats([eur[d] for d in days], [eur[d] for d in sorted(eur) if START <= d <= END]),
        },
    }
    print(yaml.safe_dump({"quiet": record}, sort_keys=False))
    if args.write:
        p = ROOT / "data" / "global" / "investors.yaml"
        text = p.read_text(encoding="utf-8")
        marker = "  quiet: null  # pending: scripts/quiet_days_fit.py\n"
        if marker not in text:
            print("investors.yaml has no pending quiet entry; not written", file=sys.stderr)
            return 1
        block = yaml.safe_dump({"quiet": record}, sort_keys=False, width=110)
        text = text.replace(marker, "".join("  " + line + "\n" for line in block.splitlines()))
        p.write_text(text, encoding="utf-8", newline="\n")
        print("written to data/global/investors.yaml", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
