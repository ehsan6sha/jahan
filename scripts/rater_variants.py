"""B11: do other ways of rating the news beat the current one? (data/backtests/benchmarks.yaml, check B11)

    uv run python -X utf8 scripts/rater_variants.py closes           # S&P 500 closes to runs/_benchmark (git-ignored)
    uv run python -X utf8 scripts/rater_variants.py choose --rate    # rate the 2023 to June 2026 days; pick one
    uv run python -X utf8 scripts/rater_variants.py clean --rate     # rate the clean days; score them

Ratings are recorded under runs/_benchmark/recordings; without --rate the recordings are replayed. The variants' texts
are the registered ones and must not change after the first call.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jahan.config import RunConfig
from jahan.llm.provider import LLMClient, make_client
from jahan.scenario import benchmark_news as bn
from jahan.scenario.parser import parse_prompt

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "runs" / "_benchmark"
CLOSES = BENCH / "sp500_closes.csv"
EXTRA = ROOT / "data" / "backtests" / "news_days_extra.yaml"
IT = 0.379
SD = bn.SD
S_CURRENT = 1.43

MATERIAL = (
    "How to rate market_valence here: it measures how much the news changes what the whole US stock market is worth: "
    "the profits of companies overall, interest rates, or the availability of credit, compared with what investors "
    "already expected. Most news does not change these; leave market_valence at 0 unless the story clearly does. Bad "
    "news for people is not by itself bad news for shares: wars, attacks, political fights and elections are 0 unless "
    "they change oil supply, trade rules, taxes, interest rates or credit. Reserve 0.7 to 1, up or down, for events "
    "that change profits or rates across the whole economy, and use that range when they do. A story about only one "
    "part of the market is rated for that part (sectors)."
)
KNEW = "What investors already knew before this day (headlines of the previous 30 days, oldest first):"
AGAINST = (
    "Rate today's news only for what it changes against that: a further step in a story already under way counts only "
    "for the part investors had not expected."
)
VARIANTS = ("B_material", "C_context", "D_context_market", "E_material_context")


def fetch_closes() -> None:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=1640995200&period2=1794268800&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310 (fixed https endpoint)
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        js = json.load(r)["chart"]["result"][0]
    rows = [
        (dt.datetime.fromtimestamp(t, dt.UTC).date().isoformat(), c)
        for t, c in zip(js["timestamp"], js["indicators"]["quote"][0]["close"], strict=True)
        if c is not None
    ]
    BENCH.mkdir(parents=True, exist_ok=True)
    with CLOSES.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"{len(rows)} closes, {rows[0][0]} to {rows[-1][0]}")


def closes() -> tuple[list[dt.date], np.ndarray]:
    with CLOSES.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [dt.date.fromisoformat(r[0]) for r in rows], np.array([float(r[1]) for r in rows])


def load_extra() -> list[bn.NewsDay]:
    if not EXTRA.exists():
        return []
    ts, cl = closes()
    idx = {t: i for i, t in enumerate(ts)}
    out = []
    for d in yaml.safe_load(EXTRA.read_text(encoding="utf-8"))["days"]:
        day = dt.date.fromisoformat(str(d["trading_day"]))
        if day not in idx:  # a trading day with no close yet
            continue
        i = idx[day]
        real = [round(100.0 * (cl[j] / cl[j - 1] - 1.0), 4) for j in range(i, min(i + 5, len(cl)))]
        out.append(bn.NewsDay(str(d["id"]), day, "fresh", ["news"], [str(e["headline"]) for e in d["events"]], real))
    return out


def context_block(d: bn.NewsDay, everything: list[bn.NewsDay], shift_days: int = 0) -> str:
    end = d.day - dt.timedelta(days=shift_days)
    start = end - dt.timedelta(days=30)
    lines = [
        f"- {o.day.isoformat()}: {h}"
        for o in sorted(everything, key=lambda x: x.day)
        if start <= o.day < end
        for h in o.headlines
    ]
    return KNEW + "\n" + ("\n".join(lines) if lines else "none on file") + "\n" + AGAINST


def market_line(day: dt.date, ts: list[dt.date], cl: np.ndarray) -> str:
    i = {t: k for k, t in enumerate(ts)}[day]
    ret = lambda n: 100.0 * (cl[i - 1] / cl[i - 1 - n] - 1.0)  # noqa: E731
    dd = 100.0 * (1.0 - cl[i - 1] / cl[max(0, i - 252) : i].max())
    return (
        f"The S&P 500 before this day: previous day {ret(1):+.2f}%, last 5 trading days {ret(5):+.2f}%, "
        f"last 20 trading days {ret(20):+.2f}%, {dd:.1f}% below its one-year high."
    )


def notes(variant: str, d: bn.NewsDay, everything: list[bn.NewsDay], ts: list[dt.date], cl: np.ndarray) -> str:
    if variant == "B_material":
        return MATERIAL
    if variant == "C_context":
        return context_block(d, everything)
    if variant == "D_context_market":
        return context_block(d, everything) + "\n" + market_line(d.day, ts, cl)
    if variant == "E_material_context":
        return MATERIAL + "\n" + context_block(d, everything)
    if variant.startswith("P_"):  # the placebo of a context variant: the window one year earlier
        base = variant[2:]
        blk = context_block(d, everything, shift_days=365)
        return (MATERIAL + "\n" + blk) if base == "E_material_context" else blk
    raise ValueError(variant)


def rate_all(
    client: LLMClient,
    days: list[bn.NewsDay],
    variant: str,
    everything: list[bn.NewsDay],
    ts: list[dt.date],
    cl: np.ndarray,
) -> dict[str, dict[str, Any]]:
    worlds = {d.id: bn._parse_world(d.day) for d in days}  # noqa: SLF001 (built one by one: the LLM calls run in threads)

    def one(d: bn.NewsDay) -> tuple[str, dict[str, Any]]:
        res = parse_prompt(
            client,
            worlds[d.id],
            bn.news_prompt(d),
            purpose_tag=f"b11:{variant}",
            rating_notes=notes(variant, d, everything, ts, cl),
        )
        return d.id, {
            "scenario": None if res.scenario is None else res.scenario.model_dump(mode="json"),
            "questions": res.questions,
            "error": res.error,
        }

    with cf.ThreadPoolExecutor(6) as ex:
        return dict(ex.map(one, days))


def clusters(rating: dict[str, Any]) -> list[tuple[float, str]]:
    """Each story cluster's mv x salience (strongest member) and sector, as the engine prices it."""
    evs, _ = bn.news_events(rating)
    cl: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for i, e in enumerate(evs):
        if e.market_valence is None:
            continue
        cl[e.story or f"i{i}"].append((float(e.market_valence) * float(e.salience), str(e.sectors)))
    return [max(v, key=lambda t: abs(t[0])) for v in cl.values()]


def others_at_s1(rating: dict[str, Any]) -> float:
    return float(sum(x * (IT if sec == "tech" else 1.0) for x, sec in clusters(rating)))


def period(d: bn.NewsDay) -> str:
    return "2026H1" if d.day.year == 2026 else str(d.day.year)


GRID = np.round(np.arange(0.0, 20.0 + 1e-9, 0.01), 2)


def fit(p: np.ndarray, o: np.ndarray, y: np.ndarray) -> float:
    loss = np.mean(np.abs(p[None, :] + GRID[:, None] * o[None, :] - y[None, :]), axis=1)
    return float(GRID[int(np.argmin(np.round(loss, 12)))])


def lopo(days: list[bn.NewsDay], p: np.ndarray, o: np.ndarray) -> tuple[float, float, dict[str, float]]:
    y = np.array([d.real[0] for d in days])
    per = np.array([period(d) for d in days])
    errs, by, ss = [], {}, {}
    for q in ("2023", "2024", "2025", "2026H1"):
        m = per != q
        s = fit(p[m], o[m], y[m])
        e = np.abs(p[~m] + s * o[~m] - y[~m])
        errs.append(e)
        by[q], ss[q] = round(float(e.mean() / SD), 3), s
    return float(np.mean(np.concatenate(errs)) / SD), fit(p, o, y), {**by, **{f"s_{k}": v for k, v in ss.items()}}


def describe(days: list[bn.NewsDay], o: np.ndarray, ratings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    y = np.array([d.real[0] for d in days])
    nz = np.abs(o) > 1e-9
    mv = [x for d in days for x, _ in clusters(ratings[d.id])]
    return {
        "rated_days": int(nz.sum()),
        "direction_right": int((np.sign(o[nz]) == np.sign(y[nz])).sum()),
        "negative_share": round(float(np.mean(np.array(mv) < 0)), 2) if mv else None,
        "max_abs_mv_x_salience": round(float(np.max(np.abs(mv))), 2) if mv else None,
        "unparsed": sum(1 for d in days if not ratings[d.id].get("scenario")),
        "corr_with_real": round(float(np.corrcoef(o[nz], y[nz])[0, 1]), 3) if nz.sum() > 2 else None,
    }


def load_ratings(variant: str) -> dict[str, dict[str, Any]]:
    return json.loads((BENCH / f"b11_{variant}.json").read_text(encoding="utf-8"))


def save_ratings(variant: str, new: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    p = BENCH / f"b11_{variant}.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(new)
    p.write_text(json.dumps(old), encoding="utf-8")
    return old


def choose(client: LLMClient) -> None:
    base = json.loads((BENCH / "b9_basis.json").read_text(encoding="utf-8"))
    people = {r["id"]: r["people"] for r in base["rows"]}
    a_ratings = {k: v["first"] for k, v in base["ratings"].items()}
    all_days = bn.load_news_days() + load_extra()
    days = [d for d in bn.load_news_days() if d.split in ("fit", "seen")]
    ts, cl = closes()
    p = np.array([people[d.id] for d in days])
    out: dict[str, Any] = {}
    oa = np.array([others_at_s1(a_ratings[d.id]) for d in days])
    score_a, s_a, by = lopo(days, p, oa)
    out["A_current"] = {"lopo": score_a, "s_all": s_a, **by, **describe(days, oa, a_ratings)}
    for v in VARIANTS:
        r = save_ratings(v, rate_all(client, days, v, all_days, ts, cl))
        o = np.array([others_at_s1(r[d.id]) for d in days])
        score, s_all, by = lopo(days, p, o)
        out[v] = {"lopo": score, "s_all": s_all, **by, **describe(days, o, r)}
        print(v, json.dumps(out[v]), flush=True)
    y = np.array([d.real[0] for d in days])
    out["no_change_all"] = round(float(np.mean(np.abs(y)) / SD), 4)
    out["people_only_all"] = round(float(np.mean(np.abs(p - y)) / SD), 4)
    best = min(VARIANTS, key=lambda v: out[v]["lopo"])
    out["chosen"] = best if out[best]["lopo"] <= out["A_current"]["lopo"] - 0.01 else "A_current"
    text = json.dumps(rounded(out, 4), indent=1)
    (BENCH / "b11_choose.json").write_text(text, encoding="utf-8")
    print(text)
    print("cost:", round(client.budget.used_usd, 3))


SEEDS = (0, 1, 2)
KAPPA = 0.048
BASIS = BENCH / "b11_basis.json"
WORKERS = 6


def world_basis(variant: str, days: list[bn.NewsDay], ratings: dict[str, dict[str, Any]], base: RunConfig) -> None:
    """Each day's full world (B9's config), saved after every day so an interrupted run continues."""
    done = json.loads(BASIS.read_text(encoding="utf-8")) if BASIS.exists() else {}
    todo = [d for d in days if f"{variant}:{d.id}" not in done]
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        jobs = {ex.submit(_one_day, d, ratings[d.id], base): d for d in todo}
        for job in cf.as_completed(jobs):
            d = jobs[job]
            people, others, tech = job.result()
            done[f"{variant}:{d.id}"] = {"people": people, "others": others, "tech": tech}
            BASIS.write_text(json.dumps(done), encoding="utf-8")
            print(f"{variant} {d.day}: people {people:+.3f} others at s=1 {others:+.3f}", flush=True)


def _one_day(d: bn.NewsDay, rating: dict[str, Any], base: RunConfig) -> tuple[float, float, float]:
    events, _ = bn.news_events(rating)
    return bn.day_basis(d, events, base, SEEDS, KAPPA)


def row(variant: str, d: bn.NewsDay) -> bn.DayBasis:
    v = json.loads(BASIS.read_text(encoding="utf-8"))[f"{variant}:{d.id}"]
    return bn.DayBasis(d, v["people"], v["others"], v["tech"])


def rounded(x: Any, digits: int) -> Any:
    """For the saved record only: every decision compares the unrounded scores."""
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, digits) for v in x]
    return x


def l1(sim: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(sim - y)) / SD)


def boot_diff(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> list[float]:
    """95% interval of mean(|a - y| - |b - y|) / SD over days (negative: a is better)."""
    rng = np.random.default_rng(0)
    d = (np.abs(a - y) - np.abs(b - y)) / SD
    picks = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(2000)]
    return [round(float(np.percentile(picks, 2.5)), 3), round(float(np.percentile(picks, 97.5)), 3)]


def clean(client: LLMClient, chosen: str) -> None:
    b9 = bn.load_news_days()
    extra = load_extra()
    everything = b9 + extra
    ts, cl = closes()
    holdout = [d for d in b9 if d.split == "holdout"]
    outcome = [d for d in b9 if d.split == "outcome"]
    fit_days = [d for d in b9 if d.split == "fit"]
    clean_days = holdout + extra
    saved = json.loads((BENCH / "b9_basis.json").read_text(encoding="utf-8"))
    a = {k: v["first"] for k, v in saved["ratings"].items()}
    a.update({d.id: bn.rate_day(client, d) for d in extra})
    save_ratings("A_current", {d.id: a[d.id] for d in extra})
    base = RunConfig.load(ROOT / "configs" / "m1.yaml")
    rows_a = {r["id"]: r for r in saved["rows"]}
    done = json.loads(BASIS.read_text(encoding="utf-8")) if BASIS.exists() else {}
    for d in holdout + outcome + fit_days:
        r = rows_a[d.id]
        done[f"A_current:{d.id}"] = {"people": r["people"], "others": r["others_at_s1"], "tech": r["tech_at_s1"]}
    BASIS.write_text(json.dumps(done), encoding="utf-8")
    world_basis("A_current", extra, a, base)

    variants = [chosen] if chosen != "A_current" else []
    exploratory = [
        v for v in ("C_context", "E_material_context") if v != chosen
    ]  # context can only help after the cutoff
    for v in variants + exploratory:
        r = save_ratings(v, rate_all(client, clean_days + outcome, v, everything, ts, cl))
        print(v, "rated", len(clean_days + outcome), "cost", round(client.budget.used_usd, 3), flush=True)
        if v in variants:  # the exploratory ones use the current rater's people's part (as the choosing step did)
            world_basis(v, clean_days + outcome, r, base)
    for v in variants:
        world_basis(v, fit_days, load_ratings(v), base)

    y_clean = np.array([d.real[0] for d in clean_days])
    fresh = np.array([d.split == "fresh" for d in clean_days])
    ra = [row("A_current", d) for d in clean_days]
    p_a, o_a = np.array([r.people for r in ra]), np.array([r.others for r in ra])
    sim_a, sim_b0 = p_a + S_CURRENT * o_a, p_a
    res: dict[str, Any] = {"clean_days": len(clean_days), "fresh_days": int(fresh.sum()), "chosen": chosen}

    def block(m: np.ndarray) -> dict[str, float]:
        return {
            "a_current_s143": l1(sim_a[m], y_clean[m]),
            "b_people_only": l1(sim_b0[m], y_clean[m]),
            "no_change": l1(0 * y_clean[m], y_clean[m]),
        }

    all_m = np.ones(len(clean_days), bool)
    res["all_clean"] = block(all_m)
    res["fresh_only"] = block(fresh)
    res["holdout_only"] = block(~fresh)
    decision_vs: dict[str, Any] = {}
    for v in variants:
        fr = [row(v, d) for d in fit_days]
        s, _ = bn.fit_s(fr)
        lo, hi = bn.bootstrap_s(fr)
        fs = bn.b9_scores(fr, s, 1.299)
        rc = [row(v, d) for d in clean_days]
        p_c, o_c = np.array([r.people for r in rc]), np.array([r.others for r in rc])
        sim_c = p_c + s * o_c
        res[v] = {
            "s_refit": s,
            "s_ci95": [lo, hi],
            "fit_score": fs["day0"],
            "fit_s0": fs["day0_s0"],
            "fit_no_change": fs["day0_no_change"],
            "all_clean": l1(sim_c[all_m], y_clean[all_m]),
            "fresh_only": l1(sim_c[fresh], y_clean[fresh]),
            "holdout_only": l1(sim_c[~fresh], y_clean[~fresh]),
            "people_only_all_clean": l1(p_c, y_clean),
            "direction": describe(clean_days, o_c, load_ratings(v)),
            "mean_sim_minus_real": round(float(np.mean(sim_c - y_clean)), 3),
            "diff_vs_b_ci95": boot_diff(sim_c, sim_b0, y_clean),
            "diff_vs_no_change_ci95": boot_diff(sim_c, 0 * y_clean, y_clean),
            "diff_vs_a_ci95": boot_diff(sim_c, sim_a, y_clean),
            "per_day": [
                {
                    "day": str(d.day),
                    "split": d.split,
                    "real": round(float(yy), 2),
                    "sim": round(float(ss), 2),
                    "a": round(float(aa), 2),
                }
                for d, yy, ss, aa in zip(clean_days, y_clean, sim_c, sim_a, strict=True)
            ],
        }
        decision_vs[v] = sim_c
        ro = [row(v, d) for d in outcome]
        res[v]["outcome_days_shown"] = [
            {"day": str(r.day.day), "real": round(r.day.real[0], 2), "sim": round(r.people + s * r.others, 2)}
            for r in ro
        ]
    ad = describe(clean_days, o_a, {**a})
    res["a_direction"] = ad
    res["a_mean_sim_minus_real"] = round(float(np.mean(sim_a - y_clean)), 3)
    choose_doc = json.loads((BENCH / "b11_choose.json").read_text(encoding="utf-8"))
    for v in exploratory:
        s = float(choose_doc[v]["s_all"])
        rat = load_ratings(v)
        px, ox = p_a, np.array([others_at_s1(rat[d.id]) for d in clean_days])
        res[f"exploratory_{v}"] = {
            "s_from_choose_stage": s,
            "all_clean": l1(px + s * ox, y_clean),
            "fresh_only": l1(px[fresh] + s * ox[fresh], y_clean[fresh]),
            "direction": describe(clean_days, ox, load_ratings(v)),
        }
    # the registered decision
    n = len(clean_days)
    b, nc = res["all_clean"]["b_people_only"], res["all_clean"]["no_change"]
    a_score = res["all_clean"]["a_current_s143"]
    if n < 20:
        verdict = f"provisional: only {n} clean days; nothing changes"
    else:
        c_ok = False
        if variants:
            v = variants[0]
            c_ok = (
                res[v]["all_clean"] < b
                and res[v]["all_clean"] < nc
                and res[v]["fresh_only"] <= res["fresh_only"]["b_people_only"]
            )
        if c_ok:
            verdict = f"adopt {variants[0]} with s = {res[variants[0]]['s_refit']}"
        elif b < a_score:
            verdict = "other investors' news reaction defaults to s = 0"
        else:
            verdict = "the current engine stays"
    res["verdict"] = verdict
    text = json.dumps(rounded(res, 3), indent=1)
    (BENCH / "b11_result.json").write_text(text, encoding="utf-8")
    print(text)
    print("cost:", round(client.budget.used_usd, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["closes", "choose", "clean"])
    ap.add_argument("--rate", action="store_true")
    args = ap.parse_args()
    if args.stage == "closes":
        fetch_closes()
        return 0
    client = make_client(BENCH, "anthropic" if args.rate else "scripted", replay_only=not args.rate)
    if args.stage == "choose":
        choose(client)
    else:
        clean(client, json.loads((BENCH / "b11_choose.json").read_text(encoding="utf-8"))["chosen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
