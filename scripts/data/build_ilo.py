"""Build education, employment-status and occupation tables from ILOSTAT.

    # 1. download raw CSVs (network) into their own folder
    uv run python -I -X utf8 scripts/data/build_ilo.py download --raw data/raw/ilostat --countries USA,IRN,...
    # 2. build per-country tables (no network)
    uv run python -I -X utf8 scripts/data/build_ilo.py build --raw data/raw/ilostat \
        --out data/generated/ilo --countries ...

Sourced (copied from ILOSTAT, latest year with full breakdowns):
  * employment ratio and labour-force participation by sex and age band
  * unemployment = participation - employment ratio
  * youth NEET rate (used to separate students from other inactive youth)
  * educational attainment of the working-age population by sex and age band
  * employment by occupation (ISCO-08 major group) and education level
HEURISTIC (no ILO breakdown; documented in the output): how the population outside the labour
force splits into student / homemaker / retired / other inactive for ages 25+.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

API = "https://rplumber.ilo.org/data/indicator/?id={ind}&ref_area={iso}&format=.csv&timefrom={start}"
INDICATORS = [
    "EMP_DWAP_SEX_AGE_RT_A",
    "EAP_DWAP_SEX_AGE_RT_A",
    "EIP_NEET_SEX_AGE_RT_A",
    "POP_XWAP_SEX_AGE_EDU_NB_A",
    "EMP_TEMP_SEX_OCU_EDU_NB_A",
]
AGE_ROWS = [  # (age_min, age_max, ILO age codes in order of preference)
    (15, 20, ["AGE_5YRBANDS_Y15-19"]),
    (20, 25, ["AGE_5YRBANDS_Y20-24"]),
    (25, 35, ["AGE_10YRBANDS_Y25-34"]),
    (35, 45, ["AGE_10YRBANDS_Y35-44"]),
    (45, 55, ["AGE_10YRBANDS_Y45-54"]),
    (55, 65, ["AGE_10YRBANDS_Y55-64"]),
    (65, 120, ["AGE_10YRBANDS_YGE65", "AGE_5YRBANDS_YGE65", "AGE_AGGREGATE_YGE65"]),
]
YOUTH_FALLBACK = ["AGE_10YRBANDS_Y15-24", "AGE_YTHADULT_Y15-24", "AGE_AGGREGATE_Y15-24"]
# education attainment rows; the 18-24 row uses 20-24 when published, else 15-24 (understates
# attainment slightly because teenagers are still in school)
EDU_ROWS = [
    (18, 25, ["AGE_5YRBANDS_Y20-24", "AGE_10YRBANDS_Y15-24", "AGE_YTHADULT_Y15-24"]),
    (25, 35, ["AGE_10YRBANDS_Y25-34"]),
    (35, 45, ["AGE_10YRBANDS_Y35-44"]),
    (45, 55, ["AGE_10YRBANDS_Y45-54"]),
    (55, 65, ["AGE_10YRBANDS_Y55-64"]),
    (65, 120, ["AGE_10YRBANDS_YGE65", "AGE_AGGREGATE_YGE65"]),
]
ISCED11 = {f"EDU_ISCED11_{i}": i for i in range(9)}
AGGREGATE_TO_ISCED = {"EDU_AGGREGATE_LTB": 1, "EDU_AGGREGATE_BAS": 2, "EDU_AGGREGATE_INT": 3, "EDU_AGGREGATE_ADV": 6}
AGGREGATE_TO_GROUP = {
    "EDU_AGGREGATE_LTB": "low",
    "EDU_AGGREGATE_BAS": "low",
    "EDU_AGGREGATE_INT": "mid",
    "EDU_AGGREGATE_ADV": "high",
}
# HEURISTIC split of the population outside the labour force aged 25+ (sex, age_min) -> shares
OUTSIDE_LF = {
    ("F", 25): {"homemaker": 0.65, "student": 0.05, "inactive": 0.30},
    ("M", 25): {"homemaker": 0.10, "student": 0.10, "inactive": 0.80},
    ("F", 55): {"retired": 0.55, "homemaker": 0.30, "inactive": 0.15},
    ("M", 55): {"retired": 0.75, "homemaker": 0.03, "inactive": 0.22},
    ("F", 65): {"retired": 0.80, "homemaker": 0.15, "inactive": 0.05},
    ("M", 65): {"retired": 0.95, "inactive": 0.05},
}


def download(raw: Path, countries: list[str], start: int) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for iso in countries:
        for ind in INDICATORS:
            url = API.format(ind=ind, iso=iso, start=start)
            req = urllib.request.Request(url, headers={"User-Agent": "jahan-data-pipeline/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310 (fixed https host)
                data = r.read()
            (raw / f"{ind}__{iso}.csv").write_bytes(data)
            print(f"{iso} {ind}: {len(data):,} bytes")


def _read(raw: Path, ind: str, iso: str) -> list[dict]:
    p = raw / f"{ind}__{iso}.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("obs_value") not in (None, "")]


def _latest_year(rows: list[dict], need: set[str]) -> str | None:
    """Latest year whose rows cover every key in ``need`` (keys are 'sex|classif1')."""
    by_year: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_year[r["time"]].add(f"{r['sex']}|{r['classif1']}")
    for year in sorted(by_year, reverse=True):
        if need <= by_year[year]:
            return year
    return None


def _value(rows: list[dict], year: str, sex: str, codes: list[str]) -> float | None:
    for code in codes:
        for r in rows:
            if r["time"] == year and r["sex"] == sex and r["classif1"] == code:
                return float(r["obs_value"])
    return None


def employment_rows(raw: Path, iso: str) -> tuple[list[dict], dict]:
    emp = _read(raw, "EMP_DWAP_SEX_AGE_RT_A", iso)
    eap = _read(raw, "EAP_DWAP_SEX_AGE_RT_A", iso)
    neet = _read(raw, "EIP_NEET_SEX_AGE_RT_A", iso)
    need = {f"{s}|{c[0]}" for s in ("SEX_F", "SEX_M") for _, _, c in AGE_ROWS[2:]}
    year = _latest_year(emp, need)
    if year is None or _latest_year(eap, need) is None:
        return [], {}
    rows = []
    neet_year = max((r["time"] for r in neet), default=None)
    for age_min, age_max, codes in AGE_ROWS:
        for sex in ("F", "M"):
            sc = f"SEX_{sex}"
            alt = codes + (YOUTH_FALLBACK if age_min < 25 else [])
            e = _value(emp, year, sc, alt)
            lf = _value(eap, year, sc, alt)
            if e is None or lf is None:
                continue
            e, lf = e / 100, lf / 100
            u = max(lf - e, 0.0)
            out = max(1.0 - lf, 0.0)
            shares: dict[str, float] = {"employed": e, "unemployed": u}
            if age_min < 25:
                n = _value(neet, neet_year, sc, alt) if neet_year else None
                neet_rate = (n / 100) if n is not None else 0.15  # HEURISTIC fallback
                idle = min(max(neet_rate - u, 0.0), out)
                shares["student"] = out - idle
                shares["homemaker"] = idle * (0.5 if sex == "F" else 0.1)
                shares["inactive"] = idle - shares["homemaker"]
            else:
                key = (sex, 65 if age_min >= 65 else 55 if age_min >= 55 else 25)
                for k, v in OUTSIDE_LF[key].items():
                    shares[k] = shares.get(k, 0.0) + out * v
            total = sum(shares.values())
            rows.append(
                {
                    "age_min": age_min,
                    "age_max": age_max,
                    "sex": sex,
                    "shares": {k: round(100 * v / total, 3) for k, v in shares.items() if v > 0},
                }
            )
    return rows, {"employment_year": year, "neet_year": neet_year}


def education_rows(raw: Path, iso: str) -> tuple[list[dict], dict]:
    pop = _read(raw, "POP_XWAP_SEX_AGE_EDU_NB_A", iso)
    if not pop:
        return [], {}
    has_isced = any(r["classif2"] in ISCED11 for r in pop)
    years = sorted({r["time"] for r in pop}, reverse=True)
    for year in years:
        rows = []
        for age_min, age_max, codes in EDU_ROWS:
            for sex in ("F", "M"):
                dist: dict[int, float] = defaultdict(float)
                for code in codes:
                    sel = [r for r in pop if r["time"] == year and r["sex"] == f"SEX_{sex}" and r["classif1"] == code]
                    for r in sel:
                        c2 = r["classif2"]
                        if has_isced and c2 in ISCED11:
                            dist[ISCED11[c2]] += float(r["obs_value"])
                        elif not has_isced and c2 in AGGREGATE_TO_ISCED:
                            dist[AGGREGATE_TO_ISCED[c2]] += float(r["obs_value"])
                    if dist:
                        break
                total = sum(dist.values())
                if total <= 0:
                    break
                rows.append(
                    {
                        "age_min": age_min,
                        "age_max": age_max,
                        "sex": sex,
                        "shares": {str(k): round(100 * v / total, 3) for k, v in sorted(dist.items()) if v > 0},
                    }
                )
            else:
                continue
            break
        if len(rows) == 2 * len(EDU_ROWS):
            return rows, {"education_year": year, "education_detail": "ISCED-11" if has_isced else "aggregate levels"}
    return [], {}


def occupation_by_edu(raw: Path, iso: str) -> tuple[dict, dict]:
    rows = _read(raw, "EMP_TEMP_SEX_OCU_EDU_NB_A", iso)
    rows = [r for r in rows if r["sex"] == "SEX_T" and r["classif1"].startswith("OCU_ISCO08_")]
    if not rows:
        return {}, {}
    for year in sorted({r["time"] for r in rows}, reverse=True):
        acc: dict[str, dict[int, float]] = {
            "low": defaultdict(float),
            "mid": defaultdict(float),
            "high": defaultdict(float),
        }
        for r in rows:
            if r["time"] != year:
                continue
            occ = r["classif1"].removeprefix("OCU_ISCO08_")
            if not occ.isdigit():
                continue
            c2 = r["classif2"]
            if c2 in AGGREGATE_TO_GROUP:
                grp = AGGREGATE_TO_GROUP[c2]
            elif c2 in ISCED11:
                i = ISCED11[c2]
                grp = "low" if i <= 2 else "mid" if i <= 5 else "high"
            else:
                continue
            acc[grp][int(occ)] += float(r["obs_value"])
        if all(sum(v.values()) > 0 for v in acc.values()):
            out = {g: {k: round(100 * v / sum(d.values()), 3) for k, v in sorted(d.items())} for g, d in acc.items()}
            return out, {"occupation_year": year}
    return {}, {}


def build(raw: Path, out: Path, countries: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for iso in countries:
        doc: dict = {
            "iso3": iso,
            "source": "ILOSTAT (International Labour Organization), via rplumber.ilo.org API",
            "license": "CC BY 4.0",
            "built": dt.date.today().isoformat(),
            "indicators": INDICATORS,
            "heuristics": "split of population outside the labour force aged 25+ (see build_ilo.py OUTSIDE_LF)",
        }
        emp, m1 = employment_rows(raw, iso)
        edu, m2 = education_rows(raw, iso)
        occ, m3 = occupation_by_edu(raw, iso)
        doc["meta"] = {**m1, **m2, **m3}
        if emp:
            doc["employment"] = {"rows": emp}
        if edu:
            doc["education"] = {"rows": edu}
        if occ:
            doc["occupation"] = {"by_edu_group": occ}
        (out / f"{iso}.yaml").write_text(
            "# Generated by scripts/data/build_ilo.py from ILOSTAT. Do not edit by hand.\n"
            + yaml.safe_dump(doc, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"{iso}: employment {'yes ' + m1.get('employment_year', '') if emp else 'MISSING'}; "
            f"education {(m2.get('education_year', '') + ' ' + m2.get('education_detail', '')) if edu else 'MISSING'}; "
            f"occupation {'yes ' + m3.get('occupation_year', '') if occ else 'MISSING'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "build"])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--countries", default="USA")
    ap.add_argument("--start", type=int, default=2015)
    a = ap.parse_args()
    countries = [c.strip().upper() for c in a.countries.split(",")]
    if a.mode == "download":
        download(a.raw, countries, a.start)
    else:
        if a.out is None:
            ap.error("--out is required for build")
        build(a.raw, a.out, countries)


if __name__ == "__main__":
    main()
