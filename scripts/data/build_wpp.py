"""Build per-country demography files from UN World Population Prospects 2024 CSVs.

Usage (raw files downloaded into their own folder first; see docs/data-pipeline.md):

    uv run python -I scripts/data/build_wpp.py --raw data/raw/wpp2024 --out data/generated/wpp2024 \
        --year 2026 --countries USA,BRA,DEU,NGA,IRN,IND,CHN,IDN

Outputs ``{ISO3}.yaml`` with population, 5-year age/sex pyramid, age-specific fertility,
abridged life table (qx by sex) and summary indicators for the requested year
(medium variant).  Values are copied from the source files without modelling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import yaml

BASE_URL = "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"
FILES = {
    "pop": "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz",
    "ind": "WPP2024_Demographic_Indicators_Medium.csv.gz",
    "fert": "WPP2024_Fertility_by_Age5.csv.gz",
    "life": "WPP2024_Life_Table_Abridged_Medium_2024-2100.csv.gz",
}
INDICATORS = {
    "TPopulation1July": "population_thousands",
    "MedianAgePop": "median_age",
    "CBR": "crude_birth_rate_per_1000",
    "TFR": "total_fertility_rate",
    "SRB": "sex_ratio_at_birth_males_per_100_females",
    "CDR": "crude_death_rate_per_1000",
    "LEx": "life_expectancy_at_birth",
    "LExMale": "life_expectancy_male",
    "LExFemale": "life_expectancy_female",
    "IMR": "infant_mortality_per_1000_births",
    "CNMR": "net_migration_rate_per_1000",
}


def _filter_stream(path: Path, iso3: set[str], year: int, columns: list[str]) -> pa.Table:
    """Stream a large gzipped CSV and keep rows for the requested countries, year and medium variant."""
    types = {c: pa.string() for c in ("ISO3_code", "Variant", "Sex", "AgeGrp") if c in columns}
    types.update(
        {c: pa.float64() for c in columns if c not in types and c not in ("Time", "AgeGrpStart", "AgeGrpSpan")}
    )
    types.update({c: pa.int64() for c in ("Time", "AgeGrpStart", "AgeGrpSpan") if c in columns})
    convert = pacsv.ConvertOptions(include_columns=columns, strings_can_be_null=True, column_types=types)
    parts = []
    with gzip.open(path, "rb") as fh:
        reader = pacsv.open_csv(fh, convert_options=convert, read_options=pacsv.ReadOptions(block_size=1 << 24))
        for batch in reader:
            t = pa.Table.from_batches([batch])
            mask = pc.and_(pc.is_in(t["ISO3_code"], pa.array(sorted(iso3))), pc.equal(t["Time"], year))
            if "Variant" in t.column_names:
                mask = pc.and_(mask, pc.equal(t["Variant"], "Medium"))
            t = t.filter(mask)
            if t.num_rows:
                parts.append(t)
    if not parts:
        raise SystemExit(f"no rows for {sorted(iso3)} in {year} in {path.name}")
    return pa.concat_tables(parts)


def _rows(t: pa.Table, iso: str) -> list[dict]:
    return [r for r in t.to_pylist() if r["ISO3_code"] == iso]


def build(raw: Path, out: Path, year: int, countries: list[str]) -> None:
    iso = set(countries)
    pop = _filter_stream(
        raw / FILES["pop"],
        iso,
        year,
        ["ISO3_code", "Time", "Variant", "AgeGrp", "AgeGrpStart", "AgeGrpSpan", "PopMale", "PopFemale", "PopTotal"],
    )
    ind = _filter_stream(raw / FILES["ind"], iso, year, ["ISO3_code", "Time", "Variant", *INDICATORS])
    fert = _filter_stream(
        raw / FILES["fert"], iso, year, ["ISO3_code", "Time", "Variant", "AgeGrp", "AgeGrpStart", "ASFR"]
    )
    life_path = raw / FILES["life"]
    life = (
        _filter_stream(life_path, iso, year, ["ISO3_code", "Time", "Variant", "Sex", "AgeGrpStart", "AgeGrpSpan", "qx"])
        if life_path.exists()
        else None
    )
    out.mkdir(parents=True, exist_ok=True)
    for c in countries:
        p = sorted(_rows(pop, c), key=lambda r: r["AgeGrpStart"])
        starts = [int(r["AgeGrpStart"]) for r in p]
        total = sum(float(r["PopTotal"]) for r in p)
        doc: dict = {
            "iso3": c,
            "year": year,
            "variant": "Medium",
            "source": "UN DESA Population Division, World Population Prospects 2024",
            "license": "CC BY 3.0 IGO",
            "retrieved": dt.date.today().isoformat(),
            "files": [BASE_URL + f for f in FILES.values()],
            "population": round(total * 1000),
            "age_sex": {
                "bin_edges": starts + [starts[-1] + 5],
                "total_pct": [round(100 * float(r["PopTotal"]) / total, 4) for r in p],
                "male_share": [round(float(r["PopMale"]) / float(r["PopTotal"]), 5) for r in p],
            },
        }
        ir = _rows(ind, c)
        if ir:
            doc["indicators"] = {
                v: (round(float(ir[0][k]), 4) if ir[0][k] is not None else None) for k, v in INDICATORS.items()
            }
        fr = sorted(_rows(fert, c), key=lambda r: r["AgeGrpStart"])
        doc["fertility"] = {
            "age_start": [int(r["AgeGrpStart"]) for r in fr],
            "asfr_per_woman_year": [round(float(r["ASFR"]) / 1000.0, 6) for r in fr],
        }
        if life is not None:
            lt = _rows(life, c)
            by_sex: dict[str, list[tuple[int, float]]] = {}
            for r in lt:
                by_sex.setdefault(str(r["Sex"]).lower(), []).append((int(r["AgeGrpStart"]), float(r["qx"])))
            fem = sorted(by_sex.get("female", []))
            mal = sorted(by_sex.get("male", []))
            doc["life_table"] = {
                "age_start": [a for a, _ in fem],
                "qx_female": [round(q, 6) for _, q in fem],
                "qx_male": [round(q, 6) for _, q in mal],
            }
        (out / f"{c}.yaml").write_text(
            "# Generated by scripts/data/build_wpp.py from UN WPP 2024. Do not edit by hand.\n"
            + yaml.safe_dump(doc, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"{c}: population {doc['population']:,}, {len(starts)} age groups, "
            f"{len(doc['fertility']['age_start'])} fertility groups, life table: {'yes' if life is not None else 'no'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--countries", default="USA")
    a = ap.parse_args()
    build(a.raw, a.out, a.year, [c.strip().upper() for c in a.countries.split(",")])


if __name__ == "__main__":
    main()
