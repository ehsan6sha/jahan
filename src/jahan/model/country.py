"""Country specification loaded from ``data/countries/{ISO3}.yaml``.

Every block carries ``source`` (a registry id).  Percent tables are normalised to
fractions on load after checking they sum to 100 (or 1) within 1.5%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUM_TOLERANCE = 0.015


class DataError(ValueError):
    pass


def normalise(shares: dict, what: str) -> dict:
    total = float(sum(shares.values()))
    if total <= 0:
        raise DataError(f"{what}: shares sum to {total}")
    if abs(total - 100.0) <= 100.0 * SUM_TOLERANCE:
        pass
    elif abs(total - 1.0) <= SUM_TOLERANCE:
        pass
    else:
        raise DataError(f"{what}: shares sum to {total:.4f}, expected 100 or 1 (±{SUM_TOLERANCE:.1%})")
    return {k: float(v) / total for k, v in shares.items()}


def normalise_list(values: list[float], what: str) -> list[float]:
    d = normalise({i: v for i, v in enumerate(values)}, what)
    return [d[i] for i in range(len(values))]


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str


class ValueBlock(Block):
    value: float


class TZRow(BaseModel):
    offset_min: int
    share: float


class TimeZones(Block):
    rows: list[TZRow]

    @model_validator(mode="after")
    def _norm(self) -> TimeZones:
        shares = normalise_list([r.share for r in self.rows], "time_zones")
        for r, s in zip(self.rows, shares, strict=True):
            r.share = s
        return self


class AgeSex(Block):
    bin_edges: list[int]
    total_pct: list[float]
    male_share: list[float]

    @model_validator(mode="after")
    def _check(self) -> AgeSex:
        nb = len(self.total_pct)
        if len(self.bin_edges) != nb + 1 or len(self.male_share) != nb:
            raise DataError("age_sex: bin_edges must have one more entry than total_pct and male_share")
        if any(b >= a for a, b in zip(self.bin_edges[1:], self.bin_edges[:-1], strict=False)):
            raise DataError("age_sex: bin_edges must increase")
        if any(not (0.0 < m < 1.0) for m in self.male_share):
            raise DataError("age_sex: male_share must be in (0,1)")
        self.total_pct = normalise_list(self.total_pct, "age_sex.total_pct")
        return self


class Shares(Block):
    shares: dict[str, float]

    @model_validator(mode="after")
    def _norm(self) -> Shares:
        self.shares = normalise(self.shares, f"shares({self.source})")
        return self


class ReligionSpec(Shares):
    """National religious affiliation, optionally conditioned on ethnicity."""

    by_ethnicity: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _norm_by_eth(self) -> ReligionSpec:
        norm = {}
        for eth, dist in self.by_ethnicity.items():
            unknown = set(dist) - set(self.shares)
            if unknown:
                raise DataError(f"religion.by_ethnicity.{eth}: unknown religions {sorted(unknown)}")
            norm[eth] = normalise(dist, f"religion.by_ethnicity.{eth}")
        self.by_ethnicity = norm
        return self


class L1Spec(BaseModel):
    default: dict[str, float]
    by_ethnicity: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _norm(self) -> L1Spec:
        self.default = normalise(self.default, "languages.l1.default")
        self.by_ethnicity = {k: normalise(v, f"languages.l1.by_ethnicity.{k}") for k, v in self.by_ethnicity.items()}
        return self


class Languages(Block):
    l1: L1Spec
    l2_given_l1: dict[str, dict[str, float]] = Field(default_factory=dict)
    l2_education_multiplier: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])


class AgeRow(BaseModel):
    age_min: float
    age_max: float
    sex: Literal["F", "M"] | None = None
    shares: dict[str, float]

    @field_validator("shares", mode="before")
    @classmethod
    def _keys_to_str(cls, v: dict) -> dict:
        return {str(k): float(x) for k, x in v.items()}

    @model_validator(mode="after")
    def _norm(self) -> AgeRow:
        self.shares = normalise(self.shares, f"age row {self.age_min}-{self.age_max}")
        return self


class AgeRows(Block):
    rows: list[AgeRow]


class Occupation(Block):
    by_edu_group: dict[Literal["low", "mid", "high"], dict[int, float]]

    @model_validator(mode="after")
    def _norm(self) -> Occupation:
        for g in ("low", "mid", "high"):
            if g not in self.by_edu_group:
                raise DataError(f"occupation.by_edu_group missing '{g}'")
            self.by_edu_group[g] = {int(k): v for k, v in normalise(self.by_edu_group[g], g).items()}
        return self


class IncomeRow(BaseModel):
    median: float
    sigma: float


class Income(Block):
    currency: str
    by_occupation: dict[int, IncomeRow]


class AgeShare(BaseModel):
    age_min: float
    age_max: float
    share: float


class MotherAge(BaseModel):
    min: float
    max: float
    mean: float
    sd: float


class Household(Block):
    size_pct: dict[int, float]
    married_share_by_age: list[AgeShare]
    spouse_age_gap_mean: float
    spouse_age_gap_sd: float
    adult_child_at_home_by_age: list[AgeShare]
    mother_age_at_birth: MotherAge

    @model_validator(mode="after")
    def _norm(self) -> Household:
        self.size_pct = {int(k): v for k, v in normalise(self.size_pct, "household.size_pct").items()}
        return self

    def mean_size(self) -> float:
        return sum(k * v for k, v in self.size_pct.items())


class AgeShareRows(Block):
    rows: list[AgeShare]


class InfoTierSpec(Block):
    by_edu_group: dict[Literal["low", "mid", "high"], list[float]]

    @model_validator(mode="after")
    def _norm(self) -> InfoTierSpec:
        for g, v in self.by_edu_group.items():
            if len(v) != 4:
                raise DataError("info_tier rows need 4 values")
            self.by_edu_group[g] = normalise_list(v, f"info_tier.{g}")
        return self


class Sensory(Block):
    vision_pct: list[AgeShare]
    hearing_pct: list[AgeShare]


class Big5Spec(Block):
    mean: list[float]
    sd: list[float]


class ValuesSpec(Block):
    mean: dict[str, float]
    concentration: float

    @model_validator(mode="after")
    def _norm(self) -> ValuesSpec:
        self.mean = normalise(self.mean, "values.mean")
        return self


class ChannelSpec(BaseModel):
    name: str
    kind: str
    lang: str
    reach_by_tier: list[float]
    credibility: float
    local_hours: tuple[int, int]

    @model_validator(mode="after")
    def _check(self) -> ChannelSpec:
        if len(self.reach_by_tier) != 4 or any(not (0 <= r <= 1) for r in self.reach_by_tier):
            raise DataError(f"channel {self.name}: reach_by_tier needs 4 values in [0,1]")
        a, b = self.local_hours
        if not (0 <= a < b <= 24):
            raise DataError(f"channel {self.name}: local_hours must satisfy 0 <= start < end <= 24")
        return self


class Media(Block):
    channels: list[ChannelSpec]


SHIFT_KINDS = ("religion", "ethnicity", "education_group", "region", "age_band")
CAMP_TAU = 1.0 / 3.0  # political camp band boundary on the [-1, 1] alignment scale (camp shares come from data)


class AxisLabels(BaseModel):
    """Words for the two ends of the country's main political axis (+1 = the incumbent's camp)."""

    model_config = ConfigDict(extra="forbid")
    pos: str
    neg: str
    note: str = ""


class AlignmentShift(BaseModel):
    """A sourced log-odds offset on the latent political lean for one demographic group."""

    model_config = ConfigDict(extra="forbid")
    by: Literal["religion", "ethnicity", "education_group", "region", "age_band"]
    key: str  # a religion/ethnicity code; low|mid|high; urban|rural; "18-29"
    logit: float
    source: str  # registry id
    note: str = ""


class Politics(Block):
    """Camp shares of adults on one axis per country.  ``placeholder`` marks shares with no source."""

    axis: AxisLabels
    camps: dict[Literal["pos", "neutral", "neg"], float]
    shifts: list[AlignmentShift] = Field(default_factory=list)
    placeholder: bool = False

    @model_validator(mode="after")
    def _norm(self) -> Politics:
        if set(self.camps) != {"pos", "neutral", "neg"}:
            raise DataError("politics.camps needs pos, neutral and neg")
        self.camps = normalise(self.camps, "politics.camps")
        return self


class NamesRef(Block):
    file: str


class GeneratedRef(Block):
    """A file produced by a scripts/data/build_*.py pipeline.

    kind ``wpp``: population, age_sex, fertility, life_table, indicators
    kind ``wdi``: urban_share, internet_target, fx_local_per_usd (unless the country file sets it), wdi values
    kind ``ilo``: education, employment, occupation (when the source publishes them)
    """

    kind: Literal["wpp", "wdi", "ilo"]
    file: str


class Fertility(Block):
    age_start: list[int]
    asfr_per_woman_year: list[float]

    @model_validator(mode="after")
    def _check(self) -> Fertility:
        if len(self.age_start) != len(self.asfr_per_woman_year):
            raise DataError("fertility: age_start and asfr lengths differ")
        return self

    def tfr(self) -> float:
        widths = [b - a for a, b in zip(self.age_start, self.age_start[1:], strict=False)] + [5]
        return sum(w * f for w, f in zip(widths, self.asfr_per_woman_year, strict=True))


class LifeTable(Block):
    age_start: list[int]
    qx_female: list[float]
    qx_male: list[float]

    @model_validator(mode="after")
    def _check(self) -> LifeTable:
        n = len(self.age_start)
        if len(self.qx_female) != n or len(self.qx_male) != n:
            raise DataError("life_table: lengths differ")
        if any(not (0 <= q <= 1) for q in self.qx_female + self.qx_male):
            raise DataError("life_table: qx must be in [0,1]")
        return self


class Vital(Block):
    """Rates not covered by WPP (marriage, divorce, labour-market flows)."""

    marriage_rate_per_1000: float  # marriages per 1000 population per year
    divorce_rate_per_1000: float  # divorces per 1000 population per year
    job_change_per_year: float  # share of employed who change employer per year
    job_loss_per_month: float  # employed -> unemployed monthly hazard
    job_find_per_month: float  # unemployed -> employed monthly hazard
    marriage_age_peak: float = 29.0
    marriage_age_sd: float = 8.0


class CountrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iso3: str
    name: str
    currency: str
    fx_local_per_usd: float
    fx_note: str = ""
    languages_official: list[str]
    start_dow: int = 0
    generated: list[GeneratedRef] = Field(default_factory=list)
    internet_target: float | None = None  # share of the whole population online (World Bank); rescales the age profile
    wdi: dict[str, float] = Field(default_factory=dict)
    population: ValueBlock
    urban_share: ValueBlock
    time_zones: TimeZones
    age_sex: AgeSex
    fertility: Fertility | None = None
    life_table: LifeTable | None = None
    indicators: dict[str, float | None] = Field(default_factory=dict)
    vital: Vital | None = None
    ethnicity: Shares
    religion: ReligionSpec
    languages: Languages
    education: AgeRows
    employment: AgeRows
    occupation: Occupation
    income: Income
    household: Household
    internet: AgeShareRows
    info_tier: InfoTierSpec
    sensory: Sensory
    big5: Big5Spec
    values: ValuesSpec
    media: Media
    names: NamesRef
    politics: Politics | None = None

    @model_validator(mode="after")
    def _politics_keys(self) -> CountrySpec:
        if self.politics is None:
            return self
        known = {
            "religion": set(self.religion.shares),
            "ethnicity": set(self.ethnicity.shares),
            "education_group": {"low", "mid", "high"},
            "region": {"urban", "rural"},
        }
        for s in self.politics.shifts:
            if s.by == "age_band":
                lo, _, hi = s.key.partition("-")
                if not (lo.isdigit() and hi.isdigit() and int(lo) < int(hi)):
                    raise DataError(f"politics shift age_band {s.key!r} must look like '18-29'")
            elif s.key not in known[s.by]:
                raise DataError(f"politics shift {s.by}={s.key!r} is not a known code")
        return self

    def source_ids(self) -> list[str]:
        out = []
        for name in type(self).model_fields:
            v = getattr(self, name)
            if isinstance(v, Block):
                out.append(v.source)
        if self.politics is not None:
            out += [s.source for s in self.politics.shifts]
        return out

    def ethnicity_codes(self) -> list[str]:
        return list(self.ethnicity.shares)

    def religion_codes(self) -> list[str]:
        return list(self.religion.shares)


def load_country(data_dir: Path, iso3: str) -> CountrySpec:
    path = data_dir / "countries" / f"{iso3}.yaml"
    if not path.exists():
        raise DataError(f"no country file {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for ref in raw.get("generated", []):
        gen_path = data_dir / ref["file"]
        if not gen_path.exists():
            raise DataError(f"{iso3}: generated file {gen_path} missing (run scripts/data/build_{ref['kind']}.py)")
        with gen_path.open(encoding="utf-8") as f:
            gen = yaml.safe_load(f)
        _merge(raw, ref["kind"], ref["source"], gen, iso3)
    try:
        return CountrySpec.model_validate(raw)
    except Exception as e:  # pydantic ValidationError or DataError
        raise DataError(f"{iso3}: {e}") from e


def _merge(raw: dict, kind: str, src: str, gen: dict, iso3: str) -> None:
    def put(key: str, value: dict) -> None:
        if key in raw:
            raise DataError(f"{iso3}: '{key}' is supplied by the {kind} file; remove it from the country file")
        raw[key] = value

    if kind == "wpp":
        put("population", {"source": src, "value": gen["population"]})
        put("age_sex", {"source": src, **gen["age_sex"]})
        if "fertility" in gen:
            put("fertility", {"source": src, **gen["fertility"]})
        if "life_table" in gen:
            put("life_table", {"source": src, **gen["life_table"]})
        raw["indicators"] = gen.get("indicators", {})
    elif kind == "wdi":
        ind = {k: v["value"] for k, v in gen.get("indicators", {}).items()}
        raw["wdi"] = ind
        if "urban_pct" in ind:
            put("urban_share", {"source": src, "value": ind["urban_pct"] / 100.0})
        if "internet_users_pct" in ind:
            raw["internet_target"] = ind["internet_users_pct"] / 100.0
        if "fx_local_per_usd" not in raw and "fx_local_per_usd" in ind:
            raw["fx_local_per_usd"] = ind["fx_local_per_usd"]
            raw["fx_note"] = f"official rate, World Bank PA.NUS.FCRF ({src})"
    elif kind == "ilo":
        for key in ("education", "employment", "occupation"):
            if key in gen:
                put(key, {"source": src, **gen[key]})
    else:  # pragma: no cover - guarded by the schema
        raise DataError(f"unknown generated kind {kind}")


def generated_files(data_dir: Path, iso3: str) -> list[Path]:
    with (data_dir / "countries" / f"{iso3}.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [data_dir / r["file"] for r in raw.get("generated", [])]
