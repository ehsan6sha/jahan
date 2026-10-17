"""Everyday activity settings: how people look at their accounts, how life events and conversation move mood,
saving and cashing out (``data/global/everyday.yaml``), and every country's own stock market
(``data/global/investors_by_country.yaml``; the US ownership bands stay in ``investors.yaml``).

Every number carries its provenance: a registry ``source``, ``heuristic: true`` with a note, or
``borrowed_from: <ISO3>`` with a note (a number measured in one country used for another).  Neither file is
hashed into snapshots: `economy_cfg` copies the resolved numbers into the world (at generation or in the
switch event), so editing the files never changes or breaks an existing world.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jahan.model.investors import InvestorsConfig, load_investors

COUNTRY_FILE = "investors_by_country.yaml"
EVERYDAY_FILE = "everyday.yaml"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str | None = None
    heuristic: bool = False
    borrowed_from: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _one(self) -> Provenance:
        n = sum([self.source is not None, self.heuristic, self.borrowed_from is not None])
        if n != 1:
            raise ValueError("give exactly one of source, heuristic: true, borrowed_from")
        if (self.heuristic or self.borrowed_from) and not self.note:
            raise ValueError("a heuristic or borrowed value needs a note saying why")
        return self

    def label(self) -> str:
        if self.source is not None:
            return self.source
        return "HEURISTIC" if self.heuristic else f"borrowed from {self.borrowed_from}"


class Number(Provenance):
    value: float


class Numbers(Provenance):
    values: list[float]


# ------------------------------------------------------------------------------------- the shared mechanism


class Attention(BaseModel):
    """Days logged in out of ``of_days`` at each percentile of investors (piecewise-linear between knots)."""

    model_config = ConfigDict(extra="forbid")
    source: str
    of_days: float = Field(gt=0)
    percentiles: list[float]
    days: list[float]
    published_mean: float
    down_day_factor: Number
    weekend_factor: Number

    @model_validator(mode="after")
    def _knots(self) -> Attention:
        if len(self.percentiles) != len(self.days) or self.percentiles[0] != 0 or self.percentiles[-1] != 100:
            raise ValueError("attention knots must pair up and run from percentile 0 to 100")
        if sorted(self.percentiles) != self.percentiles or sorted(self.days) != self.days:
            raise ValueError("attention knots must increase")
        return self

    def mean_days(self) -> float:
        """Mean days logged in implied by the interpolation (reported next to the published mean)."""
        tot = 0.0
        for k in range(1, len(self.percentiles)):
            tot += (self.percentiles[k] - self.percentiles[k - 1]) / 100.0 * 0.5 * (self.days[k] + self.days[k - 1])
        return tot


class Mood(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unemployed_shift: Number
    contagion: Number


class CashoutByBalance(Provenance):
    """Share of people leaving a job who take cash, by plan balance: ``cash_share[k]`` applies below
    ``balance_usd[k]``, the last one above the last threshold."""

    balance_usd: list[float]
    cash_share: list[float]

    @model_validator(mode="after")
    def _shape(self) -> CashoutByBalance:
        if len(self.cash_share) != len(self.balance_usd) + 1 or sorted(self.balance_usd) != self.balance_usd:
            raise ValueError("cash_share needs one more entry than the increasing balance_usd thresholds")
        return self


class Saving(BaseModel):
    """Workplace retirement saving; only the countries listed are modelled."""

    model_config = ConfigDict(extra="forbid")
    countries: list[str]
    participation_by_wage_quartile: Numbers  # percent of employed workers in each wage quartile
    contribution_share_of_pay: Number
    equity_share: Number  # share of plan balances in stocks
    dc_share_of_equity: Number  # share of a holder's stock held in retirement plans
    cashout_by_balance: CashoutByBalance
    hardship_rate: Number  # share of participants taking a hardship withdrawal per year
    hardship_amount_usd: Number
    drawdown_rate: Number  # share of a retiree's stock sold per year (display only: steady flows are priced in)


class Everyday(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attention: Attention
    mood: Mood
    saving: Saving
    min_holders: Number  # fewer simulated holders than this: the market is labelled illustrative

    def sources(self) -> list[str]:
        out = {self.attention.source}
        for v in (
            self.attention.down_day_factor,
            self.attention.weekend_factor,
            self.mood.unemployed_shift,
            self.mood.contagion,
            self.min_holders,
            *[getattr(self.saving, f) for f in Saving.model_fields if f != "countries"],
        ):
            if v.source:
                out.add(v.source)
        return sorted(out)


# --------------------------------------------------------------------------------------------- per country


class IndexSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str  # "DAX"
    label: str  # "German stocks": used in headlines and the watch list
    currency: str
    weekdays: list[int]  # trading days, Python weekday numbers (0 = Monday)
    weekdays_source: Provenance
    max_daily_pct: Number


class Ownership(Provenance):
    """Share of adults holding listed stocks (directly or through funds, as the source defines it)."""

    kind: Literal["bands", "survey_share", "unique_investors", "accounts"]
    value: float | None = None  # percent of adults (survey_share), or a count of people or accounts
    definition: str = ""
    as_of: str = ""
    adults: float | None = None  # adults the count is divided by (unique_investors, accounts)
    adults_source: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"


class Gradient(Provenance):
    """How ownership rises with household income: percent owning in each income band (shifted on the log-odds
    scale so the population average equals the country's share)."""

    bands: list[dict[str, float]] = Field(default_factory=list)  # pct_from, pct_to, own_share


class CountryInvestors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: IndexSpec
    ownership: Ownership
    gradient: Gradient | None = None  # None: the US SCF bands as they are (USA)
    holdings: Provenance  # band holdings; outside the US only their ratios matter (borrowed shape)
    attention_scale: Number
    multiplier: Number
    kappa: Provenance
    cross_checks: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    illustrative: str = ""  # why this market is only an illustration, beyond weak ownership data


class InvestorsByCountry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    countries: dict[str, CountryInvestors]


def load_everyday(data_dir: Path) -> Everyday | None:
    path = data_dir / "global" / EVERYDAY_FILE
    if not path.exists():
        return None
    return Everyday.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_by_country(data_dir: Path) -> InvestorsByCountry | None:
    path = data_dir / "global" / COUNTRY_FILE
    if not path.exists():
        return None
    return InvestorsByCountry.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------- resolution


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def shift_bands(bands: list[dict[str, float]], share_pct: float) -> list[float]:
    """Band ownership shares (percent) moved together on the log-odds scale until their population average
    (bands weighted by their width in income percentiles) equals ``share_pct``."""
    w = [(b["pct_to"] - b["pct_from"]) / 100.0 for b in bands]
    base = [_logit(b["own_share"] / 100.0) for b in bands]
    target = share_pct / 100.0
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        avg = sum(wk * _sigmoid(x + mid) for wk, x in zip(w, base, strict=True))
        lo, hi = (mid, hi) if avg < target else (lo, mid)
    d = 0.5 * (lo + hi)
    return [100.0 * _sigmoid(x + d) for x in base]


def adult_share(own: Ownership) -> float | None:
    """Percent of adults owning stock (None for the US bands, which are used as they are)."""
    if own.kind == "bands":
        return None
    if own.kind == "survey_share":
        assert own.value is not None
        return float(own.value)
    assert own.value is not None and own.adults
    return 100.0 * float(own.value) / float(own.adults)


def _us_holdings(us_bands: list[dict[str, float]], band: dict[str, float]) -> float:
    """US median holdings of the US band containing this band's middle (outside the US only the ratios
    between bands matter, so the US shape is borrowed)."""
    mid = 0.5 * (band["pct_from"] + band["pct_to"])
    for b in us_bands:
        if b["pct_from"] <= mid < b["pct_to"]:
            return float(b["holdings"])
    return float(us_bands[-1]["holdings"])


def market_cfg(iso: str, ci: CountryInvestors, us: InvestorsConfig, everyday: Everyday) -> dict[str, Any]:
    """The JSON-able copy one market keeps in the world (``macro["economy"]["markets"][ISO3]["cfg"]``)."""
    us_bands = [
        {"pct_from": b.pct_from, "pct_to": b.pct_to, "own_share": b.own_share, "holdings": b.median_holdings_usd}
        for b in us.ownership.bands
    ]
    share = adult_share(ci.ownership)
    if share is None:
        bands = [dict(b) for b in us_bands]
    else:
        shape = ci.gradient.bands if ci.gradient is not None and ci.gradient.bands else us_bands
        shares = shift_bands([dict(b) for b in shape], share)
        bands = [
            {"pct_from": b["pct_from"], "pct_to": b["pct_to"], "own_share": s, "holdings": _us_holdings(us_bands, b)}
            for b, s in zip(shape, shares, strict=True)
        ]
    for b in bands:
        b["holdings_usd"] = b.pop("holdings")  # read by investor_masks; outside the US only ratios matter
    kappa = us.trading.kappa.value
    reasons = []
    if ci.ownership.confidence == "low":
        reasons.append(f"who owns stock rests on a weak number ({ci.ownership.definition or ci.ownership.kind})")
    if ci.illustrative:
        reasons.append(ci.illustrative)
    borrowed = [
        name
        for name, v in (
            ("price impact", ci.multiplier),
            ("reaction size", ci.kappa),
            ("how often people look", ci.attention_scale),
            ("holdings", ci.holdings),
        )
        if v.borrowed_from
    ]
    return {
        "iso3": iso,
        "name": ci.index.name,
        "label": ci.index.label,
        "news_label": ci.index.label,
        "currency": ci.index.currency,
        "weekdays": list(ci.index.weekdays),
        "max_daily_pct": float(ci.index.max_daily_pct.value),
        "countries": [iso],
        "bands": bands,
        "adult_share": share,
        "multiplier": float(ci.multiplier.value),
        "kappa": float(kappa),
        "kappa_from": ci.kappa.label(),
        "requires_internet": us.trading.requires_internet,
        "attention_scale": float(ci.attention_scale.value),
        "borrowed": borrowed,
        "illustrative": bool(reasons),
        "reasons": reasons,
        "tech_beta": (us.derived["us_tech_stocks"].beta if iso == "USA" and "us_tech_stocks" in us.derived else None),
        "tech_label": (us.derived["us_tech_stocks"].label if iso == "USA" and "us_tech_stocks" in us.derived else None),
        "sources": sorted(
            {
                v
                for v in (
                    ci.ownership.source,
                    ci.ownership.adults_source,
                    None if ci.gradient is None else ci.gradient.source,
                    ci.holdings.source,
                    ci.multiplier.source,
                    ci.kappa.source,
                    ci.index.weekdays_source.source,
                    ci.index.max_daily_pct.source,
                )
                if v
            }
            | (set(us.source_ids()) if iso == "USA" else set())
        ),
    }


def everyday_cfg(everyday: Everyday) -> dict[str, Any]:
    a = everyday.attention
    s = everyday.saving
    return {
        "attention": {
            "pct": list(a.percentiles),
            "days": list(a.days),
            "of_days": a.of_days,
            "down_day_factor": a.down_day_factor.value,
            "weekend_factor": a.weekend_factor.value,
        },
        "mood": {
            "unemployed_shift": everyday.mood.unemployed_shift.value,
            "contagion": everyday.mood.contagion.value,
        },
        "saving": {
            "countries": list(s.countries),
            "participation": list(s.participation_by_wage_quartile.values),
            "contribution": s.contribution_share_of_pay.value,
            "equity_share": s.equity_share.value,
            "dc_share": s.dc_share_of_equity.value,
            "cashout_thresholds": list(s.cashout_by_balance.balance_usd),
            "cashout_shares": list(s.cashout_by_balance.cash_share),
            "hardship_rate": s.hardship_rate.value,
            "hardship_amount": s.hardship_amount_usd.value,
            "drawdown_rate": s.drawdown_rate.value,
        },
        "min_holders": everyday.min_holders.value,
        "sources": everyday.sources(),
    }


def economy_cfg(data_dir: Path, countries: list[str]) -> dict[str, Any] | None:
    """The world's own copy of everyday activity for these countries (None if the data files are missing)."""
    us = load_investors(data_dir)
    everyday = load_everyday(data_dir)
    by = load_by_country(data_dir)
    if us is None or everyday is None or by is None:
        return None
    markets = {iso: market_cfg(iso, by.countries[iso], us, everyday) for iso in countries if iso in by.countries}
    return {"version": 1, "everyday": everyday_cfg(everyday), "markets": markets}
