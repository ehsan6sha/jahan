"""Who holds US stock and how much (``data/global/investors.yaml``), for the people-driven market (sim/market.py).

Ownership follows the Federal Reserve's Survey of Consumer Finances 2022: the share of families holding stock
(directly or through funds and retirement accounts) and their median holdings, by usual-income group.  Jahan
ranks US adults by their household's pooled income instead of ranking families (a stated bridge).  Who holds
stock is a lifetime draw per person (no stored column), so it is the same in every run and experiment branch.

The file is read once when a world is generated and copied into ``world.macro["market"]["cfg"]``; the engine
reads only that copy, so editing the file never changes or breaks an existing world.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from jahan.rng import Stream, u01

ADULT_AGE = 18.0


class Band(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    pct_from: float = Field(ge=0, le=100)
    pct_to: float = Field(ge=0, le=100)
    own_share: float = Field(ge=0, le=100)  # percent holding stock
    median_holdings_usd: float = Field(gt=0)


class CrossCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    what: str
    value: float


class Ownership(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    basis: str
    countries: list[str]
    bands: list[Band]
    cross_checks: list[CrossCheck] = Field(default_factory=list)


class Multiplier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0)
    source: str
    note: str = ""


class Kappa(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(ge=0)
    unit: str
    source: str
    rule: str
    fitted: dict[str, Any] | None = None
    superseded: list[dict[str, Any]] = Field(default_factory=list)  # earlier fits, kept with the reason
    # everyday activity (late reactions): its own pre-registered rule and fit (data/backtests/benchmarks.yaml)
    rule_v2: str | None = None
    fitted_v2: dict[str, Any] | None = None
    rule_v3: str | None = None  # news priced when it arrives (after the v2 run; see superseded)
    fitted_v3: dict[str, Any] | None = None


class Trading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    multiplier: Multiplier
    requires_internet: bool = True
    max_daily_pct: float = Field(gt=0)
    kappa: Kappa


class DerivedSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    of: str
    beta: float | None = Field(default=None, gt=0)
    method: str
    source: str
    label: str
    note: str = ""
    r2: float | None = None
    n: int | None = None
    range: str | None = None


class InvestorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ownership: Ownership
    trading: Trading
    derived: dict[str, DerivedSeries] = Field(default_factory=dict)

    def source_ids(self) -> list[str]:
        ids = {self.ownership.source, self.trading.multiplier.source, self.trading.kappa.source}
        ids |= {c.source for c in self.ownership.cross_checks}
        ids |= {d.source for d in self.derived.values()}
        return sorted(ids)

    def to_world(self) -> dict[str, Any]:
        """The JSON-able copy a new world keeps in ``macro["market"]["cfg"]``."""
        tech = self.derived.get("us_tech_stocks")
        return {
            "countries": list(self.ownership.countries),
            "bands": [
                {
                    "pct_from": b.pct_from,
                    "pct_to": b.pct_to,
                    "own_share": b.own_share,
                    "holdings_usd": b.median_holdings_usd,
                }
                for b in self.ownership.bands
            ],
            "multiplier": self.trading.multiplier.value,
            "kappa": self.trading.kappa.value,
            "requires_internet": self.trading.requires_internet,
            "max_daily_pct": self.trading.max_daily_pct,
            "tech_beta": None if tech is None else tech.beta,
            "tech_label": None if tech is None else tech.label,
            "sources": self.source_ids(),
        }


def load_investors(data_dir: Path) -> InvestorsConfig | None:
    path = data_dir / "global" / "investors.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return InvestorsConfig.model_validate(yaml.safe_load(f))


def band_problems(cfg: InvestorsConfig) -> list[str]:
    """Bands must cover 0-100 in order without gaps."""
    out = []
    edge = 0.0
    for b in cfg.ownership.bands:
        if b.pct_from != edge or b.pct_to <= b.pct_from:
            out.append(f"band {b.name!r} should start at {edge:g} and end above its start")
        edge = b.pct_to
    if edge != 100.0:
        out.append("bands must end at 100")
    return out


@dataclass
class Investors:
    holds: NDArray[np.bool_]  # US adults who hold stock (lifetime draw within their income band)
    trades: NDArray[np.bool_]  # holders who trade daily (with internet, if required)
    holdings: NDArray[np.float64]  # median holdings of their band, US dollars (0 for non-holders)
    band: NDArray[np.int8]  # income band index, -1 outside the population considered
    adults: NDArray[np.bool_]  # the population considered (residents of the countries in cfg, adult)


def eligible_adults(world: Any, countries: list[str]) -> NDArray[np.bool_]:
    """Alive adults living in ``countries``, with population weight, not public figures, not played by anyone."""
    pop = world.pop
    codes = [c.iso3 for c in world.countries]
    home = np.isin(pop["country"], [codes.index(c) for c in countries if c in codes])
    return (
        pop["alive"]
        & home
        & (pop["pop_weight"] > 0)
        & ~pop["is_notable"]
        & (pop["controlled_by"] < 0)
        & (world.ages() >= ADULT_AGE)
    )


def income_bands(world: Any, who: NDArray[np.bool_], bands: list[dict[str, Any]]) -> NDArray[np.int8]:
    """Band index per row (-1 outside ``who``) from the population-weighted percentile of household income."""
    pop = world.pop
    out = np.full(pop.n, -1, np.int8)
    rows = np.flatnonzero(who)
    if rows.size == 0:
        return out
    hh = pop["household_id"][rows]
    inc = pop["income_month"][rows].astype(np.float64)
    has_hh = (hh >= 0) & (hh < world.households.n)
    inc[has_hh] = world.households["income_pooled"][hh[has_hh]].astype(np.float64)
    w = pop["pop_weight"][rows].astype(np.float64)
    order = np.lexsort((pop["id"][rows], inc))  # stable: ties broken by agent id
    cum = np.cumsum(w[order])
    mid = 100.0 * (cum - 0.5 * w[order]) / cum[-1]  # midrank percentile
    pct = np.empty_like(mid)
    pct[order] = mid
    idx = np.zeros(rows.size, np.int8)
    for k, b in enumerate(bands):
        idx[pct >= float(b["pct_from"])] = k
    out[rows] = idx
    return out


def investor_masks(world: Any, cfg: dict[str, Any], requires_internet: bool | None = None) -> Investors:
    """Holders and daily traders among the world's people, from the world's own copy of the config."""
    pop = world.pop
    adults = eligible_adults(world, list(cfg["countries"]))
    band = income_bands(world, adults, cfg["bands"])
    share = np.array([float(b["own_share"]) / 100.0 for b in cfg["bands"]])
    usd = np.array([float(b["holdings_usd"]) for b in cfg["bands"]])
    u = u01(world.sim_seed, Stream.INVEST, 0, pop["id"])  # a lifetime draw: day fixed at 0
    b = np.maximum(band, 0)
    holds = adults & (u < share[b])
    need_net = bool(cfg["requires_internet"]) if requires_internet is None else requires_internet
    trades = holds & (pop["internet"] if need_net else True)
    holdings = np.where(holds, usd[b], 0.0)
    return Investors(holds=holds, trades=trades, holdings=holdings, band=band, adults=adults)


def ownership_check(world: Any, cfg: dict[str, Any]) -> dict[str, float]:
    """Population-weighted share of considered adults holding stock, overall and by education (Gallup check)."""
    inv = investor_masks(world, cfg)
    w = world.pop["pop_weight"].astype(np.float64)
    edu = world.pop["education"]

    def share(mask: NDArray[np.bool_]) -> float:
        m = inv.adults & mask
        tot = float(w[m].sum())
        return float(w[m & inv.holds].sum() / tot) if tot > 0 else float("nan")

    return {
        "adults": share(np.ones(world.pop.n, bool)),
        "college_graduates": share(edu >= 6),
        "high_school_or_less": share(edu <= 3),
        "trading_share_of_holders": float(w[inv.trades].sum() / w[inv.holds].sum()) if inv.holds.any() else 0.0,
    }
