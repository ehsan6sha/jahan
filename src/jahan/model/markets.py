"""Market reactions to scheduled announcements (``data/global/markets.yaml``).

Not a market model: the simulated people do not trade and do not set prices.  On an announcement (a Fed
rate decision, a jobs report) a US stock index, the gold index and the euro's exchange rate move by published
event-study estimates times the surprise, the announced value minus what markets expected the day before.
Every number comes from the data file, which cites a registry entry for each estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

INDICATORS: tuple[str, ...] = ("us_policy_rate", "us_payrolls")
Indicator = Literal["us_policy_rate", "us_payrolls"]


class Effect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["us_stocks", "gold", "dollar"]
    pct: float  # published percent move ...
    per: float = Field(gt=0)  # ... per this much surprise, in the indicator's unit
    window: str
    source: str  # registry id
    quote: str = ""
    active: bool = True  # False: kept for comparison, never moves the world
    note: str = ""


class IndicatorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    country: str
    unit: str
    expected_hint: str = ""
    guidance_unit: str = ""
    surprise: list[Effect]
    guidance: list[Effect] = Field(default_factory=list)
    target_share_of_2y: float | None = None  # 2-year yield move per unit of target surprise (rate decisions)


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: Literal["us_stocks", "gold", "fx_local_per_usd"]
    countries: list[str] | None = None  # fx: the countries whose currency moves
    label: str


class MarketsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: dict[str, Target]
    indicators: dict[str, IndicatorCfg]

    def source_ids(self) -> list[str]:
        return sorted({e.source for ind in self.indicators.values() for e in ind.surprise + ind.guidance})


def load_markets(data_dir: Path) -> MarketsConfig | None:
    path = data_dir / "global" / "markets.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return MarketsConfig.model_validate(yaml.safe_load(f))


@dataclass(frozen=True)
class Move:
    """One estimate applied to one announcement."""

    target: str  # us_stocks | gold | dollar
    label: str
    term: str  # surprise | guidance
    size: float  # surprise in the indicator's unit (guidance: bp of the 2-year yield)
    pct: float  # percent move of the target
    window: str
    source: str
    active: bool
    note: str = ""


def market_moves(
    cfg: MarketsConfig, indicator: str, actual: float, expected: float, guidance_bp: float = 0.0
) -> list[Move]:
    """Every estimate applied to one announcement, inactive ones included (for display)."""
    ind = cfg.indicators[indicator]
    out: list[Move] = []
    for term, size, effects in (("surprise", actual - expected, ind.surprise), ("guidance", guidance_bp, ind.guidance)):
        for e in effects:
            out.append(
                Move(
                    target=e.target,
                    label=cfg.targets[e.target].label,
                    term=term,
                    size=float(size),
                    pct=float(e.pct * size / e.per),
                    window=e.window,
                    source=e.source,
                    active=e.active,
                    note=e.note,
                )
            )
    return out


def combined(moves: list[Move]) -> dict[str, float]:
    """Total percent move per target from the active estimates (terms add up)."""
    out: dict[str, float] = {}
    for m in moves:
        if m.active:
            out[m.target] = out.get(m.target, 0.0) + m.pct
    return out
