"""Run configuration."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from jahan.sim.params import SimParams


class RunConfig(BaseModel):
    name: str = "world"
    seed: int = 20261013
    countries: list[str] = Field(default_factory=lambda: ["USA"])
    n_agents: int = 1000
    allocation: Literal["equal", "sqrt", "proportional"] = "sqrt"
    start_date: dt.date = Field(default_factory=dt.date.today)
    snapshot_every: int = 30
    max_slots: int = 128  # concurrently tracked info items
    diaspora_file: str | None = "generated/migration/stock_2024.yaml"  # relative to the data dir; None disables
    notables: bool = True  # insert public figures from data/notables (real names by default)
    params: SimParams = Field(default_factory=SimParams)

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
