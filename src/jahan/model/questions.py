"""Watched questions (event odds): data files and validation.

``data/global/odds.yaml`` holds who trades on event questions and how their beliefs are aggregated;
``data/global/questions.yaml`` holds offline seed questions read from Polymarket on a stated date.  Both are
read by the command line when a question is added and copied into the world with it, never on load, so editing
them never changes or breaks an existing world.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

QUESTION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,199}$")


class TraderShare(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0, le=1)
    source: str
    adult_age: float = 18.0


class Skew(BaseModel):
    model_config = ConfigDict(extra="forbid")
    male_under_35: float = Field(gt=0)
    male_35_plus: float = Field(gt=0)
    female_under_35: float = Field(gt=0)
    female_35_plus: float = Field(gt=0)


class Aggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str
    source: str


class OddsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trader_share: TraderShare
    skew: Skew
    aggregation: Aggregation

    def source_ids(self) -> list[str]:
        return sorted({self.trader_share.source, self.aggregation.source})


class SeedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    event_slug: str = ""
    resolves_by: str | None = None
    price: float = Field(ge=0, le=1)
    topic: str = "world"
    countries: list[str] | None = None
    note: str = ""


class SeedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    fetch_date: str
    questions: list[SeedQuestion]


def load_odds_config(data_dir: Path) -> OddsConfig | None:
    path = data_dir / "global" / "odds.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return OddsConfig.model_validate(yaml.safe_load(f))


def load_question_seeds(data_dir: Path) -> SeedFile | None:
    path = data_dir / "global" / "questions.yaml"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return SeedFile.model_validate(yaml.safe_load(f))


def odds_problems(data_dir: Path, registry: dict[str, Any]) -> list[str]:
    """Both files parse; every source cites the registry; seed ids are valid question ids."""
    out: list[str] = []
    try:
        cfg = load_odds_config(data_dir)
        seeds = load_question_seeds(data_dir)
    except Exception as e:  # noqa: BLE001 - reported as a data problem
        return [f"global/odds.yaml or questions.yaml: {e}"]
    ids = (cfg.source_ids() if cfg else []) + ([seeds.source] if seeds else [])
    out += [f"global/odds.yaml: source id '{s}' not in registry" for s in ids if s not in registry]
    for q in seeds.questions if seeds else []:
        if not QUESTION_ID.fullmatch(q.id):
            out.append(f"global/questions.yaml: {q.id!r} is not a valid question id")
    return out
