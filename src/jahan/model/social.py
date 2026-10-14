"""Configuration of the social-media sources (``data/feeds/social.yaml``): platforms, accounts per public
figure and channel, topic searches, and each platform's share of a country's internet users.

Reading the networks belongs to the ingest layer; this module has no network code, so data validation can
use it without importing that layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jahan.model.notables import load_roster

Platform = Literal["x", "bluesky", "mastodon", "telegram"]
PLATFORMS: tuple[str, ...] = ("x", "bluesky", "mastodon", "telegram")


class Account(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform: Platform
    handle: str
    followers: int | None = None
    instance: str | None = None  # Mastodon server
    name: str = ""
    lang: str | None = None
    note: str = ""
    own_query: bool = False  # X: search this account alone (it posts so often it would crowd out others)


class CountrySocial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    figures: dict[str, list[Account]] = Field(default_factory=dict)  # roster key -> accounts
    channels: list[Account] = Field(default_factory=list)  # institutions and news channels
    topics: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)  # lang -> topic -> platform -> query
    note: str = ""


class SourceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    kind: Literal["official", "unofficial", "public_preview"]
    search_url: str | None = None
    author_feed_url: str | None = None
    preview_url: str | None = None
    max_searches: int = 10
    note: str = ""


class CountryUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str  # registry id
    internet_users_m: float | None = None
    platforms_m: dict[str, float | None] = Field(default_factory=dict)


class PlatformUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_share: dict[str, float] = Field(default_factory=dict)
    countries: dict[str, CountryUsage] = Field(default_factory=dict)


class SocialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: dict[str, SourceCfg]
    lookback_days: int = Field(2, ge=1, le=14)
    platform_usage: PlatformUsage = Field(default_factory=PlatformUsage)
    countries: dict[str, CountrySocial] = Field(default_factory=dict)


def load_social(data_dir: Path) -> SocialConfig:
    path = data_dir / "feeds" / "social.yaml"
    with path.open(encoding="utf-8") as f:
        return SocialConfig.model_validate(yaml.safe_load(f))


def validate_social(data_dir: Path, countries: list[str]) -> list[str]:
    """Problems with social.yaml: unknown roster keys, accounts without a platform configuration."""
    try:
        cfg = load_social(data_dir)
    except Exception as e:  # noqa: BLE001 - reported as a data problem
        return [f"feeds/social.yaml: {e}"]
    problems = []
    for iso in countries:
        cs = cfg.countries.get(iso)
        if cs is None:
            continue
        roster = load_roster(data_dir, iso)
        keys = {e.key for e in roster.entries} if roster else set()
        for key, accounts in cs.figures.items():
            if key not in keys:
                problems.append(f"feeds/social.yaml: {iso} figure {key} is not in notables/{iso}.yaml")
            for a in accounts:
                if a.platform not in cfg.sources:
                    problems.append(f"feeds/social.yaml: {iso} {key} uses unconfigured platform {a.platform}")
        for a in cs.channels:
            if a.platform == "mastodon" and not a.instance:
                problems.append(f"feeds/social.yaml: {iso} Mastodon channel {a.handle} needs an instance")
            if a.lang is None:
                problems.append(f"feeds/social.yaml: {iso} channel {a.handle} needs a lang")
    return problems
