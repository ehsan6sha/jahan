"""Media channels from the country spec."""

from __future__ import annotations

from jahan.model.country import CountrySpec
from jahan.model.enums import CHANNEL_KIND_NAMES


def hours_mask(start: int, end: int) -> int:
    m = 0
    for h in range(start, end):
        m |= 1 << (h % 24)
    return m


def channel_rows(spec: CountrySpec, country_idx: int, lang_codes: list[str]) -> list[dict]:
    main_tz = max(spec.time_zones.rows, key=lambda r: r.share).offset_min
    rows = []
    for ch in spec.media.channels:
        if ch.kind not in CHANNEL_KIND_NAMES:
            raise ValueError(f"unknown channel kind {ch.kind}")
        rows.append(
            {
                "name": f"{spec.iso3}.{ch.name}",
                "country": country_idx,
                "kind": int(CHANNEL_KIND_NAMES[ch.kind]),
                "lang": lang_codes.index(ch.lang) if ch.lang in lang_codes else lang_codes.index("other"),
                "reach_by_tier": ch.reach_by_tier,
                "credibility": ch.credibility,
                "hours_mask_local": hours_mask(*ch.local_hours),
                "local_frame": True,
                "tz_offset_min": main_tz,
            }
        )
    return rows
