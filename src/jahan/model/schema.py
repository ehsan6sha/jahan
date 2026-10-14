"""Column specifications for every SoA table.

Changing a column name, dtype or shape changes the snapshot format: bump
``SCHEMA_VERSION`` and refuse to load older snapshots.
"""

from __future__ import annotations

from jahan.model.enums import N_TOPICS, N_VALUES
from jahan.soa import Col

SCHEMA_VERSION = 6
MAX_LANGS = 3

AGENT_COLUMNS: dict[str, Col] = {
    # identity
    "id": Col("i8", default=-1),
    "alive": Col("?", default=True),
    "country": Col("i2"),  # country of residence (home); households and workplaces belong to it
    "birth_country": Col("i2", default=-1),  # differs from `country` for first-generation migrants
    "region": Col("i2"),  # 0 rural, 1 urban (M1)
    "household_id": Col("i4", default=-1),
    "first_name_id": Col("i4", default=-1),
    "last_name_id": Col("i4", default=-1),
    "pop_weight": Col("f4", default=0.0),
    # demographics
    "sex": Col("i1"),
    "birth_day": Col("i4"),  # simulation day of birth (negative = before world start)
    "ethnicity": Col("i2", default=-1),
    "religion": Col("i2", default=-1),
    "marital": Col("i1"),
    "spouse_id": Col("i8", default=-1),
    "mother_id": Col("i8", default=-1),
    "father_id": Col("i8", default=-1),
    "n_children": Col("i1"),
    "last_birth_day": Col("i4", default=-100000),
    "death_day": Col("i4", default=-1),
    # body
    "health": Col("f4", default=1.0),
    "vision_impaired": Col("i1"),
    "hearing_impaired": Col("i1"),
    "strength_z": Col("f4"),
    "attractiveness_z": Col("f4"),
    # mind
    "big5": Col("f4", (5,)),  # z-scores
    "risk_tolerance": Col("f4", default=0.5),
    "values": Col("f4", (N_VALUES,)),  # simplex weights
    "wants_kids": Col("f4", default=0.5),
    "alignment": Col("f4", default=0.0),  # lean on the country's main political axis, -1..1 (gen/politics_gen.py)
    "interests": Col("f4", (N_TOPICS,)),  # simplex weights
    "discernment": Col("f4", default=0.5),
    "honesty": Col("f4", default=0.8),
    "aggression": Col("f4", default=0.0),
    # upbringing
    "childhood_ses": Col("f4", default=0.5),
    "parent_edu_max": Col("i1", default=-1),
    "raised_by": Col("i1"),
    # capital
    "education": Col("i1"),
    "field": Col("i1", default=-1),
    "emp_status": Col("i1"),
    "occupation": Col("i1", default=-1),  # ISCO-08 major group 0..9
    "workplace_id": Col("i4", default=-1),
    "income_month": Col("f4"),  # local currency
    "wealth": Col("f4"),
    # information
    "info_tier": Col("i1"),
    "internet": Col("?", default=False),
    "focus_topic": Col("i1", default=-1),
    # language
    "lang": Col("i2", (MAX_LANGS,), default=-1),
    "lang_prof": Col("f4", (MAX_LANGS,)),
    # time
    "tz_offset_min": Col("i2"),
    "chronotype": Col("f4"),  # hours shift of the sleep window, + = later
    "work_shift": Col("i1"),
    # social
    "global_rep": Col("f4", default=0.5),
    "last_outcome_day": Col("i4", default=-1),
    "sentiment": Col("f4"),  # economic mood, roughly [-1, 1]; moved by news and macro shocks, decays
    # public role
    "visibility": Col("f4", default=1.0),  # log10 potential audience
    "is_notable": Col("?", default=False),
    "role_id": Col("i2", default=-1),
    "controlled_by": Col("i4", default=-1),
    # travel (M1 travel-lite): where the agent physically is today; differs from `country` only while travelling
    "location_country": Col("i2"),
    "home_tz_offset_min": Col("i2"),  # time zone at home; tz_offset_min follows the agent while travelling
    "trip_end_day": Col("i4", default=-1),  # -1 = at home
}

HOUSEHOLD_COLUMNS: dict[str, Col] = {
    "id": Col("i4", default=-1),
    "country": Col("i2"),
    "region": Col("i2"),
    "n_members": Col("i2"),
    "income_pooled": Col("f4"),
}

WORKPLACE_COLUMNS: dict[str, Col] = {
    "id": Col("i4", default=-1),
    "country": Col("i2"),
    "region": Col("i2"),
    "kind": Col("i1"),  # 0 firm, 1 school
    "size": Col("i4"),
}

EDGE_COLUMNS: dict[str, Col] = {
    "uid": Col("i8", default=-1),
    "src": Col("i8", default=-1),  # truster
    "dst": Col("i8", default=-1),  # trusted
    "kind": Col("i1"),
    "strength": Col("f4"),  # base daily contact propensity
    "prior_p": Col("f4", default=0.5),  # trust prior without halo
    "halo": Col("f4"),  # logit shift from attractiveness, fades with evidence
    "alpha": Col("f4"),  # evidence pseudo-counts (positive)
    "beta": Col("f4"),  # evidence pseudo-counts (negative)
    "n_out": Col("i2"),  # number of realised outcomes
    "last_update_day": Col("i4", default=-1),
    "created_day": Col("i4"),
    "last_contact_day": Col("i4", default=-1),
    "active": Col("?", default=True),
}

CHANNEL_COLUMNS: dict[str, Col] = {
    "id": Col("i4", default=-1),
    "country": Col("i2"),
    "kind": Col("i1"),
    "lang": Col("i2"),
    "reach_by_tier": Col("f4", (4,)),
    "credibility": Col("f4", default=0.6),
    "hours_mask_local": Col("u4"),  # broadcast hours as local-time bins
    "local_frame": Col("?", default=True),  # True: hours are in each viewer's local time (domestic channel)
    "tz_offset_min": Col("i2"),  # home time zone, used when local_frame is False
}

ITEM_COLUMNS: dict[str, Col] = {
    "id": Col("i4", default=-1),
    "created_day": Col("i4"),
    "release_utc_hour": Col("i1"),
    "topic": Col("i1"),
    "valence": Col("f4"),
    "truth": Col("f4", default=0.5),
    "salience": Col("f4", default=0.5),
    "credibility": Col("f4", default=0.5),
    "origin_kind": Col("i1"),
    "origin_id": Col("i8", default=-1),
    "entity_id": Col("i4", default=-1),
    "lang": Col("i2", default=-1),
    "active_until": Col("i4"),
    "slot": Col("i2", default=-1),  # tracked-belief column, -1 = ambient
    "content_hash": Col("S16", default=b""),
    "sentiment_weight": Col("f4"),  # how much awareness of this item moves economic sentiment
    "macro_key": Col("i4", default=-1),  # macro auto-news: hash of (variable, country), else -1
}

PRODUCT_COLUMNS: dict[str, Col] = {
    "id": Col("i4", default=-1),
    "topic": Col("i1"),
    "attributes": Col("f4", (N_VALUES,)),
    "true_quality": Col("f4", default=0.5),
    "item_id": Col("i4", default=-1),
    # per-country arrays are stored in World.product_price / product_available
}

PUSH_COLUMNS: dict[str, Col] = {
    "item_id": Col("i4"),
    "channel_id": Col("i4"),
    "start_day": Col("i4"),
    "end_day": Col("i4"),  # exclusive
    "intensity": Col("f4", default=1.0),
}
