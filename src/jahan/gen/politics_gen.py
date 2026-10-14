"""Political alignment: each person's lean on their country's main political axis (-1 .. +1).

+1 is the camp of the incumbent head of government at the roster date, -1 the main opposition camp
(labels in each country file's ``politics.axis``).  Camps are bands on the scale: above +1/3, below -1/3,
and neutral in between.  The band boundary only fixes where camps sit on the scale; how many adults fall
in each camp comes from the country's sourced camp shares.

Adults: a latent score ``z = sum of sourced demographic log-odds shifts + logistic noise`` is ranked and
mapped through the camp shares, so camp sizes match the data exactly while groups with a positive shift
sit higher on the scale.  Children take their mother's camp with probability 0.70 (parent-child party
agreement; Jennings, Stoker & Bowers 2009, registry ``global.alignment_inheritance``, provisional),
otherwise a fresh draw from the shares.

All draws use the stateless hash stream ``Stream.ALIGNMENT``, so no other generated attribute changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.gen.population_gen import edu_group_of
from jahan.gen.sampler import logit
from jahan.model.country import CAMP_TAU, CountrySpec, Politics
from jahan.rng import DOMAIN_GEN, Stream, u01

__all__ = ["CAMP_TAU", "alignment_words", "camp_of", "inherit_alignment", "quantile_alignment", "sample_alignment"]

PARENT_CHILD_SAME_CAMP = 0.70  # registry global.alignment_inheritance (provisional)

Arr = NDArray[Any]


def camp_of(alignment: Arr) -> NDArray[np.int8]:
    """+1, 0 or -1 per person."""
    a = np.asarray(alignment, dtype=np.float64)
    return np.where(a > CAMP_TAU, 1, np.where(a < -CAMP_TAU, -1, 0)).astype(np.int8)


INSET = 1e-4  # keep positions strictly inside their band, so float32 rounding never moves anyone across ±1/3


def _inset(x: Arr) -> NDArray[np.float64]:
    return INSET + (1.0 - 2.0 * INSET) * np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def quantile_alignment(u: Arr, politics: Politics | None) -> NDArray[np.float32]:
    """Map uniform u in [0,1) to the scale: neg camp on [-1,-tau), neutral on (-tau,tau), pos on (tau,1]."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0 - 1e-12)
    if politics is None:
        return np.zeros(u.shape, np.float32)
    neg, neu = politics.camps["neg"], politics.camps["neutral"]
    w = 1.0 - CAMP_TAU
    out = np.where(
        u < neg,
        -1.0 + w * _inset(u / max(neg, 1e-12)),
        np.where(
            u < neg + neu,
            -CAMP_TAU + 2 * CAMP_TAU * _inset((u - neg) / max(neu, 1e-12)),
            CAMP_TAU + w * _inset((u - neg - neu) / max(1.0 - neg - neu, 1e-12)),
        ),
    )
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _band(camp: Arr, u: Arr) -> NDArray[np.float32]:
    """A uniform position inside a camp's band."""
    w = 1.0 - CAMP_TAU
    pos = CAMP_TAU + w * _inset(u)
    return np.where(camp > 0, pos, np.where(camp < 0, -pos, -CAMP_TAU + 2 * CAMP_TAU * _inset(u))).astype(np.float32)


def inherit_alignment(parent: Arr, u_same: Arr, u_draw: Arr, politics: Politics | None) -> NDArray[np.float32]:
    """A child's lean: inside the parent's camp band with p = 0.70, else a fresh draw from the camp shares."""
    if politics is None:
        return np.zeros(np.shape(parent), np.float32)
    same = np.asarray(u_same) < PARENT_CHILD_SAME_CAMP
    return np.where(same, _band(camp_of(parent), np.asarray(u_draw)), quantile_alignment(u_draw, politics)).astype(
        np.float32
    )


def shift_logits(spec: CountrySpec, people: dict[str, Arr], age: Arr) -> NDArray[np.float64]:
    """Sum of the sourced log-odds shifts that apply to each person."""
    n = age.shape[0]
    z = np.zeros(n)
    if spec.politics is None:
        return z
    rel_codes = spec.religion_codes()
    eth_codes = spec.ethnicity_codes()
    edu = edu_group_of(people["education"])
    for s in spec.politics.shifts:
        if s.by == "religion":
            hit = people["religion"] == rel_codes.index(s.key)
        elif s.by == "ethnicity":
            hit = people["ethnicity"] == eth_codes.index(s.key)
        elif s.by == "education_group":
            hit = edu == ("low", "mid", "high").index(s.key)
        elif s.by == "region":
            hit = people["region"] == (1 if s.key == "urban" else 0)
        else:  # age_band
            lo, _, hi = s.key.partition("-")
            hit = (age >= int(lo)) & (age < int(hi))
        z += s.logit * hit
    return z


def sample_alignment(
    spec: CountrySpec, people: dict[str, Arr], age: Arr, mother: Arr, ids: Arr, seed: int
) -> NDArray[np.float32]:
    """Political lean of a freshly generated country population (``mother`` = local row index or -1)."""
    n = age.shape[0]
    out = np.zeros(n, np.float32)
    if spec.politics is None or n == 0:
        return out
    u0 = u01(seed, Stream.ALIGNMENT, 0, ids, k=0, domain=DOMAIN_GEN)
    u1 = u01(seed, Stream.ALIGNMENT, 0, ids, k=1, domain=DOMAIN_GEN)
    u2 = u01(seed, Stream.ALIGNMENT, 0, ids, k=2, domain=DOMAIN_GEN)
    adult = age >= 18
    z = shift_logits(spec, people, age) + logit(np.clip(u0, 1e-9, 1 - 1e-9))
    rows = np.flatnonzero(adult)
    if rows.size:
        order = np.argsort(z[rows], kind="stable")
        rank = np.empty(rows.size)
        rank[order] = np.arange(rows.size)
        out[rows] = quantile_alignment((rank + 0.5) / rows.size, spec.politics)
    kids = np.flatnonzero(~adult)
    with_mother = kids[mother[kids] >= 0]
    out[with_mother] = inherit_alignment(out[mother[with_mother]], u1[with_mother], u2[with_mother], spec.politics)
    orphans = kids[mother[kids] < 0]
    out[orphans] = quantile_alignment(u2[orphans], spec.politics)
    return out


def alignment_words(alignment: float, politics: Politics | None) -> str | None:
    """How a person's lean reads in a profile ("leans toward the Democratic camp"), or None without a block."""
    if politics is None:
        return None
    a = float(alignment)
    if abs(a) <= CAMP_TAU:
        words = "no strong political lean"
    else:
        label = politics.axis.pos if a > 0 else politics.axis.neg
        words = f"{'firmly in' if abs(a) >= 2 * CAMP_TAU else 'leans toward'} {label}"
    if politics.placeholder:
        words += " (camp shares for this country are a placeholder)"
    return words
