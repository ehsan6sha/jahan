"""Stateless, order-independent hash RNG.

Every random draw in the simulation is a pure function of
``(seed, stream, day, k, key)``.  Keys are stable integer ids (agent ids, edge
uids), never array positions.  Consequences:

* no RNG state needs to be saved in snapshots;
* phases can be vectorised and reordered without changing results;
* baseline and treated branches of an experiment share identical draws
  (common random numbers), which makes paired differences low-variance.

Canonical encoding (part of the reproducibility contract, pinned by
``tests/unit/test_rng_vectors.py``)::

    fields:  seed:u64  stream:u16  day:u32  k:u32  key:u64
    base  = mix(mix(seed ^ domain) ^ ((stream << 48) | day)) ^ (k * GOLDEN)
    value = mix(base ^ mix(key))

``mix`` is the splitmix64 finaliser.  ``u01`` uses the top 53 bits.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB

DOMAIN_SIM = 0x5157_4A41_4841_4E31  # simulation draws
DOMAIN_GEN = 0x4745_4E4A_4148_414E  # world-generation draws


class Stream(IntEnum):
    """Frozen registry of stream ids.  Never renumber; only append."""

    MEDIA = 1
    CONTACT = 2
    SHARE = 3
    ACCEPT = 4
    DECIDE = 5
    OUTCOME = 6
    DEATH = 7
    BIRTH = 8
    MARRY = 9
    DIVORCE = 10
    JOB = 11
    DISTORT = 12
    BOUNDED = 13
    TRAVEL = 14
    NOTABLE = 15
    SAMPLE = 16  # report sampling of narratives
    ACTION = 17  # delivery and acceptance of actions by human-controlled and notable agents
    AUTONOMY = 18  # tie-breaks among notable autonomy triggers
    ALIGNMENT = 19  # political alignment at generation (DOMAIN_GEN) and at birth
    SOCIAL = 20  # social media: reposts by ordinary people


def mix_int(x: int) -> int:
    """splitmix64 finaliser on a Python int (wraps to 64 bits)."""
    x &= MASK64
    x ^= x >> 30
    x = (x * _M1) & MASK64
    x ^= x >> 27
    x = (x * _M2) & MASK64
    x ^= x >> 31
    return x


def mix_np(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """splitmix64 finaliser, vectorised.  numpy uint64 arithmetic wraps silently."""
    x = np.array(x, dtype=np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= np.uint64(_M1)
    x ^= x >> np.uint64(27)
    x *= np.uint64(_M2)
    x ^= x >> np.uint64(31)
    return x


def base_key(seed: int, stream: int, day: int, k: int = 0, domain: int = DOMAIN_SIM) -> int:
    if not (0 <= stream < (1 << 16)):
        raise ValueError(f"stream out of range: {stream}")
    if not (0 <= day < (1 << 32)):
        raise ValueError(f"day out of range: {day}")
    if not (0 <= k < (1 << 32)):
        raise ValueError(f"k out of range: {k}")
    b = mix_int(mix_int((seed & MASK64) ^ domain) ^ ((stream << 48) | day))
    return b ^ ((k * GOLDEN) & MASK64)


def _keys_u64(keys: ArrayLike) -> NDArray[np.uint64]:
    arr = np.asarray(keys)
    if arr.dtype.kind == "i":
        if arr.size and arr.min() < 0:
            raise ValueError("RNG keys must be non-negative ids")
        return arr.astype(np.uint64)
    if arr.dtype.kind == "u":
        return arr.astype(np.uint64)
    raise TypeError(f"RNG keys must be integers, got {arr.dtype}")


def hash64(
    seed: int, stream: int, day: int, keys: ArrayLike, k: int = 0, domain: int = DOMAIN_SIM
) -> NDArray[np.uint64]:
    b = np.uint64(base_key(seed, stream, day, k, domain))
    return mix_np(mix_np(_keys_u64(keys)) ^ b)


def u01(seed: int, stream: int, day: int, keys: ArrayLike, k: int = 0, domain: int = DOMAIN_SIM) -> NDArray[np.float64]:
    """Uniform [0, 1) per key."""
    h = hash64(seed, stream, day, keys, k, domain)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def normal(
    seed: int, stream: int, day: int, keys: ArrayLike, k: int = 0, domain: int = DOMAIN_SIM
) -> NDArray[np.float64]:
    """Standard normal per key (Box-Muller over sub-draws 2k and 2k+1)."""
    a = u01(seed, stream, day, keys, 2 * k, domain)
    b = u01(seed, stream, day, keys, 2 * k + 1, domain)
    a = np.maximum(a, 1e-300)
    return np.sqrt(-2.0 * np.log(a)) * np.cos(2.0 * np.pi * b)


def derive_seed(seed: int, *parts: int) -> int:
    """Derive a child seed (e.g. experiment replicate s) from a parent seed."""
    x = seed & MASK64
    for p in parts:
        x = mix_int(x ^ mix_int(p & MASK64))
    return x
