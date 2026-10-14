"""Vectorised sampling helpers used by the world generator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class HasAgeRange(Protocol):
    age_min: float
    age_max: float


def row_index(ages: NDArray[np.float64], rows: Sequence[HasAgeRange]) -> NDArray[np.int64]:
    """Index of the first row whose [age_min, age_max) contains each age, or -1."""
    out = np.full(ages.shape[0], -1, dtype=np.int64)
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        out[(ages >= r.age_min) & (ages < r.age_max)] = i
    return out


def categorical_rows(
    rng: np.random.Generator, probs: NDArray[np.float64], row_of: NDArray[np.int64]
) -> NDArray[np.int64]:
    """For each element, draw a category from ``probs[row_of[i]]``.  Rows must sum to 1."""
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0
    u: NDArray[np.float64] = np.asarray(rng.random(int(row_of.shape[0])))
    return (u[:, None] >= cum[row_of]).sum(axis=1).astype(np.int64)


def categorical(rng: np.random.Generator, p: NDArray[np.float64], size: int) -> NDArray[np.int64]:
    return rng.choice(p.shape[0], size=size, p=p / p.sum()).astype(np.int64)


def sigmoid(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p: NDArray[np.float64] | float) -> NDArray[np.float64]:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))
