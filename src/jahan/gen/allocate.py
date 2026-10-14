"""Split the agent budget across countries."""

from __future__ import annotations

import math


def allocate(n_agents: int, populations: list[float], method: str) -> list[int]:
    if not populations:
        return []
    if method == "equal":
        w = [1.0] * len(populations)
    elif method == "sqrt":
        w = [math.sqrt(p) for p in populations]
    elif method == "proportional":
        w = list(populations)
    else:
        raise ValueError(f"unknown allocation {method}")
    total = sum(w)
    raw = [n_agents * x / total for x in w]
    out = [int(math.floor(r)) for r in raw]
    # largest remainder, ties broken by country order (deterministic)
    rem = sorted(range(len(raw)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in rem[: n_agents - sum(out)]:
        out[i] += 1
    return out
