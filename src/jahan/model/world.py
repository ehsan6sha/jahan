"""The World: all state of one simulated timeline."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.config import RunConfig
from jahan.gen.names import NameTable
from jahan.model.country import CountrySpec
from jahan.model.markets import MarketsConfig
from jahan.model.schema import (
    AGENT_COLUMNS,
    CHANNEL_COLUMNS,
    EDGE_COLUMNS,
    HOUSEHOLD_COLUMNS,
    ITEM_COLUMNS,
    PRODUCT_COLUMNS,
    PUSH_COLUMNS,
    WORKPLACE_COLUMNS,
)
from jahan.soa import SoATable

MAX_PRODUCTS = 32


class BeliefStore:
    """Dense per-agent state for tracked info items (one column per slot)."""

    MATS: dict[str, tuple[str, float | int]] = {
        "aware": ("u1", 0),
        "belief": ("f4", 0.5),
        "heard_from": ("i8", -1),  # latest accepted adviser: agent id, or -(channel_id+2)
        "heard_val": ("f4", 0.5),  # value conveyed by that adviser
        "first_day": ("i4", -1),
        "adopted_day": ("i4", -1),
        "satisfaction": ("f4", -1.0),
    }

    def __init__(self, rows: int, slots: int) -> None:
        self.slots = slots
        self._rows = rows
        self.m: dict[str, NDArray[Any]] = {k: np.full((rows, slots), d, dtype=dt_) for k, (dt_, d) in self.MATS.items()}
        self.slot_item = np.full(slots, -1, dtype=np.int32)  # item id occupying each slot

    def ensure_rows(self, rows: int) -> None:
        if rows <= self._rows:
            return
        new_rows = max(rows, int(self._rows * 1.5) + 1)
        for k, (dt_, d) in self.MATS.items():
            a = np.full((new_rows, self.slots), d, dtype=dt_)
            a[: self._rows] = self.m[k]
            self.m[k] = a
        self._rows = new_rows

    def alloc_slot(self, item_id: int) -> int:
        free = np.flatnonzero(self.slot_item < 0)
        if free.size == 0:
            return -1
        s = int(free[0])
        self.slot_item[s] = item_id
        for k, (_dt, d) in self.MATS.items():
            self.m[k][:, s] = d
        return s

    def free_slot(self, slot: int) -> None:
        self.slot_item[slot] = -1

    def to_arrays(self, n: int) -> dict[str, NDArray[Any]]:
        out = {f"belief.{k}": np.ascontiguousarray(v[:n]) for k, v in self.m.items()}
        out["belief.slot_item"] = self.slot_item.copy()
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, NDArray[Any]], capacity: int) -> BeliefStore:
        slot_item = arrays["belief.slot_item"]
        bs = cls(capacity, int(slot_item.shape[0]))
        n = arrays["belief.aware"].shape[0]
        for k in cls.MATS:
            bs.m[k][:n] = arrays[f"belief.{k}"]
        bs.slot_item[:] = slot_item
        return bs


class ParasocialStore:
    """Trust of every agent in every notable (Beta evidence; the prior is computed from attributes)."""

    RESERVED = 16  # spare columns for figures created during a run (create-agent --notable)

    def __init__(self, rows: int, cols: int) -> None:
        self.alpha = np.zeros((rows, cols), np.float32)
        self.beta = np.zeros((rows, cols), np.float32)

    @property
    def cols(self) -> int:
        return int(self.alpha.shape[1])

    def ensure_rows(self, rows: int) -> None:
        if rows <= self.alpha.shape[0]:
            return
        new = max(rows, int(self.alpha.shape[0] * 1.5) + 1)
        for name in ("alpha", "beta"):
            a = np.zeros((new, self.cols), np.float32)
            a[: getattr(self, name).shape[0]] = getattr(self, name)
            setattr(self, name, a)


@dataclass
class World:
    config: RunConfig
    countries: list[CountrySpec]
    names: list[NameTable]
    languages: list[str]
    day: int = 0
    sim_seed: int = 0
    pop: SoATable = field(default_factory=lambda: SoATable(AGENT_COLUMNS))
    households: SoATable = field(default_factory=lambda: SoATable(HOUSEHOLD_COLUMNS))
    workplaces: SoATable = field(default_factory=lambda: SoATable(WORKPLACE_COLUMNS))
    edges: SoATable = field(default_factory=lambda: SoATable(EDGE_COLUMNS))
    channels: SoATable = field(default_factory=lambda: SoATable(CHANNEL_COLUMNS))
    items: SoATable = field(default_factory=lambda: SoATable(ITEM_COLUMNS))
    products: SoATable = field(default_factory=lambda: SoATable(PRODUCT_COLUMNS))
    pushes: SoATable = field(default_factory=lambda: SoATable(PUSH_COLUMNS))
    beliefs: BeliefStore = field(default_factory=lambda: BeliefStore(16, 8))
    product_price: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    product_available: NDArray[np.int32] = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    # USD list price per product and country (NaN = priced in local currency); local price follows the fx rate
    product_price_usd: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0), np.float64))
    # macro state (JSON-able): fx[c] local per USD, fx0[c] at world start, gold_index and us_stocks_index
    # (100 = world start; worlds made before the stock index have no key until a stock move happens),
    # last_auto_news {key: day}; "market" (worlds made since people trade: config copy and the two parts of
    # the stock index); "odds" (once a question is watched)
    macro: dict[str, Any] = field(default_factory=dict)
    # notable public figures: roster entries (JSON-able) + agent_id + country index; column j of ptrust
    notables: list[dict[str, Any]] = field(default_factory=list)
    ptrust: ParasocialStore = field(default_factory=lambda: ParasocialStore(16, ParasocialStore.RESERVED))
    roles: dict[str, Any] = field(default_factory=dict)  # role -> RoleSpec (data/roles.yaml)
    markets: MarketsConfig | None = None  # market reactions to announcements (data/global/markets.yaml)
    product_names: list[str] = field(default_factory=list)  # index = product id
    counters: dict[str, int] = field(
        default_factory=lambda: {"agent": 0, "edge": 0, "item": 0, "product": 0, "household": 0}
    )
    scheduled: list[dict[str, Any]] = field(default_factory=list)  # pending scenario events (JSON-able)
    channel_names: list[str] = field(default_factory=list)
    _edge_index: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None

    # ---------- helpers ----------
    @property
    def n_countries(self) -> int:
        return len(self.countries)

    def date(self, day: int | None = None) -> dt.date:
        return self.config.start_date + dt.timedelta(days=self.day if day is None else day)

    def dow(self, day: int | None = None) -> int:
        """0 = Monday."""
        return self.date(day).weekday()

    def ages(self, day: int | None = None) -> NDArray[np.float64]:
        d = self.day if day is None else day
        return (d - self.pop["birth_day"].astype(np.float64)) / 365.25

    def lang_id(self, code: str) -> int:
        try:
            return self.languages.index(code)
        except ValueError:
            return self.languages.index("other")

    def alloc_ids(self, kind: str, count: int) -> NDArray[np.int64]:
        start = self.counters[kind]
        self.counters[kind] = start + count
        return np.arange(start, start + count, dtype=np.int64)

    def agent_row(self, agent_id: int) -> int:
        """Agent ids equal row positions in M1 (ids are never reused and rows never deleted)."""
        ids = self.pop["id"]
        if 0 <= agent_id < ids.shape[0] and ids[agent_id] == agent_id:
            return int(agent_id)
        hit = np.flatnonzero(ids == agent_id)
        if hit.size == 0:
            raise KeyError(f"no agent {agent_id}")
        return int(hit[0])

    def ensure_product_arrays(self) -> None:
        if self.product_price.shape != (MAX_PRODUCTS, self.n_countries):
            price = np.zeros((MAX_PRODUCTS, self.n_countries), np.float32)
            avail = np.full((MAX_PRODUCTS, self.n_countries), -1, np.int32)
            r = min(self.product_price.shape[0], MAX_PRODUCTS)
            c = min(self.product_price.shape[1] if self.product_price.ndim == 2 else 0, self.n_countries)
            if self.product_price.size and c:
                price[:r, :c] = self.product_price[:r, :c]
                avail[:r, :c] = self.product_available[:r, :c]
            self.product_price, self.product_available = price, avail
        if self.product_price_usd.shape != (MAX_PRODUCTS, self.n_countries):
            usd = np.full((MAX_PRODUCTS, self.n_countries), np.nan, np.float64)
            if self.product_price_usd.ndim == 2 and self.product_price_usd.size:
                r = min(self.product_price_usd.shape[0], MAX_PRODUCTS)
                c = min(self.product_price_usd.shape[1], self.n_countries)
                usd[:r, :c] = self.product_price_usd[:r, :c]
            self.product_price_usd = usd

    def init_macro(self, market_cfg: dict[str, Any] | None = None) -> None:
        """Macro state of a new world.  ``market_cfg`` (from data/global/investors.yaml) turns on people's
        trading in US stocks; worlds made before it existed have no "market" key and never trade."""
        fx = [float(c.fx_local_per_usd) for c in self.countries]
        self.macro = {
            "fx": fx,
            "fx0": list(fx),
            "gold_index": 100.0,
            "us_stocks_index": 100.0,
            "sentiment_shift": [0.0] * len(fx),
            "last_auto_news": {},
        }
        if market_cfg is not None:
            self.macro["market"] = {
                "cfg": market_cfg,
                "people_index": 100.0,  # the part of the US stock index moved by people's trading
                "scripted_index": 100.0,  # the part moved by announcements and scripted paths
                "pending_pct": 0.0,  # weekend flow, traded on the next weekday
                "last": {
                    "day": -1,
                    "scripted_pct": 0.0,
                    "people_pct": 0.0,
                    "total_pct": 0.0,
                    "flow_usd": 0.0,
                    "holders": 0,
                    "traders": 0,
                },
            }

    def refresh_usd_prices(self) -> None:
        """Local prices of USD-priced products follow the current exchange rates."""
        fx = np.asarray(self.macro.get("fx", []), dtype=np.float64)
        if fx.size == 0 or self.product_price_usd.size == 0:
            return
        usd = self.product_price_usd
        m = ~np.isnan(usd)
        self.product_price[m] = (usd * fx[None, :])[m].astype(np.float32)

    # ---------- edge lookup ----------
    def invalidate_edge_index(self) -> None:
        self._edge_index = None

    def edge_index(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if self._edge_index is None:
            key = (self.edges["src"].astype(np.int64) << 32) | self.edges["dst"].astype(np.int64)
            order = np.argsort(key, kind="stable")
            self._edge_index = (key[order], order.astype(np.int64))
        return self._edge_index

    def append_edges(self, cols: dict[str, Any]) -> None:
        """Append edges and update the lookup index incrementally (no full re-sort)."""
        start = self.edges.n
        count = int(cols["uid"].shape[0])
        self.edges.append(count, **cols)
        if self._edge_index is None or count == 0:
            return
        keys, order = self._edge_index
        new_keys = (np.asarray(cols["src"], np.int64) << 32) | np.asarray(cols["dst"], np.int64)
        o = np.argsort(new_keys, kind="stable")
        new_keys = new_keys[o]
        new_rows = start + o.astype(np.int64)
        pos = np.searchsorted(keys, new_keys, side="right")
        self._edge_index = (np.insert(keys, pos, new_keys), np.insert(order, pos, new_rows))

    def find_edges(self, src: NDArray[np.int64], dst: NDArray[np.int64]) -> NDArray[np.int64]:
        """Row index of edge src->dst for each pair, or -1."""
        keys, order = self.edge_index()
        src = np.atleast_1d(np.asarray(src, dtype=np.int64))
        dst = np.atleast_1d(np.asarray(dst, dtype=np.int64))
        if keys.shape[0] == 0:
            return np.full(src.shape[0], -1, dtype=np.int64)
        q = (src << 32) | dst
        pos = np.searchsorted(keys, q)
        pos_c = np.minimum(pos, keys.shape[0] - 1)
        ok = keys[pos_c] == q
        return np.where(ok, order[pos_c], -1)
