"""Per-command LLM budget: token caps always; a USD cap when prices are known.

Default prices are Anthropic's list prices for the default models as published on 2026-10-06 (Haiku 5.5's
rate applies to prompts up to 100K tokens, which covers every prompt Jahan sends). Prices change: check
them before relying on the dollar figures. Other models have no price, so only their tokens are capped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ModelPrice:
    usd_per_mtok_input: float | None = None
    usd_per_mtok_output: float | None = None


DEFAULT_PRICES = {
    "claude-haiku-5-5": ModelPrice(0.10, 0.50),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
}


@dataclass
class Budget:
    max_input_tokens: int = 2_000_000
    max_output_tokens: int = 400_000
    max_usd: float | None = None
    prices: dict[str, ModelPrice] = field(default_factory=dict)
    role_share: dict[str, float] = field(default_factory=dict)  # e.g. {"notable_autonomy": 0.3}
    used_input: int = 0
    used_output: int = 0
    used_usd: float = 0.0
    by_role: dict[str, float] = field(default_factory=dict)  # spend units (USD if priced, else tokens)
    unpriced_models: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> Budget:
        """Default caps and prices; JAHAN_MAX_USD sets a dollar cap per command."""
        cap = os.environ.get("JAHAN_MAX_USD")
        return cls(max_usd=float(cap) if cap else None, prices=dict(DEFAULT_PRICES))

    def _cost(self, model: str, tin: int, tout: int) -> float | None:
        p = self.prices.get(model)
        if p is None or p.usd_per_mtok_input is None or p.usd_per_mtok_output is None:
            return None
        return tin / 1e6 * p.usd_per_mtok_input + tout / 1e6 * p.usd_per_mtok_output

    def check(self, model: str, role: str, est_input: int, est_output: int) -> None:
        if self.used_input + est_input > self.max_input_tokens:
            raise BudgetExceeded(f"input-token cap {self.max_input_tokens:,} reached")
        if self.used_output + est_output > self.max_output_tokens:
            raise BudgetExceeded(f"output-token cap {self.max_output_tokens:,} reached")
        cost = self._cost(model, est_input, est_output)
        if self.max_usd is not None and cost is not None and self.used_usd + cost > self.max_usd:
            raise BudgetExceeded(f"USD cap ${self.max_usd:.2f} reached")
        share = self.role_share.get(role)
        if share is not None:
            cap_tokens = share * (self.max_input_tokens + self.max_output_tokens)
            if self.by_role.get(role, 0.0) + est_input + est_output > cap_tokens:
                raise BudgetExceeded(f"role '{role}' share of the budget ({share:.0%}) reached")

    def charge(self, model: str, role: str, tin: int, tout: int) -> None:
        self.used_input += tin
        self.used_output += tout
        cost = self._cost(model, tin, tout)
        if cost is not None:
            self.used_usd += cost
        else:
            self.unpriced_models.add(model)
        self.by_role[role] = self.by_role.get(role, 0.0) + tin + tout

    def summary(self) -> str:
        usd = f", about ${self.used_usd:.4f}" if self.prices else " (no prices configured: USD not tracked)"
        if self.unpriced_models:
            usd += f" (not counted: {', '.join(sorted(self.unpriced_models))} has no price)"
        return f"LLM use: {self.used_input:,} input / {self.used_output:,} output tokens{usd}"
