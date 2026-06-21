# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Trend alpha: SMA50 vs SMA200 (golden/death cross style).
# WHERE src/strategies/examples/sma_trend.py | HOW +1 if SMA50>SMA200, -1 if <,
#       0 if within an epsilon band or during warmup.
# DEPENDS_ON src/strategies/base.py, context.py | USED_BY registry/env, tests.
"""SMA trend alpha (+1 fast>slow / -1 fast<slow / 0 inactive)."""
from __future__ import annotations
from src.strategies.base import BaseStrategy
from src.strategies.context import MarketContext


class SmaTrendStrategy(BaseStrategy):
    name = "sma_trend_50_200"

    def __init__(self, tf: str = "1m", eps: float = 0.0):
        super().__init__(self.name)
        self.tf = tf
        self.eps = eps

    def compute_signal(self, ctx: MarketContext) -> int:
        fast = ctx.ind("sma_p50_s0", self.tf)
        slow = ctx.ind("sma_p200_s0", self.tf)
        if not MarketContext.is_valid(fast, slow):
            return 0
        if fast > slow * (1 + self.eps):
            return 1
        if fast < slow * (1 - self.eps):
            return -1
        return 0
