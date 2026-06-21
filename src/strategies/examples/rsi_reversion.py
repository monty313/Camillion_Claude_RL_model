# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Mean-reversion alpha: buy oversold, sell overbought (RSI14).
# WHERE src/strategies/examples/rsi_reversion.py | HOW +1 if RSI<lower, -1 if
#       RSI>upper, else 0.
# DEPENDS_ON src/strategies/base.py, context.py | USED_BY registry/env, tests.
"""RSI mean-reversion alpha (+1 oversold / -1 overbought / 0 inactive)."""
from __future__ import annotations
from src.strategies.base import BaseStrategy
from src.strategies.context import MarketContext


class RsiReversionStrategy(BaseStrategy):
    name = "rsi14_reversion"

    def __init__(self, tf: str = "1m", lower: float = 30.0, upper: float = 70.0):
        super().__init__(self.name)
        self.tf = tf
        self.lower = lower
        self.upper = upper

    def compute_signal(self, ctx: MarketContext) -> int:
        rsi = ctx.ind("rsi14_raw", self.tf)
        if not MarketContext.is_valid(rsi):
            return 0
        if rsi < self.lower:
            return 1
        if rsi > self.upper:
            return -1
        return 0
