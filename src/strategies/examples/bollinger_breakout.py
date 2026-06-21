# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Breakout alpha: close beyond the 20/2.0 Bollinger band.
# WHERE src/strategies/examples/bollinger_breakout.py | HOW +1 if close>upper,
#       -1 if close<lower, else 0.
# DEPENDS_ON src/strategies/base.py, context.py | USED_BY registry/env, tests.
"""Bollinger breakout alpha (+1 above upper / -1 below lower / 0 inside)."""
from __future__ import annotations
from src.strategies.base import BaseStrategy
from src.strategies.context import MarketContext


class BollingerBreakoutStrategy(BaseStrategy):
    name = "bb20_2_breakout"

    def __init__(self, tf: str = "1m", period: int = 20, dev: float = 2.0):
        super().__init__(self.name)
        self.tf = tf
        self.upper_col = f"bb{period}_dev{dev}_upper"
        self.lower_col = f"bb{period}_dev{dev}_lower"

    def compute_signal(self, ctx: MarketContext) -> int:
        upper = ctx.ind(self.upper_col, self.tf)
        lower = ctx.ind(self.lower_col, self.tf)
        if not MarketContext.is_valid(upper, lower, ctx.close):
            return 0
        if ctx.close > upper:
            return 1
        if ctx.close < lower:
            return -1
        return 0
