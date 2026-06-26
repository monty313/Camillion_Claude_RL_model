# =====================================================================
# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  The read-only snapshot an alpha sees at one bar: current close +
#      indicator values keyed by "{tf}__{column}". The env builds this from
#      the cache each step and passes it to every strategy.
# WHERE src/strategies/context.py
# HOW  Small dataclass with ind(col, tf) lookup that returns NaN if missing.
# DEPENDS_ON: config/constants.py
# USED_BY: src/strategies/examples/*, src/env/trading_env.py (Phase 1)
# CHANGE_NOTES(IRAC): I: alphas need a uniform way to read indicators. R: spec
#   "each strategy is a signal generator". A: MarketContext snapshot. C: clean
#   inputs -> simple, auditable alphas the agent can weight.
# =====================================================================
"""MarketContext: the per-bar snapshot a strategy reads (close + indicators)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from config import constants as C


@dataclass
class MarketContext:
    close: float = float("nan")
    indicators: dict = field(default_factory=dict)   # {"{tf}__{col}": value}
    bar_index: int = -1
    symbol: str = ""
    minute_of_day: int = -1   # UTC minute-of-day of the bar close (0..1439); -1 = unknown

    def ind(self, col: str, tf: str = "1m") -> float:
        """Indicator value for column `col` on timeframe `tf` (NaN if absent)."""
        return float(self.indicators.get(f"{tf}__{col}", float("nan")))

    @staticmethod
    def is_valid(*vals: float) -> bool:
        """True only if every value is finite (i.e., past warmup)."""
        return all(v == v and not math.isinf(v) for v in vals)
