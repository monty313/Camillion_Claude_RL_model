# =====================================================================
# WHEN 2026-06-25 | WHO Claude for Monty
# WHY  Opening-Range Breakout (ORB) at the New York open -- INDICES ONLY. The 4h before NY open
#      forms the opening range; a break of that range during the NY entry window (the most-liquid
#      part of the day), filtered by the 200-trend, fires +1/-1. Output +1/-1/0 only.
# WHERE src/strategies/orb_ny_breakout_indices_alpha.py
# HOW  Stateful + LEAK-FREE: accumulates the opening range from CLOSED bars before NY open (high/low
#      approximated by close, since the env carries close only), freezes it at the open, then breaks
#      out during the session. Resets each new UTC day. Index filter via asset_specs classifier
#      (covers the full FTMO index list). Trend = 30m BB200 middle (= SMA200; repo has no 15m TF).
# DEPENDS_ON: src/strategies/base.py, config/asset_specs.py | USED_BY: alpha_pack.register_all
# CHANGE_NOTES(IRAC): I: indices need a session-breakout alpha. R: operator ORB spec (adapted to
#   the repo: no 15m TF -> 1m opening range + 30m SMA200; close-as-high/low). A: stateful per-day
#   ORB + NY window + trend filter, index-only. C: a high-liquidity index entry signal the policy
#   can weight (its profit flows through equity; the NY index bonus rewards banking it).
# =====================================================================
"""ORB New-York-open breakout for INDEX instruments (stateful, leak-free)."""
from __future__ import annotations
import math
from config import asset_specs as A
from src.strategies.base import BaseStrategy

# UTC minute-of-day windows. New York open = 13:30 UTC.
_PRE_START, _PRE_END = 9 * 60 + 30, 13 * 60 + 30     # 09:30-13:30 = the 4h opening-range window
_SESS_START, _SESS_END = 13 * 60 + 30, 15 * 60 + 30   # 13:30-15:30 = breakout ENTRY window (2h)


class OrbNyBreakoutIndicesAlpha(BaseStrategy):
    """Opening-Range Breakout at the NY open, INDEX symbols only.

    Range = high/low (approx by close) over the 4h before 13:30 UTC. During 13:30-15:30 UTC:
    +1 if close breaks ABOVE the range AND is above the 200-trend; -1 if it breaks BELOW AND is
    below it; 0 otherwise (inside the range, against trend, or outside the window/instrument).
    The signal persists while price stays beyond the broken level + the trend holds, and resets to
    0 when price closes back inside the range. Stateful per UTC day; reset() clears it."""
    name = "orb_ny_breakout_indices"
    TREND_TF = "30m"                                  # 30m BB200 middle = the long-trend (SMA200) filter
    TIMEFRAMES = ("30m",)
    INDICATOR_COLUMNS = ["30m__bb200_dev1.0_middle"]

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)
        self.reset()

    def reset(self) -> None:
        self._orb_high = None
        self._orb_low = None
        self._prev_mod = -1

    @staticmethod
    def _finite(x) -> bool:
        return x == x and not math.isinf(x)

    def compute_signal(self, ctx) -> int:
        mod = int(getattr(ctx, "minute_of_day", -1))
        if mod < 0:                                   # no clock -> cannot run
            return 0
        if mod < self._prev_mod:                      # crossed UTC midnight -> new day, fresh range
            self._orb_high = self._orb_low = None
        self._prev_mod = mod
        if A.asset_class(ctx.symbol) != "index":      # INDICES only
            return 0
        close = float(ctx.close)
        if not self._finite(close):
            return 0
        # build the opening range over the 4h BEFORE the NY open (high/low approximated by close)
        if _PRE_START <= mod < _PRE_END:
            self._orb_high = close if self._orb_high is None else max(self._orb_high, close)
            self._orb_low = close if self._orb_low is None else min(self._orb_low, close)
            return 0
        # breakout during the NY entry window, filtered by the 200-trend
        if _SESS_START <= mod < _SESS_END and self._orb_high is not None:
            sma200 = ctx.ind("bb200_dev1.0_middle", self.TREND_TF)
            if not self._finite(sma200):
                return 0
            if close > self._orb_high and close > sma200:
                return 1
            if close < self._orb_low and close < sma200:
                return -1
            return 0
        return 0
