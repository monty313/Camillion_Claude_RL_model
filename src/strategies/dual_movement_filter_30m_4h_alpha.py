# =====================================================================
# WHEN 2026-06-26 (alpha pack) | WHO Claude for Monty
# WHY  Dual Movement Filter (30m/4h). NON-DIRECTIONAL: a "is the market moving?"
#      finder, NOT a buy/sell signal. Emits 1 = movement / 0 = no movement. Never
#      -1. Pure function of ctx (no state/future). Higher-TF twin of the 5m/30m one.
# WHERE src/strategies/dual_movement_filter_30m_4h_alpha.py
# HOW  On BOTH timeframes, require ADX rising AND ATR rising, where "rising" =
#      value > its own value 5 bars ago (SMA(1) shift 5). ADX/ATR-shift come from
#      the ALPHA-PRIVATE indicator set (ctx only) so the observation never changes.
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py (alpha-private ADX/ATR)
# USED_BY: src/strategies/alpha_pack.register_all
# CHANGE_NOTES(IRAC): I: need a higher-TF movement filter without touching the obs.
#   R: operator 2026-06-26 "we just need the signal" + 1/0 only, ADX+ATR rising on
#   both TFs. A: read alpha-private adx/atr (raw vs 5-bars-ago) on 30m & 4h -> 1/0.
#   C: a slower-regime movement gate, added WITHOUT an obs-contract change.
# =====================================================================
"""Dual Movement Filter (30m/4h): 1 when ADX & ATR are rising on BOTH TFs, else 0."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class DualMovementFilter30m4hAlpha(BaseStrategy):
    name = "dual_movement_filter_30m_4h"
    DIRECTIONAL = False   # 1/0 movement GATE -- excluded from the directional consensus
    LOW_TF, HIGH_TF = "30m", "4h"
    TIMEFRAMES = ("30m", "4h")
    # adx14_raw, adx14_sma1sh5, atr14_raw, atr14_sma1sh5 on each TF (sh5 = value 5 bars ago).
    INDICATOR_COLUMNS = [
        "30m__adx14_raw", "30m__adx14_sma1sh5", "30m__atr14_raw", "30m__atr14_sma1sh5",
        "4h__adx14_raw", "4h__adx14_sma1sh5", "4h__atr14_raw", "4h__atr14_sma1sh5",
    ]

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def _is_moving(self, ctx, tf):
        """True if ADX rising AND ATR rising on `tf` (raw > value 5 bars ago). None if data missing."""
        adx = self._v(ctx, "adx14_raw", tf); adx5 = self._v(ctx, "adx14_sma1sh5", tf)
        atr = self._v(ctx, "atr14_raw", tf); atr5 = self._v(ctx, "atr14_sma1sh5", tf)
        if None in (adx, adx5, atr, atr5):
            return None
        return (adx > adx5) and (atr > atr5)

    def compute_signal(self, ctx) -> int:
        low = self._is_moving(ctx, self.LOW_TF)
        high = self._is_moving(ctx, self.HIGH_TF)
        if low is None or high is None:
            return 0
        return 1 if (low and high) else 0
