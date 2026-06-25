# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  SMA Reversion Rally (rally, 5m/30m: 30m gravity, 5m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/sma_reversion_rally_5m_30m_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""SMA Reversion Rally (rally, 5m/30m: 30m gravity, 5m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class SmaReversionRally5m30mAlpha(BaseStrategy):
    name = "sma_reversion_rally_5m_30m"
    GRAV_TF, SIG_TF = "30m", "5m"
    TIMEFRAMES = ("5m", "30m")
    INDICATOR_COLUMNS = ['5m__sma_p1_s0', '5m__sma_p30_s0', '5m__sma_p50_s0', '5m__sma_p1_s1', '30m__sma_p1_s0', '30m__sma_p30_s0', '30m__sma_p50_s0', '30m__sma_p1_s1']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        g30 = self._v(ctx, "sma_p30_s0", self.GRAV_TF); g50 = self._v(ctx, "sma_p50_s0", self.GRAV_TF)
        sc = self._v(ctx, "sma_p1_s0", self.SIG_TF); s30 = self._v(ctx, "sma_p30_s0", self.SIG_TF); sprev = self._v(ctx, "sma_p1_s1", self.SIG_TF)
        if None in (g30, g50, sc, s30, sprev):
            return 0
        bull = g30 > g50 and sprev <= s30 and sc > s30   # rejoin up through sma30 in an up-regime
        bear = g30 < g50 and sprev >= s30 and sc < s30   # rejoin down through sma30 in a down-regime
        return 1 if bull else (-1 if bear else 0)
