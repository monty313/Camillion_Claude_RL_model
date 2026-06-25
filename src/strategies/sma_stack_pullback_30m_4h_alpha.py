# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  SMA Stack Prophet (pullback, 30m/4h: 4h gravity, 30m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/sma_stack_pullback_30m_4h_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""SMA Stack Prophet (pullback, 30m/4h: 4h gravity, 30m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class SmaStackPullback30m4hAlpha(BaseStrategy):
    name = "sma_stack_pullback_30m_4h"
    GRAV_TF, SIG_TF = "4h", "30m"
    TIMEFRAMES = ("30m", "4h")
    INDICATOR_COLUMNS = ['30m__sma_p1_s0', '30m__sma4_sh4_high', '30m__sma4_sh4_low', '4h__sma_p1_s0', '4h__sma4_sh4_high', '4h__sma4_sh4_low']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        gc = self._v(ctx, "sma_p1_s0", self.GRAV_TF); ghi = self._v(ctx, "sma4_sh4_high", self.GRAV_TF); glo = self._v(ctx, "sma4_sh4_low", self.GRAV_TF)
        sc = self._v(ctx, "sma_p1_s0", self.SIG_TF); shi = self._v(ctx, "sma4_sh4_high", self.SIG_TF); slo = self._v(ctx, "sma4_sh4_low", self.SIG_TF)
        if None in (gc, ghi, glo, sc, shi, slo):
            return 0
        bull = gc > ghi and gc > glo and sc < shi and sc < slo
        bear = gc < ghi and gc < glo and sc > shi and sc > slo
        return 1 if bull else (-1 if bear else 0)
