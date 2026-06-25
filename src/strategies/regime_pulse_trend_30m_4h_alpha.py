# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  Regime Pulse Tracker (trend, 30m/4h: 4h gravity, 30m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/regime_pulse_trend_30m_4h_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""Regime Pulse Tracker (trend, 30m/4h: 4h gravity, 30m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class RegimePulseTrend30m4hAlpha(BaseStrategy):
    name = "regime_pulse_trend_30m_4h"
    GRAV_TF, SIG_TF = "4h", "30m"
    TIMEFRAMES = ("30m", "4h")
    INDICATOR_COLUMNS = ['30m__sma_p1_s0', '30m__bb20_dev1.0_middle', '30m__bb200_dev1.0_middle', '4h__sma_p1_s0', '4h__bb20_dev1.0_middle', '4h__bb200_dev1.0_middle']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        gc = self._v(ctx, "sma_p1_s0", self.GRAV_TF); g20 = self._v(ctx, "bb20_dev1.0_middle", self.GRAV_TF); g200 = self._v(ctx, "bb200_dev1.0_middle", self.GRAV_TF)
        sc = self._v(ctx, "sma_p1_s0", self.SIG_TF); s20 = self._v(ctx, "bb20_dev1.0_middle", self.SIG_TF); s200 = self._v(ctx, "bb200_dev1.0_middle", self.SIG_TF)
        if None in (gc, g20, g200, sc, s20, s200):
            return 0
        bull = gc > g200 and gc > g20 and sc > s200 and sc > s20
        bear = gc < g200 and gc < g20 and sc < s200 and sc < s20
        return 1 if bull else (-1 if bear else 0)
