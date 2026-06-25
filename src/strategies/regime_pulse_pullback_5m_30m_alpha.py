# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  Regime Pulse Tracker (pullback, 5m/30m: 30m gravity, 5m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/regime_pulse_pullback_5m_30m_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""Regime Pulse Tracker (pullback, 5m/30m: 30m gravity, 5m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class RegimePulsePullback5m30mAlpha(BaseStrategy):
    name = "regime_pulse_pullback_5m_30m"
    GRAV_TF, SIG_TF = "30m", "5m"
    TIMEFRAMES = ("5m", "30m")
    INDICATOR_COLUMNS = ['5m__sma_p1_s0', '5m__bb20_dev1.0_middle', '5m__bb200_dev1.0_middle', '30m__sma_p1_s0', '30m__bb20_dev1.0_middle', '30m__bb200_dev1.0_middle']

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
        bull = gc > g200 and gc > g20 and sc > s200 and sc < s20   # trigger dipped below BB20-mid
        bear = gc < g200 and gc < g20 and sc < s200 and sc > s20   # trigger popped above BB20-mid
        return 1 if bull else (-1 if bear else 0)
