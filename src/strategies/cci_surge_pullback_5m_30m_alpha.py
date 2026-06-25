# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  CCI Surge Sentinel (pullback, 5m/30m: 30m gravity, 5m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/cci_surge_pullback_5m_30m_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""CCI Surge Sentinel (pullback, 5m/30m: 30m gravity, 5m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class CciSurgePullback5m30mAlpha(BaseStrategy):
    name = "cci_surge_pullback_5m_30m"
    GRAV_TF, SIG_TF = "30m", "5m"
    TIMEFRAMES = ("5m", "30m")
    INDICATOR_COLUMNS = ['5m__cci30_raw', '5m__cci100_raw', '30m__cci30_raw', '30m__cci100_raw']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        g30 = self._v(ctx, "cci30_raw", self.GRAV_TF); g100 = self._v(ctx, "cci100_raw", self.GRAV_TF)
        s30 = self._v(ctx, "cci30_raw", self.SIG_TF); s100 = self._v(ctx, "cci100_raw", self.SIG_TF)
        if None in (g30, g100, s30, s100):
            return 0
        bull = g30 > 0 and g100 > 0 and s30 < 0 and s100 > 0
        bear = g30 < 0 and g100 < 0 and s30 > 0 and s100 < 0
        return 1 if bull else (-1 if bear else 0)
