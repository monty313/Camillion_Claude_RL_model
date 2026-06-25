# =====================================================================
# WHEN 2026-06-22 (alpha pack v1.2.0) | WHO Claude for Monty
# WHY  CCI Surge Sentinel (trend, 30m/4h: 4h gravity, 30m trigger).  Higher TF = gravity, lower TF = trigger. Output +1/-1/0 only;
#      pure function of the current ctx (no state, no future data).
# WHERE src/strategies/cci_surge_trend_30m_4h_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# =====================================================================
"""CCI Surge Sentinel (trend, 30m/4h: 4h gravity, 30m trigger)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class CciSurgeTrend30m4hAlpha(BaseStrategy):
    name = "cci_surge_trend_30m_4h"
    GRAV_TF, SIG_TF = "4h", "30m"
    TIMEFRAMES = ("30m", "4h")
    INDICATOR_COLUMNS = ['30m__cci30_raw', '30m__cci100_raw', '4h__cci30_raw', '4h__cci100_raw']

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
        # (strong trend = both > +100 / < -100, inspectable; still outputs +/-1 only)
        bull = g30 > 0 and g100 > 0 and s30 > 0 and s100 > 0
        bear = g30 < 0 and g100 < 0 and s30 < 0 and s100 < 0
        return 1 if bull else (-1 if bear else 0)
