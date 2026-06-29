# =====================================================================
# WHEN 2026-06-29 (alpha pack) | WHO Claude for Monty
# WHY  CCI extreme-extension alignment (operator 2026-06-29): a strong directional read when CCI(30) is
#      beyond the +/-160 level on BOTH the 5m AND the 30m timeframe. Unlike cci_surge (which only checks the
#      SIGN of cci30/cci100), this fires only on EXTREME extension (|CCI| > 160) confirmed on two timeframes.
#      Output +1/-1/0 only; pure function of the current ctx (no state, no future data).
# WHERE src/strategies/cci_x160_align_5m_30m_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# CHANGE_NOTES(IRAC): I: operator wants the bot to recognise/act on strong CCI extension on 5m+30m. R:
#   "each strategy = a signal generator" (CLAUDE.md). A: a per-slot alpha reading cached cci30 on 5m+30m,
#   gated at +/-160. C: the bot SEES this setup (alpha obs blocks) and the alpha-shaping reward motivates
#   trading WITH it / BEATING it -> better, higher-conviction entries toward consistent +2.5% days.
# =====================================================================
"""CCI extreme-extension alignment (|cci30| > 160 on BOTH 5m and 30m -> directional)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy

CCI_LEVEL = 160.0   # operator: the "160 level" — extreme extension (well beyond the classic +/-100)


class CciX160Align5m30mAlpha(BaseStrategy):
    name = "cci_x160_align_5m_30m"
    TIMEFRAMES = ("5m", "30m")
    INDICATOR_COLUMNS = ['5m__cci30_raw', '30m__cci30_raw']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        c5 = self._v(ctx, "cci30_raw", "5m")
        c30 = self._v(ctx, "cci30_raw", "30m")
        if None in (c5, c30):
            return 0
        bull = c5 > CCI_LEVEL and c30 > CCI_LEVEL          # both 5m AND 30m extended ABOVE +160
        bear = c5 < -CCI_LEVEL and c30 < -CCI_LEVEL        # both extended BELOW -160
        return 1 if bull else (-1 if bear else 0)
