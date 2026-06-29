# =====================================================================
# WHEN 2026-06-29 (alpha pack) | WHO Claude for Monty
# WHY  Forward-displaced SMA(4) alignment (operator 2026-06-29): a directional read when price is ABOVE (or
#      BELOW) the FORWARD-DISPLACED SMA(4) of close on BOTH the 5m AND the 30m timeframe. The repo's cached
#      `sma_p4_s3` is SMA(4) of CLOSE displaced forward 3 bars on the chart (built from PAST data only —
#      leak-free; there is no future-leaking shift in this repo). Output +1/-1/0; pure function of ctx.
# WHERE src/strategies/fwd_sma4_align_5m_30m_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py
# CHANGE_NOTES(IRAC): I: operator wants action when price holds above/below the forward-shifted SMA(4) on 2+
#   TFs. R: "each strategy = a signal generator" (CLAUDE.md); no future data (leak-free). A: a per-slot alpha
#   comparing close to the cached sma_p4_s3 on 5m+30m. C: the bot SEES this trend-hold setup and the
#   alpha-shaping reward motivates trading WITH it -> trend-aligned entries toward consistent +2.5% days.
#   NOTE: the cached displacement is 3 bars (sma_p4_s3); a different forward shift would need new precompute.
# =====================================================================
"""Forward-displaced SMA(4) alignment (close above/below sma_p4_s3 on BOTH 5m and 30m -> directional)."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


class FwdSma4Align5m30mAlpha(BaseStrategy):
    name = "fwd_sma4_align_5m_30m"
    TIMEFRAMES = ("5m", "30m")
    INDICATOR_COLUMNS = ['5m__sma_p4_s3', '30m__sma_p4_s3']

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        px = ctx.close
        m5 = self._v(ctx, "sma_p4_s3", "5m")
        m30 = self._v(ctx, "sma_p4_s3", "30m")
        if px is None or math.isnan(px) or None in (m5, m30):
            return 0
        bull = px > m5 and px > m30                         # price above the fwd-SMA(4) on BOTH 5m and 30m
        bear = px < m5 and px < m30
        return 1 if bull else (-1 if bear else 0)
