# =====================================================================
# WHEN 2026-06-29 (alpha pack) | WHO Claude for Monty
# WHY  Bollinger DOUBLE-breakout, any timeframe (operator 2026-06-29): a directional read when price is
#      ABOVE BOTH the BB(200, dev1) AND BB(20, dev1) UPPER bands (long) — or BELOW both LOWER bands (short) —
#      on ANY ONE of the timeframes. A strong-trend/breakout confirmation. Output +1/-1/0; pure function of
#      the current ctx (no state, no future data). Both bands are already cached (BOLLINGER_PERIODS=(20,200),
#      dev 1.0), so this needs NO precompute.
# WHERE src/strategies/bb_double_breakout_anytf_alpha.py
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py, config/constants.py
# CHANGE_NOTES(IRAC): I: operator wants action when price clears BB200 & BB20 (dev1) on any TF. R: "each
#   strategy = a signal generator" (CLAUDE.md). A: a per-slot alpha comparing close to both bands per TF,
#   firing if ANY TF qualifies (no conflicting opposite TF). C: the bot SEES the breakout setup and the
#   alpha-shaping reward motivates trading it -> trend-aligned entries toward consistent +2.5% days.
# =====================================================================
"""Bollinger double-breakout (close beyond BOTH BB200(dev1) and BB20(dev1) on ANY timeframe -> directional)."""
from __future__ import annotations
import math
from config import constants as C
from src.strategies.base import BaseStrategy


class BbDoubleBreakoutAnyTfAlpha(BaseStrategy):
    name = "bb_double_breakout_anytf"
    TIMEFRAMES = C.TIMEFRAMES
    INDICATOR_COLUMNS = [f"{tf}__bb{p}_dev1.0_{b}"
                         for tf in C.TIMEFRAMES for p in (200, 20) for b in ("upper", "lower")]

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        px = ctx.close
        if px is None or math.isnan(px):
            return 0
        any_bull = any_bear = False
        for tf in self.TIMEFRAMES:
            up200 = self._v(ctx, "bb200_dev1.0_upper", tf); lo200 = self._v(ctx, "bb200_dev1.0_lower", tf)
            up20 = self._v(ctx, "bb20_dev1.0_upper", tf); lo20 = self._v(ctx, "bb20_dev1.0_lower", tf)
            if None not in (up200, up20) and px > up200 and px > up20:
                any_bull = True
            if None not in (lo200, lo20) and px < lo200 and px < lo20:
                any_bear = True
        if any_bull and not any_bear:
            return 1
        if any_bear and not any_bull:
            return -1
        return 0                                            # none, or conflicting TFs -> abstain
