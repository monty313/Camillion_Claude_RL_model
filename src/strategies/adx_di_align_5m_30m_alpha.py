# =====================================================================
# WHEN 2026-06-28 (contract v1.6.0) | WHO Claude for Monty
# WHY  ADX-DI Alignment (5m & 30m). Reads the ADX/DMI directional lines +DI and -DI
#      for periods 14 AND 45 on BOTH the 5m and 30m timeframes (4 DI pairs). Fires
#      ONLY when all four agree (operator 2026-06-28):
#        -DI ABOVE +DI on ALL four  -> SELL (-1)   (bears in control everywhere)
#        -DI BELOW +DI on ALL four  -> BUY  (+1)   (bulls in control everywhere)
#        otherwise / any value warming up -> 0 (inactive).
#      Output +1/-1/0 only; pure function of ctx (no state, no future data).
# WHERE src/strategies/adx_di_align_5m_30m_alpha.py
# HOW  The +DI/-DI come from the strategy-only DI side-channel (src/data/aux_features.py),
#      injected into ctx by the env -- NOT from the 220 indicator block, so the obs is
#      untouched. compute_signal just compares cached numbers (CLAUDE.md rule #3).
# DEPENDS_ON src/strategies/base.py, src/strategies/context.py, src/data/aux_features (DI columns)
# =====================================================================
"""ADX-DI Alignment (5m & 30m): -DI vs +DI agreement across periods 14&45 on both TFs."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy

_TFS = ("5m", "30m")
_PERIODS = (14, 45)
# the DI columns this alpha reads (must exist in the env's DI side-channel)
_DI_COLUMNS = [f"{tf}__{sign}_di{p}" for tf in _TFS for p in _PERIODS for sign in ("plus", "minus")]


class AdxDiAlign5m30mAlpha(BaseStrategy):
    name = "adx_di_align_5m_30m"
    TIMEFRAMES = _TFS
    PERIODS = _PERIODS
    INDICATOR_COLUMNS = _DI_COLUMNS

    def __init__(self, name: str | None = None):
        super().__init__(name or self.name)

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def compute_signal(self, ctx) -> int:
        all_bear = True   # every -DI strictly ABOVE its +DI -> SELL
        all_bull = True   # every -DI strictly BELOW its +DI -> BUY
        for tf in self.TIMEFRAMES:
            for p in self.PERIODS:
                plus = self._v(ctx, f"plus_di{p}", tf)
                minus = self._v(ctx, f"minus_di{p}", tf)
                if plus is None or minus is None:
                    return 0                      # warming up -> abstain
                all_bear = all_bear and (minus > plus)
                all_bull = all_bull and (minus < plus)
        return -1 if all_bear else (1 if all_bull else 0)
