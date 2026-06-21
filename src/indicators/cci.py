# =====================================================================
# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  CCI for each period emits TWO lines: the raw CCI and a post-smoothed
#      line = SMA(2) of CCI, shifted 4 bars. (Per Monty's correction.)
# WHERE src/indicators/cci.py
# HOW  Phase 0: raw CCI returns NaN (length N) -- TA-Lib wired in Phase 1.
#      The post-smoothing (SMA2 + shift4) is already real and reused from
#      sma.py, so Phase 1 only needs to drop in talib.CCI for the raw line.
# DEPENDS_ON: numpy, src/indicators/sma.py
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: need cci30 & cci100, each raw+shifted. R: spec.
#   A: stub raw now, real post-smoothing now. C: locks the 4 column slots so
#   the 190-wide indicator block is correct from Phase 0.
# =====================================================================
"""CCI (period 30 & 100), each as raw + SMA(2)-shifted-4. Raw is a Phase-0 stub."""
from __future__ import annotations
import numpy as np
from src.indicators.sma import sma


def cci_raw(high, low, close, period: int) -> np.ndarray:
    """RAW CCI(period). PHASE-0 STUB -> NaN length N (wire talib.CCI in Phase 1)."""
    n = np.asarray(close).ravel().shape[0]
    return np.full(n, np.nan, dtype=np.float32)  # TODO(Phase 1): talib.CCI


def cci_post(high, low, close, period: int, post_sma: int = 2,
             post_shift: int = 4) -> np.ndarray:
    """SMA(post_sma) of CCI(period), shifted post_shift bars. Length N float32."""
    raw = cci_raw(high, low, close, period)
    return sma(raw, post_sma, shift=post_shift)
