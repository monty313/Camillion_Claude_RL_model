# =====================================================================
# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  RSI for each period emits TWO lines: the raw RSI and a post-smoothed
#      line = SMA(2) of RSI, shifted 2 bars. (Per Monty's correction.)
# WHERE src/indicators/rsi.py
# HOW  Phase 0: raw RSI returns NaN (length N); post-smoothing is real.
# DEPENDS_ON: numpy, src/indicators/sma.py
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: need rsi4 & rsi14, each raw+shifted. R: spec.
#   A: stub raw now, real post-smoothing now. C: locks the 4 RSI column slots.
# =====================================================================
"""RSI (period 4 & 14), each as raw + SMA(2)-shifted-2. Raw is a Phase-0 stub."""
from __future__ import annotations
import numpy as np
from src.indicators.sma import sma


def rsi_raw(close, period: int) -> np.ndarray:
    """RAW RSI(period). PHASE-0 STUB -> NaN length N (wire talib.RSI in Phase 1)."""
    n = np.asarray(close).ravel().shape[0]
    return np.full(n, np.nan, dtype=np.float32)  # TODO(Phase 1): talib.RSI


def rsi_post(close, period: int, post_sma: int = 2,
             post_shift: int = 2) -> np.ndarray:
    """SMA(post_sma) of RSI(period), shifted post_shift bars. Length N float32."""
    raw = rsi_raw(close, period)
    return sma(raw, post_sma, shift=post_shift)
