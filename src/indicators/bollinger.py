# =====================================================================
# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Bollinger Bands: periods {20,200} x deviations {0.5,1,2,4} -> 8 configs,
#      each producing upper/middle/lower = 24 lines per timeframe.
# WHERE src/indicators/bollinger.py
# HOW  Phase 0: returns NaN arrays (length N) for all three bands. Middle is
#      just SMA(period) so Phase 1 can fill it from sma.py; upper/lower add
#      dev*rolling_std (TA-Lib BBANDS in Phase 1).
# DEPENDS_ON: numpy
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: need 24 BB columns/timeframe. R: spec. A: stub now,
#   lock the 24 slots. C: keeps the 190 indicator block shape correct.
# =====================================================================
"""Bollinger Bands (upper/middle/lower). Phase-0 stub returns NaN arrays."""
from __future__ import annotations
import numpy as np


def bollinger(values, period: int, dev: float):
    """Return (upper, middle, lower), each length N float32. PHASE-0 STUB -> NaN."""
    n = np.asarray(values).ravel().shape[0]
    nan = np.full(n, np.nan, dtype=np.float32)
    return nan.copy(), nan.copy(), nan.copy()  # TODO(Phase 1): talib.BBANDS
