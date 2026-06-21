# =====================================================================
# WHEN 2026-06-21 (Phase 0; Phase 1 made NaN-aware) | WHO Claude for Monty
# WHY  SMA with optional bar SHIFT. Reused to post-smooth CCI/RSI, which carry
#      leading NaN during warmup -> the SMA must be NaN-aware (skip warmup,
#      stay valid afterwards) instead of propagating NaN forever.
# WHERE src/indicators/sma.py
# HOW  pandas rolling(period).mean() (NaN-aware via min_periods) then shift.
#      out[i] = SMA value from `shift` bars ago. float32, length N.
# DEPENDS_ON: numpy, pandas
# USED_BY: src/indicators/base.py, cci.py, rsi.py
# CHANGE_NOTES(IRAC): I: cumsum SMA turned post-smoothed CCI/RSI all-NaN. R:
#   spec SMA shifts + the CCI/RSI raw+shifted requirement. A(Phase1): pandas
#   rolling (NaN-aware) + shift. C: the shifted indicator lines now carry real
#   values, so the agent actually sees them.
# =====================================================================
"""SMA (raw, optional bar shift), NaN-aware so it can post-smooth indicators."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(values, period: int, shift: int = 0) -> np.ndarray:
    """Return SMA(period) of `values`, shifted forward by `shift` bars.

    NaN-aware: a window containing NaN (warmup of an upstream indicator) yields
    NaN, but values become valid once the window clears the NaN region. Always
    returns float32 of the same length as the input (bar-aligned).
    """
    s = pd.Series(np.asarray(values, dtype=np.float64).ravel())
    if period < 1:
        period = 1
    ma = s.rolling(window=period, min_periods=period).mean()
    if shift > 0:
        ma = ma.shift(shift)
    return ma.to_numpy(dtype=np.float32)
