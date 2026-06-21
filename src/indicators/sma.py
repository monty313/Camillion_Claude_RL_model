# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Simple Moving Average with an optional bar SHIFT (value from N bars
#      ago). This is the ONE indicator implemented for real in Phase 0
#      (pure numpy, no TA-Lib) -- it is also reused to post-smooth CCI/RSI.
# WHERE src/indicators/sma.py
# HOW  cumsum trick -> O(N); leading values are NaN; shift moves the series
#      forward so out[i] == sma[i-shift]. Output is float32, length N.
# DEPENDS_ON: numpy
# USED_BY: src/indicators/base.py, cci.py, rsi.py
# CHANGE_NOTES(IRAC): I: need raw SMA incl. shifted lookbacks. R: spec SMA
#   p1/s0,p2/s1,p3/s2,p4/s3,p50/s0,p200/s0. A: numpy cumsum + shift. C: fast,
#   dependency-free, exact -> safe in the cached precompute step.
# =====================================================================
"""Simple Moving Average (raw, with optional bar shift). Pure numpy."""
from __future__ import annotations
import numpy as np


def sma(values, period: int, shift: int = 0) -> np.ndarray:
    """Return SMA(period) of `values`, shifted forward by `shift` bars.

    out[i] = mean(values[i-period+1 : i+1]) then shifted so out[i] uses the
    SMA value from `shift` bars ago. Leading undefined entries are NaN.
    Always returns float32 of the same length as the input (bar-aligned).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if period > 0 and n >= period:
        csum = np.cumsum(np.insert(v, 0, 0.0))
        ma = (csum[period:] - csum[:-period]) / float(period)  # len n-period+1
        out[period - 1:] = ma
    if shift > 0:
        shifted = np.full(n, np.nan, dtype=np.float64)
        if shift < n:
            shifted[shift:] = out[:n - shift]
        out = shifted
    return out.astype(np.float32)
