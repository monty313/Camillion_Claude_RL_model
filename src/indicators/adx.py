# =====================================================================
# WHEN 2026-06-26 (alpha-private) | WHO Claude for Monty
# WHY  Wilder ADX(14): trend-STRENGTH (non-directional). Used ONLY by the
#      dual-movement-filter alphas to detect "the market is moving". This is an
#      ALPHA-PRIVATE indicator -- it is NOT added to the observation (the bot
#      consumes it only through the alpha's 1/0 signal), so the obs stays 479.
# WHERE src/indicators/adx.py
# HOW  Real Wilder ADX via pandas EWM (alpha=1/period) of +DM/-DM/TR -> DI ->
#      DX -> ADX; auto-uses TA-Lib ADX if installed. float32, length N (NaN warmup).
# DEPENDS_ON: numpy, pandas, (optional) talib
# USED_BY: src/indicators/base.py (compute_timeframe_alpha_private)
# CHANGE_NOTES(IRAC): I: a movement alpha needs trend strength, but ADX did not
#   exist and the obs is locked. R: operator 2026-06-26 "if we don't have to add
#   the extra indicator to the obs, don't -- we just need the signal". A: add ADX
#   as an alpha-PRIVATE indicator (read by alphas via ctx, excluded from the obs).
#   C: new movement signal with ZERO observation-contract change -> obs stays
#   stable on the road to 1000 alphas, exactly per the scaling rule.
# =====================================================================
"""ADX (Wilder) trend-strength. Pandas impl, optional TA-Lib fast-path. ALPHA-PRIVATE."""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def adx_raw(high, low, close, period: int = 14) -> np.ndarray:
    """Raw Wilder ADX(period), length N float32 (NaN during warmup).

    ADX measures trend STRENGTH only (0..100), never direction -- which is exactly
    what a non-directional movement filter wants.
    """
    high = np.asarray(high, dtype=np.float64).ravel()
    low = np.asarray(low, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    n = close.shape[0]
    if _HAS_TALIB and n > 2 * period:
        return np.asarray(talib.ADX(high, low, close, timeperiod=period), dtype=np.float32)
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up = h.diff()                       # high - prev_high
    down = -l.diff()                    # prev_low - low
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=c.index).ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=c.index).ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=a, adjust=False, min_periods=period).mean()
    return adx.to_numpy(dtype=np.float32)
