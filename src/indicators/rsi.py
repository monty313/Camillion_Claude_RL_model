# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 made real) | WHO Claude for Monty
# WHY  RSI per period emits TWO lines: raw RSI + SMA(2)-shifted-2 of it.
# WHERE src/indicators/rsi.py
# HOW  Real Wilder RSI via pandas EWM (alpha=1/period); auto-uses TA-Lib if
#      installed. Post line = sma(raw, 2, shift=2). float32, length N.
# DEPENDS_ON: numpy, pandas, src/indicators/sma.py, (optional) talib
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: stub returned NaN. R: spec rsi4 & rsi14 raw+shifted.
#   A(Phase1): real Wilder RSI (pandas), TA-Lib fast-path. C: real momentum
#   features the agent can weight; runs on Colab with zero TA-Lib install.
# =====================================================================
"""RSI (Wilder) raw + SMA(2)-shifted-2. Pandas impl, optional TA-Lib fast-path."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.sma import sma

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def rsi_raw(close, period: int) -> np.ndarray:
    """Raw RSI(period), 0..100, length N float32 (NaN during warmup)."""
    close = np.asarray(close, dtype=np.float64).ravel()
    n = close.shape[0]
    if _HAS_TALIB and n > period:
        return np.asarray(talib.RSI(close, timeperiod=period), dtype=np.float32)
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask(avg_loss == 0.0, 100.0)   # all-gains window -> RSI 100
    rsi = rsi.mask(avg_gain == 0.0, 0.0)     # all-losses window -> RSI 0
    return rsi.to_numpy(dtype=np.float32)


def rsi_post(close, period: int, post_sma: int = 2, post_shift: int = 2) -> np.ndarray:
    """SMA(post_sma) of RSI(period), shifted post_shift bars. Length N float32."""
    return sma(rsi_raw(close, period), post_sma, shift=post_shift)
