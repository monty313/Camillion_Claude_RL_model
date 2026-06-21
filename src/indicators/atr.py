# =====================================================================
# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  ATR(14) per timeframe emits TWO lines: raw ATR + SMA(2)-shifted-4 of it.
#      The shifted line lets the bot compare volatility NOW vs ~4 bars ago.
# WHERE src/indicators/atr.py
# HOW  Real Wilder ATR via pandas (EWM of True Range, alpha=1/period); auto-uses
#      TA-Lib ATR if installed. Post line = sma(raw, 2, shift=4). float32, len N.
# DEPENDS_ON: numpy, pandas, src/indicators/sma.py, (optional) talib
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: Monty added ATR14 raw+shifted to each TF. R: operator
#   request 2026-06-21 + the established CCI/RSI raw+shifted pattern. A: real
#   Wilder ATR (pandas), TA-Lib fast-path, +2 cols/TF (obs 357->367, v1.1.0).
#   C: volatility context + its 4-bar slope helps size/avoid trades near
#   FTMO drawdown walls.
# =====================================================================
"""ATR (Wilder) raw + SMA(2)-shifted-4. Pandas impl, optional TA-Lib fast-path."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.sma import sma

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def atr_raw(high, low, close, period: int = 14) -> np.ndarray:
    """Raw ATR(period) (Wilder), length N float32 (NaN during warmup)."""
    high = np.asarray(high, dtype=np.float64).ravel()
    low = np.asarray(low, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    n = close.shape[0]
    if _HAS_TALIB and n > period:
        return np.asarray(talib.ATR(high, low, close, timeperiod=period), dtype=np.float32)
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr.to_numpy(dtype=np.float32)


def atr_post(high, low, close, period: int = 14, post_sma: int = 2, post_shift: int = 4) -> np.ndarray:
    """SMA(post_sma) of ATR(period), shifted post_shift bars. Length N float32."""
    return sma(atr_raw(high, low, close, period), post_sma, shift=post_shift)
