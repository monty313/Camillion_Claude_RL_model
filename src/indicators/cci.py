# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 made real) | WHO Claude for Monty
# WHY  CCI per period emits TWO lines: raw CCI + SMA(2)-shifted-4 of it.
# WHERE src/indicators/cci.py
# HOW  Real CCI via pandas (typical price, SMA, mean-deviation, /0.015);
#      auto-uses TA-Lib if installed. Post line = sma(raw, 2, shift=4).
# DEPENDS_ON: numpy, pandas, src/indicators/sma.py, (optional) talib
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: stub returned NaN. R: spec cci30 & cci100 raw+shifted.
#   A(Phase1): real CCI (pandas), TA-Lib fast-path. C: real cyclicity features;
#   zero-install on Colab.
# =====================================================================
"""CCI raw + SMA(2)-shifted-4. Pandas impl, optional TA-Lib fast-path."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.sma import sma

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def cci_raw(high, low, close, period: int) -> np.ndarray:
    """Raw CCI(period), length N float32 (NaN during warmup)."""
    high = np.asarray(high, dtype=np.float64).ravel()
    low = np.asarray(low, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    n = close.shape[0]
    if _HAS_TALIB and n > period:
        return np.asarray(talib.CCI(high, low, close, timeperiod=period), dtype=np.float32)
    tp = pd.Series((high + low + close) / 3.0)
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma) / (0.015 * md)
    return cci.to_numpy(dtype=np.float32)


def cci_post(high, low, close, period: int, post_sma: int = 2, post_shift: int = 4) -> np.ndarray:
    """SMA(post_sma) of CCI(period), shifted post_shift bars. Length N float32."""
    return sma(cci_raw(high, low, close, period), post_sma, shift=post_shift)
