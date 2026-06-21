# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 made real) | WHO Claude for Monty
# WHY  Bollinger Bands: periods {20,200} x devs {0.5,1,2,4} -> upper/middle/lower.
# WHERE src/indicators/bollinger.py
# HOW  Real bands via pandas (SMA +/- dev * population std, ddof=0 to match
#      TA-Lib); auto-uses TA-Lib BBANDS if installed.
# DEPENDS_ON: numpy, pandas, (optional) talib
# USED_BY: src/indicators/base.py
# CHANGE_NOTES(IRAC): I: stub returned NaN. R: spec 8 BB configs/timeframe.
#   A(Phase1): real bands (pandas, ddof=0), TA-Lib fast-path. C: real
#   volatility envelopes; zero-install on Colab.
# =====================================================================
"""Bollinger Bands (upper/middle/lower). Pandas impl, optional TA-Lib fast-path."""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def bollinger(values, period: int, dev: float):
    """Return (upper, middle, lower), each length N float32 (NaN during warmup)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.shape[0]
    if _HAS_TALIB and n > period:
        u, m, l = talib.BBANDS(v, timeperiod=period, nbdevup=dev, nbdevdn=dev, matype=0)
        return (np.asarray(u, np.float32), np.asarray(m, np.float32), np.asarray(l, np.float32))
    s = pd.Series(v)
    m = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)            # population std -> matches TA-Lib
    upper = (m + dev * sd).to_numpy(dtype=np.float32)
    middle = m.to_numpy(dtype=np.float32)
    lower = (m - dev * sd).to_numpy(dtype=np.float32)
    return upper, middle, lower
