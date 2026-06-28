# =====================================================================
# WHEN 2026-06-28 (Phase 2) | WHO Claude for Monty
# WHY  ADX / DMI directional system: the +DI and -DI lines (and ADX trend
#      strength) per period. Two new alphas compare -DI vs +DI across periods
#      14 & 45 on several timeframes (operator request 2026-06-28). These lines
#      are NOT part of the locked 499 observation -- they feed ONLY the two
#      ADX-DI alphas via a strategy-only side-channel (see src/data/aux_features.py).
# WHERE src/indicators/adx.py
# HOW  Real Wilder DMI via pandas (EWM of +DM / -DM / TR, alpha=1/period, matching
#      src/indicators/atr.py's smoothing); auto-uses TA-Lib PLUS_DI/MINUS_DI/ADX if
#      installed. float32, length N, NaN during warmup (so an alpha abstains early).
# DEPENDS_ON: numpy, pandas, (optional) talib
# USED_BY: src/data/aux_features.py (the DI side-channel), jax_tpu parity tests
# CHANGE_NOTES(IRAC): I: operator wants two ADX-DI agreement alphas, but ADX/DI did
#   not exist in the repo. R: operator 2026-06-28 ("-DI above +DI = sell; below = buy").
#   A: add Wilder +DI/-DI (+ADX) here, fed to the alphas through a side-channel so the
#   499 observation shape is UNTOUCHED (CLAUDE.md rule #1). C: the bot gains a trend-
#   direction alpha family without any contract bump or retrain incompatibility.
# =====================================================================
"""ADX / DMI: Wilder +DI, -DI, ADX. Pandas impl, optional TA-Lib fast-path. NaN warmup."""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def _wilder(x: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing == pandas EWM(alpha=1/period, adjust=False, min_periods=period).
    Same convention as src/indicators/atr.py so the DI denominator (ATR) matches the repo."""
    return x.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _dmi(high, low, close, period: int):
    """Return (plus_di, minus_di) as float64 numpy arrays (NaN during warmup), Wilder method."""
    h = pd.Series(np.asarray(high, dtype=np.float64).ravel())
    l = pd.Series(np.asarray(low, dtype=np.float64).ravel())
    c = pd.Series(np.asarray(close, dtype=np.float64).ravel())
    up = h.diff()                # high[t]   - high[t-1]
    dn = -l.diff()              # low[t-1]  - low[t]
    plus_dm = np.where((up > dn) & (up > 0.0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, period)
    plus_di = 100.0 * _wilder(pd.Series(plus_dm, index=h.index), period) / atr
    minus_di = 100.0 * _wilder(pd.Series(minus_dm, index=h.index), period) / atr
    return plus_di.to_numpy(), minus_di.to_numpy()


def plus_di(high, low, close, period: int = 14) -> np.ndarray:
    """+DI(period): bullish directional movement, 0..100. float32, NaN during warmup."""
    n = np.asarray(close).ravel().shape[0]
    if _HAS_TALIB and n > period:
        return np.asarray(talib.PLUS_DI(np.asarray(high, np.float64).ravel(),
                                        np.asarray(low, np.float64).ravel(),
                                        np.asarray(close, np.float64).ravel(),
                                        timeperiod=period), dtype=np.float32)
    return _dmi(high, low, close, period)[0].astype(np.float32)


def minus_di(high, low, close, period: int = 14) -> np.ndarray:
    """-DI(period): bearish directional movement, 0..100. float32, NaN during warmup."""
    n = np.asarray(close).ravel().shape[0]
    if _HAS_TALIB and n > period:
        return np.asarray(talib.MINUS_DI(np.asarray(high, np.float64).ravel(),
                                         np.asarray(low, np.float64).ravel(),
                                         np.asarray(close, np.float64).ravel(),
                                         timeperiod=period), dtype=np.float32)
    return _dmi(high, low, close, period)[1].astype(np.float32)


def adx(high, low, close, period: int = 14) -> np.ndarray:
    """ADX(period): trend STRENGTH (0..100), Wilder-smoothed DX. float32, NaN during warmup.
    Not used by the current rule (which only compares -DI vs +DI) but provided for completeness
    and a possible future strength filter."""
    n = np.asarray(close).ravel().shape[0]
    if _HAS_TALIB and n > 2 * period:
        return np.asarray(talib.ADX(np.asarray(high, np.float64).ravel(),
                                    np.asarray(low, np.float64).ravel(),
                                    np.asarray(close, np.float64).ravel(),
                                    timeperiod=period), dtype=np.float32)
    p, m = _dmi(high, low, close, period)
    denom = p + m
    dx = 100.0 * np.abs(p - m) / np.where(denom == 0.0, np.nan, denom)
    return _wilder(pd.Series(dx), period).to_numpy().astype(np.float32)
