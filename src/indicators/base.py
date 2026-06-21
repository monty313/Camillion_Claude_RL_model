# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  THE indicator registry: enumerates every indicator column (38/tf,
#      190 total) in a FIXED order and computes them per timeframe. This is
#      what locks the 190-wide raw-indicator block of the observation.
# WHERE src/indicators/base.py
# HOW  per_tf_columns() builds the 38 names from constants; compute_timeframe_
#      indicators() fills a (N,38) float32 matrix. SMA is real (numpy); CCI/
#      RSI/BB are NaN stubs until TA-Lib is wired in Phase 1. Length always
#      equals the input bar count (bar-aligned).
# DEPENDS_ON: config/constants.py, src/indicators/{sma,cci,rsi,bollinger}.py, numpy
# USED_BY: src/observation/builder.py (Phase 1), src/data/cache_builder.py,
#          tests/test_indicator_shapes.py
# CHANGE_NOTES(IRAC): I: 5 modules must agree on column order/count. R: spec
#   indicator list + Monty's CCI/RSI raw+shifted correction. A: one registry,
#   one order. C: a single locked order keeps the 190 block reproducible so
#   cached features and the live observation always line up.
# =====================================================================
"""Indicator registry: fixed column order + per-timeframe compute (190 total)."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.indicators.sma import sma
from src.indicators.cci import cci_raw, cci_post
from src.indicators.rsi import rsi_raw, rsi_post
from src.indicators.bollinger import bollinger
from src.indicators.atr import atr_raw, atr_post


def per_tf_columns() -> list[str]:
    """The 38 indicator column names for ONE timeframe, in canonical order."""
    cols: list[str] = []
    for period, shift in C.SMA_SPECS:                       # 6
        cols.append(f"sma_p{period}_s{shift}")
    for period in C.CCI_PERIODS:                            # 4
        cols.append(f"cci{period}_raw")
        cols.append(f"cci{period}_sma{C.CCI_POST_SMA}sh{C.CCI_POST_SHIFT}")
    for period in C.RSI_PERIODS:                            # 4
        cols.append(f"rsi{period}_raw")
        cols.append(f"rsi{period}_sma{C.RSI_POST_SMA}sh{C.RSI_POST_SHIFT}")
    for period in C.BOLLINGER_PERIODS:                      # 24
        for dev in C.BOLLINGER_DEVS:
            for band in C.BOLLINGER_BANDS:
                cols.append(f"bb{period}_dev{dev}_{band}")
    for period in C.ATR_PERIODS:                            # 2 (raw + shifted)
        cols.append(f"atr{period}_raw")
        cols.append(f"atr{period}_sma{C.ATR_POST_SMA}sh{C.ATR_POST_SHIFT}")
    return cols


PER_TF_COLUMNS: list[str] = per_tf_columns()
# Full 190-name list, prefixed by timeframe, in TIMEFRAMES order.
ALL_INDICATOR_COLUMNS: list[str] = [
    f"{tf}__{name}" for tf in C.TIMEFRAMES for name in PER_TF_COLUMNS
]


def compute_timeframe_indicators(high, low, close) -> np.ndarray:
    """Compute all 38 indicators for one timeframe -> (N, 38) float32.

    SMA columns are real; CCI/RSI/BB are NaN placeholders (Phase-0 stubs).
    Output length always equals the input bar count.
    """
    close = np.asarray(close, dtype=np.float64).ravel()
    high = np.asarray(high, dtype=np.float64).ravel() if high is not None else close
    low = np.asarray(low, dtype=np.float64).ravel() if low is not None else close
    n = close.shape[0]
    out = np.full((n, len(PER_TF_COLUMNS)), np.nan, dtype=np.float32)

    col = 0
    for period, shift in C.SMA_SPECS:
        out[:, col] = sma(close, period, shift); col += 1
    for period in C.CCI_PERIODS:
        out[:, col] = cci_raw(high, low, close, period); col += 1
        out[:, col] = cci_post(high, low, close, period); col += 1
    for period in C.RSI_PERIODS:
        out[:, col] = rsi_raw(close, period); col += 1
        out[:, col] = rsi_post(close, period); col += 1
    for period in C.BOLLINGER_PERIODS:
        for dev in C.BOLLINGER_DEVS:
            upper, middle, lower = bollinger(close, period, dev)
            out[:, col] = upper; col += 1
            out[:, col] = middle; col += 1
            out[:, col] = lower; col += 1
    for period in C.ATR_PERIODS:
        out[:, col] = atr_raw(high, low, close, period); col += 1
        out[:, col] = atr_post(high, low, close, period); col += 1
    assert col == len(PER_TF_COLUMNS), (col, len(PER_TF_COLUMNS))
    return out
