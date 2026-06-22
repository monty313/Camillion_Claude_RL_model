# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 real, leakage-safe) | WHO Claude
# WHY  Precompute every indicator ONCE and align all timeframes onto the 1m
#      timeline with NO future leakage, then store float32 for fast env reads.
# WHERE src/data/cache_builder.py
# HOW  Resample 1m -> higher TFs; compute indicators per TF; align by taking,
#      at each 1m bar, the LAST HIGHER-TF BAR THAT HAS CLOSED (close_time <=
#      this 1m bar's close_time). The in-progress higher-TF bar is never used.
#      env.step() then only reads cached float32 (no TA-Lib/MT5/pandas).
# DEPENDS_ON: config/constants.py, src/indicators/base.py, numpy, pandas
# USED_BY: src/env/trading_env.py (Phase 1), tests.
# CHANGE_NOTES(IRAC): I: multi-TF alignment is the #1 leakage trap. R: operator
#   leakage discipline 2026-06-21. A: 'last closed bar' alignment via searchsorted
#   on close_times; proven leak-free in tests. C: honest features -> honest
#   FTMO pass estimate; cache makes training CPU-fast.
# =====================================================================
"""Leakage-safe multi-timeframe indicator cache (precompute once, read float32)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from config import constants as C
from src.indicators import base

_TF_RULE = {"1m": "1min", "5m": "5min", "30m": "30min", "4h": "4h", "1d": "1D"}
_TF_MIN = {"1m": 1, "5m": 5, "30m": 30, "4h": 240, "1d": 1440}
_NS_PER_MIN = np.int64(60_000_000_000)


def resample_ohlcv(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 1m OHLCV to `tf` (bar index = OPEN time; close = open + tf)."""
    if tf == "1m":
        return df1m
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    use = {c: agg[c] for c in agg if c in df1m.columns}
    return df1m.resample(_TF_RULE[tf], label="left", closed="left").agg(use).dropna(how="any")


def _align_to_1m(ind_vals, tf_open_ns, tf_min, close1m_ns):
    """At each 1m close, take the latest TF bar whose CLOSE <= that 1m close."""
    close_tf_ns = tf_open_ns + np.int64(tf_min) * _NS_PER_MIN
    order = np.argsort(close_tf_ns)
    close_tf_ns, ind_vals = close_tf_ns[order], ind_vals[order]
    idx = np.searchsorted(close_tf_ns, close1m_ns, side="right") - 1   # last closed bar
    out = np.full((close1m_ns.shape[0], ind_vals.shape[1]), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = ind_vals[idx[ok]]
    return out


def build_aligned_indicators(df1m: pd.DataFrame) -> np.ndarray:
    """Return (T, 200) float32 aligned to the 1m index, columns ==
    base.ALL_INDICATOR_COLUMNS order. NO future leakage (last-closed-bar rule)."""
    open_ns = df1m.index.values.astype("datetime64[ns]").astype(np.int64)
    close1m_ns = open_ns + _NS_PER_MIN
    blocks = []
    for tf in C.TIMEFRAMES:
        d = resample_ohlcv(df1m, tf)
        m = base.compute_timeframe_indicators(d["high"].values, d["low"].values, d["close"].values)
        tf_open_ns = d.index.values.astype("datetime64[ns]").astype(np.int64)
        blocks.append(_align_to_1m(m, tf_open_ns, _TF_MIN[tf], close1m_ns))
    mat = np.hstack(blocks).astype(np.float32)
    assert mat.shape[1] == C.N_INDICATORS_TOTAL == len(base.ALL_INDICATOR_COLUMNS)
    return mat


def build_cache(df1m: pd.DataFrame, out_dir: str, symbol: str = "EURUSD") -> dict:
    """Precompute + persist float32 cache (indicators, close, timestamps)."""
    os.makedirs(out_dir, exist_ok=True)
    mat = build_aligned_indicators(df1m)
    np.save(os.path.join(out_dir, f"{symbol}_indicators.npy"), mat)
    np.save(os.path.join(out_dir, f"{symbol}_close.npy"), df1m["close"].values.astype(np.float32))
    np.save(os.path.join(out_dir, f"{symbol}_time_ns.npy"),
            df1m.index.values.astype("datetime64[ns]").astype(np.int64))
    return {"symbol": symbol, "bars": int(len(df1m)), "indicator_cols": int(mat.shape[1])}


def load_cache(out_dir: str, symbol: str = "EURUSD"):
    """memmap-load the cache for zero-copy hot-loop reads -> (indicators, close, time_ns)."""
    j = lambda n: os.path.join(out_dir, f"{symbol}_{n}.npy")
    return (np.load(j("indicators"), mmap_mode="r"),
            np.load(j("close"), mmap_mode="r"),
            np.load(j("time_ns"), mmap_mode="r"))


def load_ohlcv_csv(path):
    """Load a 1-minute OHLCV CSV with flexible column names -> DataFrame indexed by
    datetime (sorted, de-duplicated). Column values are taken positionally so they
    align to the parsed timestamps, not to the CSV's original integer index.
    Feed the result to build_aligned_indicators(df)."""
    import numpy as _np, pandas as _pd
    df = _pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    def pick(opts):
        for o in opts:
            if o in low:
                return low[o]
        return None
    tcol = pick(("datetime", "date_time", "timestamp", "time", "date", "<date>", "gmt time", "gmt_time"))
    if tcol is not None:
        idx = _pd.to_datetime(df[tcol], errors="coerce")
    elif "date" in low and "time" in low:
        idx = _pd.to_datetime(df[low["date"]].astype(str) + " " + df[low["time"]].astype(str), errors="coerce")
    else:
        raise ValueError(f"{path}: no datetime column found (cols={list(df.columns)})")
    def col(opts):
        nm = pick(opts)
        return _pd.to_numeric(df[nm], errors="coerce").to_numpy() if nm else None
    o = col(("open", "o", "<open>")); h = col(("high", "h", "<high>")); l = col(("low", "l", "<low>"))
    c = col(("close", "c", "<close>", "adj close", "price"))
    v = col(("volume", "vol", "v", "<vol>", "tickvol", "tick_volume"))
    if c is None:
        raise ValueError(f"{path}: no close column found (cols={list(df.columns)})")
    out = _pd.DataFrame({"open": o if o is not None else c, "high": h if h is not None else c,
                         "low": l if l is not None else c, "close": c,
                         "volume": v if v is not None else _np.ones(len(df))},
                        index=_pd.DatetimeIndex(_np.asarray(idx)))
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
