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
    """Load a 1-minute OHLCV file with flexible column names AND delimiter -> DataFrame indexed by
    datetime (sorted, de-duplicated). Handles comma / TAB / semicolon files, angle-bracket headers,
    and MetaTrader-style SPLIT date+time columns (e.g. MT5 export: `<DATE>\\t<TIME>\\t<OPEN>...` with
    dotted dates `2021.01.13`). Values are positional so they align to the parsed timestamps.
    Feed the result to build_aligned_indicators(df)."""
    import numpy as _np, pandas as _pd
    # 1) sniff the delimiter from the header line (fast C engine; MT5 history exports are TAB-separated)
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    counts = {d: header.count(d) for d in ("\t", ";", "|", ",")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ","
    df = _pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    # 2) normalize headers: lowercase, strip whitespace AND angle brackets  (<DATE> -> date)
    low = {c.lower().strip().strip("<>").strip(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in low:
                return low[o]
        return None
    # 3) build the timestamp — combine SEPARATE date+time first (MT5), else a single datetime column
    dt_col, date_col, time_col = pick("datetime", "date_time", "timestamp", "gmt time", "gmt_time"), pick("date"), pick("time")
    if dt_col is not None:
        raw = df[dt_col].astype(str)
    elif date_col is not None and time_col is not None:
        raw = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    elif date_col is not None:
        raw = df[date_col].astype(str)
    elif time_col is not None:
        raw = df[time_col].astype(str)
    else:
        raise ValueError(f"{path}: no datetime column found (cols={list(df.columns)})")
    idx = _pd.to_datetime(raw, errors="coerce")
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d"):   # MT5 dotted dates if default parse failed
        if idx.isna().mean() > 0.5:
            idx = _pd.to_datetime(raw, format=fmt, errors="coerce")
    def col(*opts):
        nm = pick(*opts)
        return _pd.to_numeric(df[nm], errors="coerce").to_numpy() if nm else None
    o = col("open", "o"); h = col("high", "h"); l = col("low", "l")
    c = col("close", "c", "adj close", "price")
    # prefer TICK volume (real <VOL> is usually 0 on forex MT5 exports)
    v = col("tickvol", "tick_volume", "tickvolume", "volume", "vol", "v")
    if c is None:
        raise ValueError(f"{path}: no close column found (cols={list(df.columns)})")
    out = _pd.DataFrame({"open": o if o is not None else c, "high": h if h is not None else c,
                         "low": l if l is not None else c, "close": c,
                         "volume": v if v is not None else _np.ones(len(df))},
                        index=_pd.DatetimeIndex(_np.asarray(idx)))
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
