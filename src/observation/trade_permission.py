# =====================================================================
# WHEN 2026-07-01 (training wheels) | WHO Claude for Monty
# WHY  The operator KNOWS his strategy works; RL-from-reward can't FIND it (exploration can't hit a tiny
#      structured region). So we GATE the action space to the conditions he trades under ("training wheels"):
#      the bot may only OPEN a SELL when a sell-condition holds, a BUY when a buy-condition holds; RL still
#      learns WHETHER to take it, the exit, and the size WITHIN that window. Removable (curriculum) -> the
#      removal is itself the proof it can ride without the wheels.
# WHERE src/observation/trade_permission.py
# HOW  PRECOMPUTE-ONLY (pandas resample fine; NEVER from step()). LEAK-FREE (SMA/shift use only bars <= t;
#      "shift 4 forward" = the value from 4 TF-bars ago; higher-TF columns are last-closed aligned). Produces a
#      per-bar (sell_allowed, buy_allowed) 0/1 mask the env applies as a HARD directional open-gate.
#      Operator's 3 OR-conditions (SELL; BUY = mirror):
#        G1  close < SMA(4,shift4) of High AND of Low, on 5m & 30m & 4h
#        G2  on 5m & 30m: CCI30 & CCI100 both < -100 AND both < their SMA(2,shift4)
#        G3  close < middle of BB(20,dev1) AND BB(200,dev1), on 5m & 30m & 4h
# DEPENDS_ON: numpy, pandas, src.indicators.base (CCI/BB columns), src.data.aux_features (1m High/Low)
# USED_BY: src/env/trading_env.py (precompute) -> the env's directional open mask; the condition backtest
# CHANGE_NOTES(IRAC): I: RL can't discover a known discretionary edge. R: operator 2026-07-01 gave the exact
#   entry conditions as "training wheels". A: precompute a leak-free per-bar directional trade-permission mask
#   from those conditions. C: RL only explores the operator's known-good windows -> it can FIND (and improve on)
#   the edge instead of flailing; wheels come off later to test generalization.
# =====================================================================
"""Directional TRADE-PERMISSION gate ("training wheels") — the operator's exact sell/buy entry conditions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.base import ALL_INDICATOR_COLUMNS
from src.data.aux_features import OHLC_COLUMNS

# tunable but fixed to the operator's spec
CCI_LEVEL = 100.0          # both CCIs beyond +/- this
SMA_HL_PERIOD, SMA_HL_SHIFT = 4, 4        # G1: SMA(4) shift 4 (forward-plot = value from 4 TF-bars ago)
SMA_CCI_PERIOD, SMA_CCI_SHIFT = 2, 4      # G2: SMA(2) shift 4 applied to each CCI
_TF_MIN = {"5m": 5, "30m": 30, "4h": 240}
_EPS = 1e-9
_H1 = OHLC_COLUMNS.index("1m__high")
_L1 = OHLC_COLUMNS.index("1m__low")


def _col(ind, name, T):
    try:
        return ind[:, ALL_INDICATOR_COLUMNS.index(name)].astype(np.float64)
    except ValueError:
        return np.full(T, np.nan)


def _tf_bars(series_1m, dt, tf_min, how):
    """(T,) leak-free: resample the 1m series to TF bars (how='max'/'min'/'last', labelled at close), and
    return the TF-bar value aligned back to 1m by the LAST-CLOSED tf bar (reindex ffill)."""
    s = pd.Series(series_1m, index=dt)
    tf = getattr(s.resample(f"{tf_min}min", label="right", closed="right"), how)()
    return tf


def _sma_shift_on_tf(tf_series, period, shift, dt):
    """SMA(period) then shift(shift) at TF resolution, aligned back to the 1m grid (last-closed ffill)."""
    sm = tf_series.rolling(period, min_periods=period).mean().shift(shift)
    return sm.reindex(dt, method="ffill").to_numpy()


def compute_trade_permission(ind, close, ohlc, time_ns) -> np.ndarray:
    """(T, 2) float32 [sell_allowed, buy_allowed] (0/1). Leak-free; precompute-only. Neutral (both 1 -> no
    restriction) is NOT used; if inputs are missing a group simply doesn't fire."""
    ind = np.asarray(ind, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64).ravel()
    ohlc = np.asarray(ohlc, dtype=np.float64)
    T = close.shape[0]
    dt = pd.to_datetime(np.asarray(time_ns).astype("int64"))

    def nz(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sell = np.ones(T, bool); buy = np.ones(T, bool)   # AND across the TFs within each group -> start True

    # --- G1: close vs SMA(4,shift4) of High and Low, on 5m & 30m & 4h ---
    have_hl = np.any(ohlc[:, _H1]) or np.any(ohlc[:, _L1])
    g1_sell = np.ones(T, bool) if have_hl else np.zeros(T, bool)
    g1_buy = g1_sell.copy()
    if have_hl:
        high1m, low1m = ohlc[:, _H1], ohlc[:, _L1]
        for tf, m in _TF_MIN.items():
            hi = _sma_shift_on_tf(_tf_bars(high1m, dt, m, "max"), SMA_HL_PERIOD, SMA_HL_SHIFT, dt)
            lo = _sma_shift_on_tf(_tf_bars(low1m, dt, m, "min"), SMA_HL_PERIOD, SMA_HL_SHIFT, dt)
            valid = np.isfinite(hi) & np.isfinite(lo)
            g1_sell &= valid & (close < hi) & (close < lo)          # below BOTH shifted SMAs
            g1_buy &= valid & (close > hi) & (close > lo)

    # --- G2: CCI30 & CCI100 both beyond -/+100 AND beyond their SMA(2,shift4), on 5m & 30m ---
    g2_sell = np.ones(T, bool); g2_buy = np.ones(T, bool)
    for tf in ("5m", "30m"):
        m = _TF_MIN[tf]
        for p in ("30", "100"):
            c = _col(ind, f"{tf}__cci{p}_raw", T)
            sma = _sma_shift_on_tf(_tf_bars(c, dt, m, "last"), SMA_CCI_PERIOD, SMA_CCI_SHIFT, dt)
            valid = np.isfinite(c) & np.isfinite(sma)
            g2_sell &= valid & (c < -CCI_LEVEL) & (c < sma)
            g2_buy &= valid & (c > CCI_LEVEL) & (c > sma)

    # --- G3: close vs the MIDDLE of BB(20,dev1) and BB(200,dev1), on 5m & 30m & 4h ---
    g3_sell = np.ones(T, bool); g3_buy = np.ones(T, bool)
    for tf in ("5m", "30m", "4h"):
        for p in ("20", "200"):
            mid = _col(ind, f"{tf}__bb{p}_dev1.0_middle", T)
            valid = np.isfinite(mid) & (mid != 0.0)
            g3_sell &= valid & (close < mid)
            g3_buy &= valid & (close > mid)

    sell = g1_sell | g2_sell | g3_sell     # OR across the 3 condition groups
    buy = g1_buy | g2_buy | g3_buy
    return np.stack([sell.astype(np.float32), buy.astype(np.float32)], axis=1)
