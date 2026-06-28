# =====================================================================
# WHEN 2026-06-28 (Phase 2, contract v1.6.0) | WHO Claude for Monty
# WHY  The per-bar features that live OUTSIDE the 220-indicator cache but still need
#      the raw OHLC (which the env does not carry). TWO concerns, ONE aligned array
#      so they share transport (built once at cache time, trimmed together by
#      align_symbol_data, threaded into the env):
#        PART A -- OHLC observation block (NEW in contract v1.6.0): raw Open/High/Low/
#          Close of the last CLOSED bar on each of the 5 timeframes = 20 floats. Goes
#          INTO the observation (the policy finally sees High/Low/Open, not just close).
#        PART B -- ADX +DI/-DI strategy side-channel: +DI & -DI for periods 14 & 45 on
#          5m/30m/4h = 12 floats. Feeds ONLY the two ADX-DI alphas via ctx; NOT in the obs.
# WHERE src/data/aux_features.py
# HOW  Pure layout constants + per-timeframe DI compute (adx). The actual resample +
#      leak-free 1m alignment lives in cache_builder.build_aligned_aux (it owns the
#      OHLC + the alignment helpers). The combined array is aux[:, :20]=OHLC, [20:]=DI.
# DEPENDS_ON: config/constants, src/indicators/adx
# USED_BY: src/data/cache_builder (build/align/save/load), src/env/trading_env (split),
#          jax_tpu/jax_static_features (OHLC placed into the static obs)
# CHANGE_NOTES(IRAC): I: two new needs (OHLC in obs + ADX-DI alphas) both require raw
#   high/low the env lacks. R: operator 2026-06-28 (add OHLC to the obs; add the two
#   ADX-DI alphas). A: one aligned aux array -- OHLC for the obs, DI for the alphas --
#   built where OHLC exists and threaded in; obs indices 0..478 are UNCHANGED (append).
#   C: the bot gains raw OHLC perception and a trend-direction alpha family in one
#   leak-free, parity-clean side array shared by the CPU and TPU pipelines.
# =====================================================================
"""Auxiliary per-bar features: PART A = OHLC obs block (20), PART B = ADX +DI/-DI side-channel (12)."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.indicators.adx import plus_di, minus_di

# --- PART A: OHLC observation block (contract v1.6.0) ----------------------------
# Raw OHLC of the last CLOSED bar on each timeframe, in TIMEFRAMES order, fields in this order.
OHLC_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
OHLC_COLUMNS: list[str] = [f"{tf}__{fld}" for tf in C.TIMEFRAMES for fld in OHLC_FIELDS]  # 5*4 = 20
N_OHLC: int = len(OHLC_COLUMNS)                                                            # 20 == C.OBS_BLOCK_OHLC

# --- PART B: ADX +DI/-DI strategy side-channel (NOT in the observation) ----------
ADX_DI_PERIODS: tuple[int, ...] = (14, 45)                      # operator: ADX 14 & 45
ADX_DI_TIMEFRAMES: tuple[str, ...] = ("5m", "30m", "4h")        # the TFs the two alphas read


def _per_tf_di_columns() -> list[str]:
    """Per-timeframe DI column names, ORDER = [plus_di{p}, minus_di{p}] for each period."""
    cols: list[str] = []
    for p in ADX_DI_PERIODS:
        cols.append(f"plus_di{p}")
        cols.append(f"minus_di{p}")
    return cols


DI_PER_TF_COLUMNS: list[str] = _per_tf_di_columns()                                       # [+14,-14,+45,-45]
DI_COLUMNS: list[str] = [f"{tf}__{c}" for tf in ADX_DI_TIMEFRAMES for c in DI_PER_TF_COLUMNS]  # 3*4 = 12
N_DI: int = len(DI_COLUMNS)                                                                # 12

# --- combined aux layout: [ OHLC (20) | DI (12) ] = 32 ---------------------------
AUX_NCOLS: int = N_OHLC + N_DI                                                             # 32
OHLC_SLICE: slice = slice(0, N_OHLC)                                                       # aux[:, :20] -> obs block
DI_SLICE: slice = slice(N_OHLC, AUX_NCOLS)                                                 # aux[:, 20:] -> alpha ctx


def compute_tf_di(high, low, close) -> np.ndarray:
    """(N, len(DI_PER_TF_COLUMNS)) for ONE timeframe -> [plus14, minus14, plus45, minus45], float32, NaN warmup."""
    high = np.asarray(high, dtype=np.float64).ravel()
    low = np.asarray(low, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    n = close.shape[0]
    out = np.full((n, len(DI_PER_TF_COLUMNS)), np.nan, dtype=np.float32)
    col = 0
    for p in ADX_DI_PERIODS:
        out[:, col] = plus_di(high, low, close, p); col += 1
        out[:, col] = minus_di(high, low, close, p); col += 1
    assert col == len(DI_PER_TF_COLUMNS), (col, len(DI_PER_TF_COLUMNS))
    return out
