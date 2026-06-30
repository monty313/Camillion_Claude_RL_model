# =====================================================================
# WHEN 2026-06-30 (Phase 2, contract v1.12.0) | WHO Claude for Monty
# WHY  The "super scalper" brief wants 1m-base momentum-cascade features for fast entries. We compute the
#      dual-BB interaction logic on 5m/30m/4h (bb_interactions, v1.11.0) but NOT on the 1m base TF. This adds
#      the 4 genuinely-new 1m signals (operator: "get it all done"): 1m fast-band distance + its roc, 1m-vs-5m
#      vol expansion (breakout ignition), and the 1m momentum CASCADE (1m fast-dist acceleration signed by the
#      5m & 30m slow trend). NOT a duplicate: bb_interactions starts at 5m; this is the 1m layer the scalper
#      enters on. (Brief said "x sign of 1h trend"; we don't run 1h as an engine TF -> use 5m & 30m, the real
#      higher TFs.)
# WHERE src/observation/scalp_momentum.py
# HOW  PURE per-bar transforms of the cached 1m/5m/30m bb20/bb200 columns + close. LEAK-FREE (1m roc over 3 1m
#      bars; higher-TF columns are last-closed aligned). Precompute-only. STATIC -> lifted byte-identical into
#      the JAX env (auto parity).
# DEPENDS_ON: numpy, pandas, src.indicators.base
# USED_BY: src/env/trading_env.py (_precompute), observation_contract, jax_static_features, tests
# CHANGE_NOTES(IRAC): I: the 1m entry-timing cascade the scalper needs wasn't in the obs. R: operator brief
#   "1m scalp_momentum". A: 4 leak-free 1m scores appended as a STATIC block (v1.11.0->v1.12.0). C: the policy
#   can time fast 1m entries that agree with the higher-TF trend, without re-encoding the 5m+ cascade it has.
# =====================================================================
"""The v1.12.0 1m SCALP-MOMENTUM block (4 leak-free per-bar scores) — the 1m entry-timing layer."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.base import ALL_INDICATOR_COLUMNS

N_SCALP_SCORES: int = 4   # == config.constants.OBS_BLOCK_SCALP_MOMENTUM

SCALP_MOMENTUM_NAMES: tuple[str, ...] = (
    "scalp_fast_dist_1m",     # signed std-devs of close from the 1m BB(20) center (tanh-bounded)
    "scalp_fast_roc_1m",      # 3-bar rate of change of scalp_fast_dist_1m (1m acceleration)
    "scalp_vol_exp_1m_vs_5m", # 1m BB width / 5m BB width (1m breakout vs 5m structure)
    "scalp_cascade_1m",       # 1m fast-dist acceleration signed by the 5m & 30m slow trend (with-trend ignition)
)

_EPS = 1e-9
_DEV = "2.0"


def _idx(name):
    try:
        return ALL_INDICATOR_COLUMNS.index(name)
    except ValueError:
        return -1


def compute_scalp_momentum(ind, close) -> np.ndarray:
    """(T, 4) float32. Leak-free; precompute-only. Neutral 0 where the bb columns are absent."""
    ind = np.asarray(ind, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]

    def col(name):
        j = _idx(name)
        return ind[:, j].astype(np.float64) if j >= 0 else np.full(T, np.nan)

    def nz(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def dist_width(tf):
        up = col(f"{tf}__bb20_dev{_DEV}_upper"); mid = col(f"{tf}__bb20_dev{_DEV}_middle")
        lo = col(f"{tf}__bb20_dev{_DEV}_lower")
        std = (up - mid) / 2.0
        dist = nz((close - mid) / np.where(np.abs(std) > _EPS, std, np.nan))
        width = nz((up - lo) / np.where(np.abs(mid) > _EPS, mid, np.nan))
        return dist, width

    def slow_dist(tf):
        up = col(f"{tf}__bb200_dev{_DEV}_upper"); mid = col(f"{tf}__bb200_dev{_DEV}_middle")
        std = (up - mid) / 2.0
        return nz((close - mid) / np.where(np.abs(std) > _EPS, std, np.nan))

    fast_dist_1m, fast_width_1m = dist_width("1m")
    _, fast_width_5m = dist_width("5m")
    slow_5m = slow_dist("5m"); slow_30m = slow_dist("30m")

    fast_dist = np.tanh(fast_dist_1m / 3.0)                              # bounded signed distance
    roc_raw = nz(fast_dist_1m - np.r_[np.full(3, np.nan), fast_dist_1m[:-3]])
    fast_roc = np.tanh(roc_raw / 2.0)
    vol_exp = np.tanh(nz(fast_width_1m / (fast_width_5m + _EPS)) - 1.0)
    cascade = np.tanh(roc_raw * np.sign(slow_5m) * np.sign(slow_30m) / 2.0)   # 1m accel WITH the higher-TF trend

    out = np.stack([fast_dist, fast_roc, vol_exp, cascade], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
