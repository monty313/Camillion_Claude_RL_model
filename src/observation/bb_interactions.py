# =====================================================================
# WHEN 2026-06-30 (Phase 2, contract v1.11.0) | WHO Claude for Monty
# WHY  The "Hierarchical dual-BB" briefing asked for engineered cross-TF Bollinger interactions. Most of it the
#      bot ALREADY sees (multi-TF agreement -> momentum.alignment + hug; band position -> momentum.location;
#      band-stack -> trade_risk; raw bb20/bb200 -> indicator block). This module adds ONLY the 3 genuinely NEW
#      logic families (operator 2026-06-30: "add any logic we don't have, but don't duplicate anything"):
#        1. BB-WIDTH SQUEEZE/EXPANSION  -- fast(20) band width vs its own recent average (coil -> breakout). We
#           only had ATR vol regime; BB squeeze is a distinct setup.
#        2. BB-DISTANCE MOMENTUM CASCADE -- signed acceleration of the fast band-distance in the DIRECTION of the
#           next higher TF's slow trend. Distinct from the static CCI alignment (this is dynamic + signed).
#        3. BB-EXTREME MEAN-REVERSION FLAGS -- price at a higher-TF BB200 edge AND the fast band reverting back
#           inside (the fade setup). Distinct from the CCI |160| exhaustion.
# WHERE src/observation/bb_interactions.py
# HOW  PURE per-bar transforms of the ALREADY-CACHED bb20/bb200 (dev2.0) columns + close. LEAK-FREE (rolling /
#      roc use only bars <= t; higher-TF columns are last-closed aligned). Precompute-only (pandas is fine --
#      NEVER from step()). STATIC (market-only) -> placed in the static obs tensor, lifted byte-identical into
#      the JAX env (auto parity, no jnp twin). NO reward change (perception only).
# DEPENDS_ON: numpy, pandas, src.indicators.base (column names)
# USED_BY: src/env/trading_env.py (_precompute), src/observation/observation_contract.py (names),
#          jax_tpu/jax_static_features.py (placed static), tests/test_bb_interactions.py
# CHANGE_NOTES(IRAC): I: 3 dual-BB logics the obs lacked (squeeze, signed cascade, band-extreme MR). R: operator
#   "add only NEW logic, no duplication". A: 12 leak-free per-bar scores appended as a STATIC block
#   (v1.10.0->v1.11.0). C: the policy perceives BB volatility-regime + cross-TF momentum acceleration + the
#   fade-the-extreme setup WITHOUT re-encoding the multi-TF agreement / band-position it already has.
# =====================================================================
"""The v1.11.0 DUAL-BB INTERACTION block (12 leak-free per-bar scores) — only the logic the obs didn't have."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.indicators.base import ALL_INDICATOR_COLUMNS

N_BB_INTERACTION_SCORES: int = 12   # == config.constants.OBS_BLOCK_BB_INTERACTIONS

# Field ORDER (== observation_contract.BB_INTERACTION_NAMES).
BB_INTERACTION_NAMES: tuple[str, ...] = (
    # 1. BB-WIDTH squeeze/expansion: fast(20) band width vs its own ~recent avg. + = expanding (breakout),
    #    - = coiling (squeeze). Plus a flag when ALL of 5m/30m/4h are coiled (breakout pending).
    "bbw_expansion_5m", "bbw_expansion_30m", "bbw_expansion_4h", "bbw_squeeze_all",
    # 2. BB-distance MOMENTUM CASCADE: fast band-distance acceleration, signed by the next higher TF's slow
    #    trend (+ = fast move accelerating WITH the higher-TF trend).
    "bb_cascade_5m_30m", "bb_cascade_30m_4h", "bb_cascade_net",
    # 3. BB-EXTREME MEAN-REVERSION flags: price at a higher-TF BB200 edge AND the 5m fast band reverting inside.
    "bb_mr_long_30m", "bb_mr_short_30m", "bb_mr_long_4h", "bb_mr_short_4h",
    # cross-TF vol ratio: 5m fast width vs 4h fast width (local vs structural volatility).
    "bbw_ratio_5m_vs_4h",
)

_EPS = 1e-9
_DEV = "2.0"   # use the standard +/-2 sigma bands for distance/width
_TFS = ("5m", "30m", "4h")
_SQUEEZE_THR = -0.20    # bbw_expansion below this on ALL TFs = coiled
_AT_EDGE_SIGMA = 1.5    # price this many slow-sigmas past center = "at the BB200 edge"
_REV_FROM = 2.0         # fast band-distance had reached this extreme...
_REV_TO = 1.5           # ...and has now come back inside this -> reverting
# roc lookback per TF, in 1m bars, ~= 3 of that TF's OWN bars (higher-TF signals are piecewise-constant on the
# 1m grid, so a 3-1m-bar diff is ~0 -- the cascade must measure change over a few real TF bars).
_ROC_K = {"5m": 15, "30m": 90, "4h": 720}


def _idx(name: str) -> int:
    try:
        return ALL_INDICATOR_COLUMNS.index(name)
    except ValueError:
        return -1


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def compute_bb_interactions(ind, close) -> np.ndarray:
    """(T, 12) float32. Leak-free; precompute-only. If the bb columns are absent/zero the block is neutral 0."""
    ind = np.asarray(ind, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]

    def col(name):
        j = _idx(name)
        return ind[:, j].astype(np.float64) if j >= 0 else np.full(T, np.nan)

    def nz(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def tf_feats(tf):
        up20 = col(f"{tf}__bb20_dev{_DEV}_upper"); mid20 = col(f"{tf}__bb20_dev{_DEV}_middle")
        lo20 = col(f"{tf}__bb20_dev{_DEV}_lower")
        up200 = col(f"{tf}__bb200_dev{_DEV}_upper"); mid200 = col(f"{tf}__bb200_dev{_DEV}_middle")
        fast_std = (up20 - mid20) / 2.0
        slow_std = (up200 - mid200) / 2.0
        fast_dist = nz((close - mid20) / np.where(np.abs(fast_std) > _EPS, fast_std, np.nan))
        slow_dist = nz((close - mid200) / np.where(np.abs(slow_std) > _EPS, slow_std, np.nan))
        fast_width = nz((up20 - lo20) / np.where(np.abs(mid20) > _EPS, mid20, np.nan))
        # squeeze/expansion: fast width vs its own ~recent average (leak-free rolling, min_periods so warmup=0)
        avg_w = pd.Series(fast_width).rolling(50, min_periods=5).mean().to_numpy()
        expansion = np.tanh(nz(fast_width / (avg_w + _EPS)) - 1.0)
        return {"fast_dist": fast_dist, "slow_dist": slow_dist, "fast_width": fast_width, "expansion": expansion}

    f = {tf: tf_feats(tf) for tf in _TFS}

    # 1. squeeze/expansion
    exp5, exp30, exp4 = f["5m"]["expansion"], f["30m"]["expansion"], f["4h"]["expansion"]
    squeeze_all = ((exp5 < _SQUEEZE_THR) & (exp30 < _SQUEEZE_THR) & (exp4 < _SQUEEZE_THR)).astype(np.float64)

    # 2. cascade: roc(fast_dist) over ~3 of that TF's OWN bars, signed by the next-higher TF's slow trend
    roc5 = nz(f["5m"]["fast_dist"] - _shift(f["5m"]["fast_dist"], _ROC_K["5m"]))
    roc30 = nz(f["30m"]["fast_dist"] - _shift(f["30m"]["fast_dist"], _ROC_K["30m"]))
    casc_5_30 = roc5 * np.sign(f["30m"]["slow_dist"])
    casc_30_4 = roc30 * np.sign(f["4h"]["slow_dist"])
    cascade_5m_30m = np.tanh(casc_5_30 / 2.0)
    cascade_30m_4h = np.tanh(casc_30_4 / 2.0)
    cascade_net = np.tanh((casc_5_30 + casc_30_4) / 4.0)

    # 3. mean-reversion: price at a higher-TF BB200 edge AND the 5m fast band reverting back inside. The
    # "recently extreme" window spans ~5 of the 5m bars (the fast dist is piecewise-constant on the 1m grid).
    fd5 = f["5m"]["fast_dist"]
    roll_max5 = pd.Series(fd5).rolling(25, min_periods=2).max().to_numpy()
    roll_min5 = pd.Series(fd5).rolling(25, min_periods=2).min().to_numpy()
    rev_down = (fd5 < _REV_TO) & (nz(roll_max5) > _REV_FROM)        # was high, now falling back inside
    rev_up = (fd5 > -_REV_TO) & (nz(roll_min5) < -_REV_FROM)        # was low, now rising back inside

    def mr_pair(tf):
        sd = f[tf]["slow_dist"]
        mr_long = ((sd < -_AT_EDGE_SIGMA) & rev_up).astype(np.float64)     # at lower BB200 edge + turning up
        mr_short = ((sd > _AT_EDGE_SIGMA) & rev_down).astype(np.float64)   # at upper BB200 edge + turning down
        return mr_long, mr_short
    mr_l30, mr_s30 = mr_pair("30m")
    mr_l4, mr_s4 = mr_pair("4h")

    # cross-TF vol ratio (local 5m vs structural 4h)
    ratio_5_4 = np.tanh(nz(f["5m"]["fast_width"] / (f["4h"]["fast_width"] + _EPS)) - 1.0)

    out = np.stack([exp5, exp30, exp4, squeeze_all,
                    cascade_5m_30m, cascade_30m_4h, cascade_net,
                    mr_l30, mr_s30, mr_l4, mr_s4, ratio_5_4], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
