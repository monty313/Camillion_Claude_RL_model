# =====================================================================
# WHEN 2026-06-30 (Phase 2, contract v1.10.0) | WHO Claude for Monty
# WHY  Operator's "Shifted SMA Hugging Pressure" agent (the green/red shifted-MA-on-High/Low envelope on the
#      US30 M5 chart). A HEAVY momentum-continuation sense: across 5m / 15m / 1h, a fast SMA(2) of High and Low
#      shifted forward 1 bar forms an envelope; price that keeps HUGGING one side (never touching the opposite
#      band) for consecutive bars = sustained directional pressure. 3+ (all) timeframes agreeing = strong
#      continuation. The policy SEES this (heavy obs block) and is REWARDED for trading with it (heavy prior +
#      an indices/metals miss-penalty live in portfolio_env, not here).
# WHERE src/observation/hug_pressure.py
# HOW  PRECOMPUTE-ONLY (pandas resample is fine here -- NEVER called from step()). Adds 15m & 1h as a contained
#      RESAMPLED side-channel from the 1m High/Low (NOT new full obs timeframes -- the engine still runs
#      1m/5m/30m/4h/1d). LEAK-FREE: the envelope is shift(1) (2 closed bars ago) and the per-TF hug is aligned
#      to 1m by LAST-CLOSED tf bar (label='right' + ffill on index<=t). STATIC (market-only, per-bar) -> placed
#      in the static obs tensor, lifted byte-identical into the JAX env (auto parity, no jnp twin).
# DEPENDS_ON: numpy, pandas, src.data.aux_features (1m High/Low columns)
# USED_BY: src/env/trading_env.py (_precompute), src/observation/observation_contract.py (names),
#          jax_tpu/jax_static_features.py (placed static), tests/test_hug_pressure.py
# CHANGE_NOTES(IRAC): I: the bot couldn't SEE the shifted-SMA hug that defines the operator's momentum read.
#   R: operator 2026-06-30 ("add this agent, make it very very heavy") + AskUserQuestion (full heavy, 15m/1h via
#   side-channel). A: 15 leak-free per-bar scores appended as a STATIC block (v1.9.0->v1.10.0). C: the policy
#   perceives multi-TF hug pressure as a first-class momentum-continuation signal it can learn to ride.
# =====================================================================
"""The v1.10.0 SHIFTED-SMA HUGGING-PRESSURE block (15 leak-free per-bar scores across 5m / 15m / 1h)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data.aux_features import OHLC_COLUMNS

N_HUG_SCORES: int = 15   # == config.constants.OBS_BLOCK_HUG_PRESSURE

# the 3 timeframes (minutes). 15m & 1h are RESAMPLED here from the 1m High/Low (contained side-channel).
HUG_TFS: tuple[tuple[str, int], ...] = (("5m", 5), ("15m", 15), ("1h", 60))
HUG_COUNT_CAP: float = 20.0   # consecutive no-opposite-touch bars are normalized by this

# Field ORDER (== observation_contract.HUG_PRESSURE_NAMES).
HUG_PRESSURE_NAMES: tuple[str, ...] = (
    # per timeframe: side (+1 bull-hug / -1 bear-hug / 0), consecutive hug count (norm), respecting-now flag
    "hug_5m_side", "hug_5m_count", "hug_5m_respect",
    "hug_15m_side", "hug_15m_count", "hug_15m_respect",
    "hug_1h_side", "hug_1h_count", "hug_1h_respect",
    # aggregate multi-TF pressure
    "hug_agree_bull",        # # of TFs bull-hugging / 3
    "hug_agree_bear",        # # of TFs bear-hugging / 3
    "hug_net_pressure",      # signed, weighted by hug counts (-1..1)
    "hug_strength",          # combined 0..1 heavy score
    "hug_continuation_3plus",# 1.0 if >=3 TFs agree on a side (ALL of 5m/15m/1h = the strong-continuation condition)
    "hug_dominant_side",     # +1 / -1 / 0 net side across TFs
)

# indices WITHIN the hug block, read by the heavy reward (portfolio_env / jax_portfolio_env).
IDX_CONTINUATION_3PLUS: int = HUG_PRESSURE_NAMES.index("hug_continuation_3plus")   # 13
IDX_DOMINANT_SIDE: int = HUG_PRESSURE_NAMES.index("hug_dominant_side")             # 14

# A >=2-TF hug continuation is "CLEAN" (worth strongly pushing into) UNLESS the momentum block says the move is
# already exhausted / extended in the SAME direction / decaying -> then we DON'T hard-force (no miss-penalty).
# Shared by CPU + JAX so the reward is identical. (See momentum_scores.IDX_EXHAUSTION/IDX_LOCATION/IDX_DECAY.)
HUG_EXH_THR: float = 0.5
HUG_DECAY_THR: float = 0.5
HUG_LOC_THR: float = 0.8

_EPS = 1e-9
_H1 = OHLC_COLUMNS.index("1m__high")
_L1 = OHLC_COLUMNS.index("1m__low")


def _hug_one_tf(high1m: np.ndarray, low1m: np.ndarray, dt_index: pd.DatetimeIndex, tf_min: int) -> np.ndarray:
    """(T, 3) [side, count_norm, respecting] for one timeframe, aligned leak-free to the 1m grid."""
    sH = pd.Series(high1m, index=dt_index)
    sL = pd.Series(low1m, index=dt_index)
    rule = f"{tf_min}min"
    # tf bars labeled at their CLOSE time (label='right') -> at 1m t, ffill picks the last bar that CLOSED <= t
    tfH = sH.resample(rule, label="right", closed="right").max()
    tfL = sL.resample(rule, label="right", closed="right").min()
    # SMA(2) of High / Low, shifted FORWARD 1 tf bar = the envelope is from 2 closed bars ago (leak-free)
    env_up = tfH.rolling(2, min_periods=2).mean().shift(1).to_numpy()
    env_lo = tfL.rolling(2, min_periods=2).mean().shift(1).to_numpy()
    hi = tfH.to_numpy(); lo = tfL.to_numpy()
    # a tf bar that does NOT touch the opposite band: bull keeps Low above the lower band; bear keeps High below
    # the upper band. NaN (warmup / empty weekend bucket) -> treated as a touch (resets the count).
    bull_ok = np.nan_to_num((lo > env_lo).astype(np.float64), nan=0.0) > 0.5
    bear_ok = np.nan_to_num((hi < env_up).astype(np.float64), nan=0.0) > 0.5
    n = len(tfH)
    bullc = np.zeros(n); bearc = np.zeros(n)
    for i in range(1, n):
        bullc[i] = bullc[i - 1] + 1.0 if bull_ok[i] else 0.0
        bearc[i] = bearc[i - 1] + 1.0 if bear_ok[i] else 0.0
    side = np.sign(bullc - bearc)
    dom_count = np.where(side > 0, bullc, np.where(side < 0, bearc, 0.0))
    respect = ((side > 0) & bull_ok) | ((side < 0) & bear_ok)
    tf_df = pd.DataFrame(
        {"side": side, "count": np.minimum(dom_count, HUG_COUNT_CAP) / HUG_COUNT_CAP,
         "respect": respect.astype(np.float64)},
        index=tfH.index)
    aligned = tf_df.reindex(dt_index, method="ffill").fillna(0.0)   # last CLOSED tf bar at each 1m step
    return aligned[["side", "count", "respect"]].to_numpy()


def compute_hug_pressure(ohlc_matrix, time_ns) -> np.ndarray:
    """(T, 15) float32 hugging-pressure scores from the 1m High/Low + the timestamps. Leak-free; precompute
    only. If `ohlc_matrix` has no real OHLC (env built without aux) the 1m H/L are 0 -> the whole block is 0
    (a safe, neutral no-signal), which is fine."""
    ohlc = np.asarray(ohlc_matrix, dtype=np.float64)
    T = ohlc.shape[0]
    high1m = ohlc[:, _H1]; low1m = ohlc[:, _L1]
    if not np.any(high1m) and not np.any(low1m):           # no OHLC threaded in -> neutral block
        return np.zeros((T, N_HUG_SCORES), dtype=np.float32)
    dt = pd.to_datetime(np.asarray(time_ns).astype("int64"))

    per_tf = [_hug_one_tf(high1m, low1m, dt, m) for _, m in HUG_TFS]   # each (T, 3)
    sides = np.stack([a[:, 0] for a in per_tf], axis=1)    # (T, 3)
    counts = np.stack([a[:, 1] for a in per_tf], axis=1)   # (T, 3)

    agree_bull = (sides > 0).sum(axis=1) / 3.0
    agree_bear = (sides < 0).sum(axis=1) / 3.0
    net_pressure = np.clip((sides * counts).sum(axis=1) / 3.0, -1.0, 1.0)
    dominant = np.sign(sides.sum(axis=1))
    cont3 = (((sides > 0).sum(axis=1) >= 3) | ((sides < 0).sum(axis=1) >= 3)).astype(np.float64)  # ALL 3 TFs agree
    strength = np.clip(0.5 * np.maximum(agree_bull, agree_bear) + 0.5 * np.abs(net_pressure), 0.0, 1.0)

    per_tf_cols = np.concatenate(per_tf, axis=1)           # (T, 9): side/count/respect x 3 TFs
    agg = np.stack([agree_bull, agree_bear, net_pressure, strength, cont3, dominant], axis=1)  # (T, 6)
    out = np.concatenate([per_tf_cols, agg], axis=1)       # (T, 15)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
