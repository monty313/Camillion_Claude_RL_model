# =====================================================================
# WHEN 2026-06-30 (Phase 2, contract v1.9.0) | WHO Claude for Monty
# WHY  Teach the agent the PRINCIPLE of momentum, not hard-coded CCI rules (see JORDAN_PRINCIPLES.md).
#      The operator decomposed "momentum" into learnable SUB-PROBLEMS (a decision tree): tradeability,
#      bias, alignment, strength, exhaustion, entry-location, structure, persistence, decay. Each becomes a
#      per-bar SCORE the policy CONSUMES (it doesn't have to invent momentum from raw indicators) and learns
#      to act on -- so it generalizes across instruments/sessions/regimes instead of memorizing thresholds.
# WHERE src/observation/momentum_scores.py
# HOW  PURE per-bar transforms of the ALREADY-CACHED indicators (CCI/SMA/ATR/BB) + close. LEAK-FREE
#      (rolling windows use only bars <= t; higher-TF columns are last-closed-bar aligned). Precompute-only
#      (pandas here is fine -- NEVER called from step()). These are STATIC features (market-only, no account
#      state) -> they live in the STATIC obs tensor, lifted byte-identical into the JAX env (auto parity).
# DEPENDS_ON: numpy, pandas, config.constants, src.indicators.base (column names)
# USED_BY: src/env/trading_env.py (_precompute), src/observation/observation_contract.py (names),
#          jax_tpu/jax_static_features.py (placed static), tests/test_momentum_scores.py
# CHANGE_NOTES(IRAC): I: hard-coding Jordan's CCI ladder = a brittle rule bot. R: operator 2026-06-30
#   "learn the principle, not the rule" + the momentum decision tree. A: 9 leak-free per-bar scores (one per
#   tree node) appended as a STATIC obs block (v1.8.0->v1.9.0). C: the policy perceives momentum the way a
#   trader decomposes it (is anything happening -> which way -> how strong -> right spot -> still alive) and
#   learns Jordan's PREFERENCES over those states -> generalizes instead of replaying one trigger.
# =====================================================================
"""The v1.9.0 MOMENTUM-PERCEPTION block (9 leak-free per-bar scores) — one per the operator's momentum tree."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import constants as C
from src.indicators.base import ALL_INDICATOR_COLUMNS

N_MOMENTUM_SCORES: int = 9   # == config.constants.OBS_BLOCK_MOMENTUM

# Field ORDER (== observation_contract.MOMENTUM_NAMES). One score per decision-tree node.
MOMENTUM_NAMES: tuple[str, ...] = (
    "mom_tradeability",   # node 1: is momentum even present? (graded, 0..1)
    "mom_bias",           # node 2: higher-TF direction (-1..1)
    "mom_alignment",      # node 3: do the timeframes agree? (-1..1)
    "mom_strength",       # node 4: how strong (graded CCI ladder, 0..1)
    "mom_exhaustion",     # node 4: blow-off risk (>extreme, 0..1)
    "mom_location",       # node 4: entry location -- extension vs pullback (band position, -1..1)
    "mom_structure",      # node 4: position in the recent range -- near a breakout level (0..1)
    "mom_persistence",    # node 5: recent follow-through / one-directional-ness (0..1)
    "mom_decay",          # node 6: momentum dying -- CCI rolled back from its recent peak (0..1)
)


def _col(name: str):
    try:
        return ALL_INDICATOR_COLUMNS.index(name)
    except ValueError:
        return -1


_EPS = 1e-9
_I = {n: _col(n) for n in (
    "5m__cci30_raw", "5m__cci100_raw", "30m__cci30_raw", "4h__cci30_raw",
    "30m__sma_p200_s0", "4h__sma_p200_s0", "5m__atr14_raw",
    "5m__bb20_dev1.0_upper", "5m__bb20_dev1.0_middle")}


def compute_momentum_scores(ind, close, time_ns=None, *, strength_level=160.0, exhaustion_span=120.0,
                            tradeability_scale=100.0, bias_atr_scale=5.0,
                            structure_win=120, persistence_win=30, decay_win=20) -> np.ndarray:
    """(T, 9) float32 momentum scores from the cached indicators + close. Leak-free; precompute-only.

    Windows/levels (CCI 50/100/160 ladder, range/persistence/decay lookbacks) are TUNABLE starting points --
    the policy LEARNS the weighting; these only have to expose the right signal. The keyword knobs exist so the
    PROOF HARNESS (Stage 6, `jax_tpu/jax_proof.py`) can PERTURB the recipe and test whether the policy learned
    the PRINCIPLE (survives a different recipe) or just memorized THIS one. Defaults reproduce v1.9.0 exactly."""
    ind = np.asarray(ind, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]

    def col(name):
        j = _I.get(name, -1)
        return ind[:, j].astype(np.float64) if j >= 0 else np.full(T, np.nan)

    c5 = col("5m__cci30_raw"); c5s = col("5m__cci100_raw")
    c30 = col("30m__cci30_raw"); c4h = col("4h__cci30_raw")
    sma30 = col("30m__sma_p200_s0"); sma4h = col("4h__sma_p200_s0")
    atr = col("5m__atr14_raw")
    bb_up = col("5m__bb20_dev1.0_upper"); bb_mid = col("5m__bb20_dev1.0_middle")
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)

    def nz(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # node 1 — tradeability: weighted multi-TF momentum magnitude (fast 5m + slower 30m/4h). 0 = dead/chop.
    # nz EACH input so a not-yet-warm higher TF (NaN) contributes 0 instead of poisoning the whole score.
    tradeability = np.clip(0.5 * np.abs(nz(c5)) / tradeability_scale + 0.3 * np.abs(nz(c30)) / tradeability_scale
                           + 0.2 * np.abs(nz(c4h)) / tradeability_scale, 0, 1)

    # node 2 — bias: higher-TF direction = how far price is above/below the 30m & 4h SMA200, in ATR units.
    d30 = nz((close - sma30) / (atr_s + _EPS)); d4h = nz((close - sma4h) / (atr_s + _EPS))
    bias = np.clip((d30 + d4h) / 2.0 / bias_atr_scale, -1, 1)

    # node 3 — alignment: net agreement of the CCI(30) direction across 5m / 30m / 4h.
    alignment = (np.sign(nz(c5)) + np.sign(nz(c30)) + np.sign(nz(c4h))) / 3.0

    # node 4 — strength (graded ladder, |5m CCI30| 0->160 = 0->1) and exhaustion (only beyond ~160 lights up).
    strength = np.clip(np.abs(c5) / strength_level, 0, 1)
    exhaustion = np.clip((np.abs(c5) - strength_level) / exhaustion_span, 0, 1)

    # node 4 — location: 5m band position (extension vs pullback). +1 ~ riding the upper band (extended up),
    # -1 ~ lower band. The policy learns "with-trend + pulled back (location toward 0/opposite) = cheap entry."
    half = bb_up - bb_mid
    location = np.clip(nz((close - bb_mid) / (np.where(np.abs(half) > _EPS, half, np.nan) + _EPS)), -1.5, 1.5) / 1.5

    cser = pd.Series(close)
    # node 4 — structure: where price sits in its recent range (0 = at lows, 1 = at highs -> near a breakout).
    rmin = cser.rolling(structure_win, min_periods=2).min().to_numpy()
    rmax = cser.rolling(structure_win, min_periods=2).max().to_numpy()
    structure = np.clip(nz((close - rmin) / (rmax - rmin + _EPS)), 0, 1)

    # node 5 — persistence: how one-directional recent price has been (|mean sign of close changes| over ~30).
    sgn = np.sign(np.diff(close, prepend=close[0]))
    persistence = np.clip(np.abs(pd.Series(sgn).rolling(persistence_win, min_periods=2).mean().to_numpy()), 0, 1)

    # node 6 — decay: 5m CCI rolled back from its recent peak -> momentum dying (1 = fully reverted to flat).
    peak = pd.Series(np.abs(c5)).rolling(decay_win, min_periods=2).max().to_numpy()
    decay = np.clip(1.0 - np.abs(c5) / (peak + _EPS), 0, 1)

    out = np.stack([tradeability, bias, alignment, strength, exhaustion,
                    location, structure, persistence, decay], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
