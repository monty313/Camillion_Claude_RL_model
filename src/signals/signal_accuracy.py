# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Tell the bot how accurate past net signals were -- using ONLY outcomes
#      that are already complete. NEVER let bar t see its own future.
# WHERE src/signals/signal_accuracy.py
# HOW  A net signal at bar X is "correct" if sign(net[X]) matches
#      sign(close[X+h]-close[X]). At bar t we only count signals with outcome
#      known by t, i.e. X <= t-h (1-bar uses up to t-1, 3-bar up to t-3).
#      Vectorised with cumulative sums; output[t] depends on close up to t only.
# DEPENDS_ON: config/variables.py (SIGNAL_ACCURACY_WINDOW), numpy
# USED_BY: src/observation/builder.py, tests/test_signal_accuracy_no_leakage.py
# CHANGE_NOTES(IRAC): I: accuracy features must not leak the future. R: spec
#   "no look-ahead leakage; rolling window=100". A: grade at X+h, expose only
#   X<=t-h, prove out[t] ignores bars>t. C: leak-free features = honest
#   generalisation and a trustworthy FTMO pass estimate.
# =====================================================================
"""Rolling past-signal accuracy with NO look-ahead leakage (1-bar & 3-bar)."""
from __future__ import annotations
import numpy as np
from config import variables as V


def rolling_accuracy(net_signal, close, window: int, horizon: int) -> np.ndarray:
    """Per-bar rolling accuracy of net signals at the given horizon (no leakage).

    out[t] = fraction of directional net signals X in (t-h-window, t-h] whose
    sign matched sign(close[X+h]-close[X]). out[t] uses close only up to index t,
    so modifying any bar > t cannot change out[t]. Returns float32 length T,
    0.0 where no graded signal exists yet.
    """
    net = np.asarray(net_signal, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]
    out = np.zeros(T, dtype=np.float32)
    if T == 0:
        return out
    h = int(horizon)
    correct = np.zeros(T, dtype=np.float64)  # 1 if signal at X correct (else 0)
    valid = np.zeros(T, dtype=np.float64)    # 1 if signal at X is gradable & directional
    if T > h:
        sX = np.sign(net[: T - h])                      # signal sign at index X
        future_dir = np.sign(close[h:] - close[: T - h])  # realised dir over h bars
        graded = sX != 0
        correct[: T - h] = ((sX == future_dir) & graded).astype(np.float64)
        valid[: T - h] = graded.astype(np.float64)
    cc = np.concatenate(([0.0], np.cumsum(correct)))
    cv = np.concatenate(([0.0], np.cumsum(valid)))
    t = np.arange(T)
    hi = t - h                                   # last gradable index visible at t
    lo = np.maximum(0, hi - int(window) + 1)
    mask = hi >= 0
    hi_c = np.clip(hi + 1, 0, T)
    lo_c = np.clip(lo, 0, T)
    num = np.where(mask, cc[hi_c] - cc[lo_c], 0.0)
    den = np.where(mask, cv[hi_c] - cv[lo_c], 0.0)
    safe = np.where(den > 0, den, 1.0)
    out = np.where(den > 0, num / safe, 0.0).astype(np.float32)
    return out


def accuracy_features(net_signal, close, window: int | None = None, t: int | None = None):
    """[acc_1bar, acc_3bar] -- at bar t (length-2) or the full series ((T,2))."""
    w = V.SIGNAL_ACCURACY_WINDOW if window is None else int(window)
    a1 = rolling_accuracy(net_signal, close, w, 1)
    a3 = rolling_accuracy(net_signal, close, w, 3)
    if t is None:
        return np.stack([a1, a3], axis=1).astype(np.float32)
    return np.array([a1[t], a3[t]], dtype=np.float32)
