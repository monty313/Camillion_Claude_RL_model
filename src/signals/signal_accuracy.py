# =====================================================================
# WHEN 2026-06-21 (Phase 0; Phase 1 added counts + 10-bar reuse) | WHO Claude
# WHY  Rolling directional accuracy of a signal vs realised price move, with
#      NO look-ahead leakage. The leak-free primitive reused for net-signal,
#      per-alpha, and policy accuracy.
# WHERE src/signals/signal_accuracy.py
# HOW  Signal at bar X is graded vs sign(close[X+h]-close[X]); at decision bar t
#      only X<=t-h are counted (outcome already known by t). out[t] depends on
#      close up to t only -> bars > t cannot change it.
# DEPENDS_ON: config/variables.py, numpy
# USED_BY: signal_memory/builder, src/signals/alpha_accuracy.py,
#          src/diagnostics/policy_accuracy.py, tests.
# CHANGE_NOTES(IRAC): I: aggregates need to know how many graded samples each
#   series has (to exclude 'no data' from mean/best). R: operator diagnostics
#   spec. A: add rolling_accuracy_counts -> (acc, valid_count). C: honest
#   reliability aggregates, still leak-free.
# =====================================================================
"""Leak-free rolling directional accuracy (+ valid-sample counts)."""
from __future__ import annotations
import numpy as np
from config import variables as V


def rolling_accuracy_counts(signal, close, window: int, horizon: int):
    """Return (accuracy, valid_count) per bar, leak-free.

    accuracy[t] = fraction of directional signals X in (t-h-window, t-h] whose
    sign matched sign(close[X+h]-close[X]); valid_count[t] = how many graded,
    directional signals are in that window. out[t] uses close only up to t.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]
    acc = np.zeros(T, dtype=np.float32)
    cnt = np.zeros(T, dtype=np.float32)
    if T == 0:
        return acc, cnt
    h = int(horizon)
    correct = np.zeros(T, dtype=np.float64)
    valid = np.zeros(T, dtype=np.float64)
    if T > h:
        sX = np.sign(sig[: T - h])
        future_dir = np.sign(close[h:] - close[: T - h])
        graded = sX != 0
        correct[: T - h] = ((sX == future_dir) & graded).astype(np.float64)
        valid[: T - h] = graded.astype(np.float64)
    cc = np.concatenate(([0.0], np.cumsum(correct)))
    cv = np.concatenate(([0.0], np.cumsum(valid)))
    t = np.arange(T)
    hi = t - h
    lo = np.maximum(0, hi - int(window) + 1)
    mask = hi >= 0
    hi_c = np.clip(hi + 1, 0, T)
    lo_c = np.clip(lo, 0, T)
    num = np.where(mask, cc[hi_c] - cc[lo_c], 0.0)
    den = np.where(mask, cv[hi_c] - cv[lo_c], 0.0)
    safe = np.where(den > 0, den, 1.0)
    acc = np.where(den > 0, num / safe, 0.0).astype(np.float32)
    cnt = den.astype(np.float32)
    return acc, cnt


def rolling_accuracy(signal, close, window: int, horizon: int) -> np.ndarray:
    """Leak-free rolling accuracy only (see rolling_accuracy_counts)."""
    return rolling_accuracy_counts(signal, close, window, horizon)[0]


def accuracy_features(net_signal, close, window: int | None = None, t: int | None = None):
    """[acc_1bar, acc_3bar] for the net signal (observation block). Leak-free."""
    w = V.SIGNAL_ACCURACY_WINDOW if window is None else int(window)
    a1 = rolling_accuracy(net_signal, close, w, 1)
    a3 = rolling_accuracy(net_signal, close, w, 3)
    if t is None:
        return np.stack([a1, a3], axis=1).astype(np.float32)
    return np.array([a1[t], a3[t]], dtype=np.float32)
