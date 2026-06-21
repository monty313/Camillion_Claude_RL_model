# =====================================================================
# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Per-alpha rolling directional accuracy at 1/3/10 bars (leak-free) and
#      aggregate reliability (mean / best / dispersion over ASSIGNED alphas).
#      DIAGNOSTICS ONLY -- never an observation feature, never a reward term.
# WHERE src/signals/alpha_accuracy.py
# HOW  Reuses the leak-free rolling_accuracy_counts per alpha column; aggregates
#      only over assigned alphas that have >= min_samples graded outcomes.
# DEPENDS_ON: config/variables.py, src/signals/signal_accuracy.py, numpy
# USED_BY: src/barbershop/policy_doctor.py, telemetry, tests.
# CHANGE_NOTES(IRAC): I: must track many alphas' reliability without bloating
#   the observation or leaking. R: operator diagnostics spec 2026-06-21. A:
#   per-alpha (T,n) accuracy + counts; aggregates over assigned+valid alphas.
#   C: rich reliability for Policy Doctor; observation stays lean/leak-free.
# =====================================================================
"""Per-alpha 1/3/10-bar accuracy + aggregate reliability (diagnostics only)."""
from __future__ import annotations
import numpy as np
from config import variables as V
from src.signals.signal_accuracy import rolling_accuracy_counts

HORIZONS = (1, 3, 10)


def per_alpha_accuracy(alpha_matrix, close, window: int | None = None,
                       horizons=HORIZONS):
    """alpha_matrix: (T, n_alphas) of +1/-1/0.

    Returns (acc, cnt): dicts keyed by horizon -> (T, n_alphas) float32.
    Leak-free per alpha (acc[t,a] ignores bars > t).
    """
    am = np.asarray(alpha_matrix, dtype=np.float64)
    if am.ndim == 1:
        am = am[:, None]
    T, n = am.shape
    w = V.SIGNAL_ACCURACY_WINDOW if window is None else int(window)
    acc, cnt = {}, {}
    for h in horizons:
        A = np.zeros((T, n), dtype=np.float32)
        Cn = np.zeros((T, n), dtype=np.float32)
        for a in range(n):
            A[:, a], Cn[:, a] = rolling_accuracy_counts(am[:, a], close, w, h)
        acc[h], cnt[h] = A, Cn
    return acc, cnt


def aggregate_reliability(acc, cnt, occupancy_mask, min_samples: int = 5,
                          horizons=HORIZONS):
    """Aggregate per-alpha accuracy over ASSIGNED alphas with enough samples.

    Returns {h: {'mean':(T,), 'best':(T,), 'dispersion':(T,), 'n_valid':(T,)}}.
    'dispersion' is the std of accuracies across valid alphas (alpha disagreement).
    """
    occ = np.asarray(occupancy_mask, dtype=bool).ravel()
    out = {}
    for h in horizons:
        A, Cn = acc[h], cnt[h]
        T, n = A.shape
        mean = np.zeros(T, np.float32); best = np.zeros(T, np.float32)
        disp = np.zeros(T, np.float32); nval = np.zeros(T, np.float32)
        valid = (Cn >= min_samples) & occ[None, :]
        for t in range(T):
            vals = A[t, valid[t]]
            if vals.size:
                mean[t] = vals.mean(); best[t] = vals.max()
                disp[t] = vals.std(); nval[t] = vals.size
        out[h] = {"mean": mean, "best": best, "dispersion": disp, "n_valid": nval}
    return out
