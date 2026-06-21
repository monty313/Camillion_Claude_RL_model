# WHEN 2026-06-21 (Phase 2) | WHO Claude for Monty
# WHY  Barbershop #5: per-alpha stats — activity, 1/3/10-bar accuracy, agreement
#      with consensus, and how often the alphas conflict.
# WHERE src/barbershop/signal_doctor.py | HOW reuses the leak-free accuracy engine.
# DEPENDS_ON src/signals/alpha_accuracy.py, numpy | USED_BY jarvis cockpit, tests.
"""Signal Doctor: per-alpha activity/accuracy + conflict detection (diagnostics)."""
from __future__ import annotations
import numpy as np
from src.signals.alpha_accuracy import per_alpha_accuracy


def report(alpha_matrix, close, occupancy, *, window=None, min_samples=5) -> dict:
    am = np.asarray(alpha_matrix, dtype=float)
    if am.ndim == 1:
        am = am[:, None]
    T, n = am.shape
    occ = np.asarray(occupancy, bool).ravel()
    acc, cnt = per_alpha_accuracy(am, close, window)
    net = np.sign(am.sum(axis=1))
    alphas = []
    for a in range(n):
        if not occ[a]:
            continue
        col = am[:, a]
        active = col != 0
        agree = float(np.mean(np.sign(col[active]) == net[active])) if active.any() else 0.0
        alphas.append({"slot": a, "activity": float(active.mean()),
                       "acc_1": float(acc[1][-1, a]), "acc_3": float(acc[3][-1, a]),
                       "acc_10": float(acc[10][-1, a]),
                       "n_samples_3": int(cnt[3][-1, a]),
                       "agreement_with_consensus": agree})
    # conflict = fraction of bars where assigned alphas disagree (both +1 and -1 present)
    has_buy = ((am == 1) & occ[None, :]).any(axis=1)
    has_sell = ((am == -1) & occ[None, :]).any(axis=1)
    conflict_rate = float(np.mean(has_buy & has_sell)) if T else 0.0
    return {"alphas": alphas, "n_assigned": int(occ.sum()),
            "conflict_rate": conflict_rate,
            "mean_activity": float(np.mean([a["activity"] for a in alphas])) if alphas else 0.0}
