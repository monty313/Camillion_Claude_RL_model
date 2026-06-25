# =====================================================================
# WHEN 2026-06-22 | WHO Claude for Monty
# WHY  Diagnostic: does each alpha actually PREDICT direction? Measures forward
#      return conditioned on each alpha's signal (long-minus-short edge + hit
#      rate). Used to tell real signal from decoration. NOT used in training.
# WHERE src/barbershop/alpha_edge.py
# =====================================================================
"""Per-alpha forward-edge diagnostic (which alphas predict direction)."""
from __future__ import annotations
import numpy as np


def per_alpha_edge(alpha_matrix, close, horizon: int = 240) -> list[dict]:
    """For each alpha slot: forward `horizon`-bar return conditioned on its signal.
    edge = mean(fwd|+1) - mean(fwd|-1); hit = sign(fwd)==sign(signal) rate."""
    am = np.asarray(alpha_matrix); close = np.asarray(close, dtype=float)
    T, K = am.shape
    fwd = np.full(T, np.nan)
    if T > horizon:
        fwd[:T - horizon] = close[horizon:] / close[:T - horizon] - 1.0
    out = []
    for k in range(K):
        sig = am[:, k]
        active = (sig != 0) & np.isfinite(fwd)
        longs = (sig == 1) & np.isfinite(fwd); shorts = (sig == -1) & np.isfinite(fwd)
        ml = float(fwd[longs].mean()) if longs.any() else 0.0
        ms = float(fwd[shorts].mean()) if shorts.any() else 0.0
        hit = float((np.sign(fwd[active]) == np.sign(sig[active])).mean()) if active.any() else 0.0
        out.append({"slot": k, "fire_rate": float(active.mean()), "n": int(active.sum()),
                    "mean_fwd_long": ml, "mean_fwd_short": ms, "edge": ml - ms, "hit_rate": hit})
    return out


def edge_table(report, names=None) -> str:
    """One line per alpha: fire%, count, edge in basis points, hit%, name."""
    lines = ["slot  fire%      n     edge(bps)  hit%  name"]
    for r in report:
        nm = names[r["slot"]] if names and r["slot"] < len(names) else ""
        lines.append(f"{r['slot']:>3}  {r['fire_rate']*100:5.1f}  {r['n']:>7}  "
                     f"{r['edge']*1e4:+8.2f}  {r['hit_rate']*100:5.1f}  {nm}")
    return "\n".join(lines)
