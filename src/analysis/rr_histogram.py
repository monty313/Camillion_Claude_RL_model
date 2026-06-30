# =====================================================================
# WHEN 2026-06-30 (Stage 4) | WHO Claude for Monty
# WHY  The "research output" of the multi-head actor: the histogram of the R:R ratio the policy CHOOSES,
#      segmented by conviction (alignment) quartile. After training, this histogram IS the answer to "what
#      ratio works best for THESE indicators/sessions" -- information no backtester gives, because it's
#      conditioned on the full observation context. Reads the per-trade bracket log (env._bracket_log).
# WHERE src/analysis/rr_histogram.py
# HOW  Pure post-hoc analysis over the close records (which carry tp_pct/sl_pct/rr/lot_used/clamped/
#      session_active/alignment_score_at_entry/trade_won/pnl). No training dependency.
# USED_BY: tests + the Stage-4 commit (run on the fed-values data), the training notebook after a real run.
# =====================================================================
"""R:R-choice histogram from the per-trade bracket log, segmented by alignment (conviction) quartile."""
from __future__ import annotations
import numpy as np

_RR_EDGES = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf)   # the R:R buckets (matches the brief's ladder)


def _closes(bracket_log):
    return [e for e in bracket_log if e.get("event") == "close" and e.get("rr") is not None]


def rr_histogram(bracket_log, n_align_bins: int = 4) -> dict:
    """Histogram the CHOSEN R:R, overall and per alignment-quartile. Returns a dict with the bin edges, the
    overall + per-quartile counts/win-rate/mean-pnl, and a printable `text`."""
    cl = _closes(bracket_log)
    if not cl:
        return {"n_trades": 0, "text": "no bracket trades to histogram"}
    rr = np.array([float(e["rr"]) for e in cl])
    al = np.array([float(e.get("alignment_score_at_entry") or 0.0) for e in cl])
    won = np.array([1.0 if e.get("trade_won") else 0.0 for e in cl])
    pnl = np.array([float(e.get("pnl") or 0.0) for e in cl])
    clamped = np.array([1.0 if e.get("clamped") else 0.0 for e in cl])
    edges = np.array(_RR_EDGES)

    def _hist(mask):
        idx = np.clip(np.searchsorted(edges, rr[mask], side="right") - 1, 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        return counts

    # alignment quartiles (collapse to a single 'all' bin if alignment has no spread, e.g. synthetic data)
    if np.ptp(al) < 1e-9:
        quartiles = [("all", np.ones(len(rr), bool))]
    else:
        qs = np.quantile(al, np.linspace(0, 1, n_align_bins + 1))
        quartiles = []
        for q in range(n_align_bins):
            lo, hi = qs[q], qs[q + 1]
            m = (al >= lo) & (al <= hi if q == n_align_bins - 1 else al < hi)
            quartiles.append((f"align[{lo:+.2f},{hi:+.2f}]", m))

    lines = [f"R:R HISTOGRAM  ({len(cl)} bracket trades; {int(clamped.sum())} lot-clamped)  bins={list(_RR_EDGES[:-1])}+",
             f"  OVERALL: counts={_hist(np.ones(len(rr), bool)).tolist()}  "
             f"mean_rr={rr.mean():.2f}  win_rate={won.mean():.1%}  mean_pnl={pnl.mean():+.2f}"]
    per_q = {}
    for name, m in quartiles:
        if m.sum() == 0:
            continue
        c = _hist(m)
        lines.append(f"  {name}: n={int(m.sum())}  counts={c.tolist()}  mean_rr={rr[m].mean():.2f}  "
                     f"win_rate={won[m].mean():.1%}  mean_pnl={pnl[m].mean():+.2f}")
        per_q[name] = {"n": int(m.sum()), "counts": c.tolist(), "mean_rr": float(rr[m].mean()),
                       "win_rate": float(won[m].mean()), "mean_pnl": float(pnl[m].mean())}
    return {"n_trades": len(cl), "edges": list(_RR_EDGES), "overall_counts": _hist(np.ones(len(rr), bool)).tolist(),
            "mean_rr": float(rr.mean()), "win_rate": float(won.mean()), "by_alignment": per_q,
            "text": "\n".join(lines)}
