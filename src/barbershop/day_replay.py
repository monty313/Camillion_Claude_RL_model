# WHEN 2026-06-21 (Phase 2) | WHO Claude for Monty
# WHY  Barbershop #2: replay a trading day — signal pressure + trades + risk
#      events bar by bar — so you can see HOW a day unfolded.
# WHERE src/barbershop/day_replay.py | HOW pure function over per-bar arrays.
# DEPENDS_ON numpy | USED_BY jarvis cockpit (Phase 2), tests.
"""Day Replay: per-bar timeline of pressure, equity, position and risk events."""
from __future__ import annotations
import numpy as np


def build_replay(time_ns, net_signal, equity, position, *, breach_flags=None,
                 target_flags=None) -> dict:
    """Return {bars:[...], summary:{...}} for one replayed segment."""
    t = np.asarray(time_ns).ravel()
    net = np.asarray(net_signal, dtype=float).ravel()
    eq = np.asarray(equity, dtype=float).ravel()
    pos = np.asarray(position, dtype=float).ravel()
    n = len(eq)
    bf = np.zeros(n, bool) if breach_flags is None else np.asarray(breach_flags, bool)
    tf = np.zeros(n, bool) if target_flags is None else np.asarray(target_flags, bool)
    peak = np.maximum.accumulate(eq) if n else eq
    bars = []
    for i in range(n):
        ev = "breach" if bf[i] else ("target" if tf[i] else "")
        bars.append({"i": i, "net": float(net[i]) if i < len(net) else 0.0,
                     "equity": float(eq[i]), "pos": float(pos[i]) if i < len(pos) else 0.0,
                     "dd_from_peak_pct": float((peak[i] - eq[i]) / peak[i] * 100) if peak[i] else 0.0,
                     "event": ev})
    summary = {
        "bars": n, "start_equity": float(eq[0]) if n else 0.0,
        "end_equity": float(eq[-1]) if n else 0.0,
        "max_equity": float(eq.max()) if n else 0.0, "min_equity": float(eq.min()) if n else 0.0,
        "max_drawdown_pct": float(((peak - eq) / peak * 100).max()) if n else 0.0,
        "breaches": int(bf.sum()), "target_hits": int(tf.sum()),
        "pct_in_position": float(np.mean(pos != 0)) if n else 0.0,
    }
    return {"bars": bars, "summary": summary}
