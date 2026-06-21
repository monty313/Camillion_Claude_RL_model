# WHEN 2026-06-21 (Phase 2) | WHO Claude for Monty
# WHY  Barbershop #3: why a trade opened — which alphas were active, the signal
#      pressure at entry, whether the trade aligned with consensus, and outcome.
# WHERE src/barbershop/trade_autopsy.py | HOW reads a ClosedTrade + bar context.
# DEPENDS_ON numpy | USED_BY jarvis cockpit (Phase 2), tests.
"""Trade Autopsy: why a trade opened and how it resolved."""
from __future__ import annotations
import numpy as np


def autopsy(trade, *, alpha_matrix=None, net_signal=None, occupancy=None,
            block_saliency=None) -> dict:
    """Explain one ClosedTrade. trade has .pnl/.is_win/.bar_index/.direction."""
    bi = getattr(trade, "bar_index", -1)
    direction = int(getattr(trade, "direction", 0))
    pressure = None
    active = []
    if net_signal is not None and 0 <= bi < len(net_signal):
        pressure = float(np.asarray(net_signal)[bi])
    if alpha_matrix is not None and 0 <= bi < len(alpha_matrix):
        row = np.asarray(alpha_matrix)[bi]
        for j, v in enumerate(row):
            if v != 0 and (occupancy is None or occupancy[j]):
                active.append({"slot": j, "alpha": int(v)})
    aligned = (pressure is not None and direction != 0 and np.sign(pressure) == np.sign(direction))
    return {
        "bar_index": int(bi), "direction": direction, "pnl": float(getattr(trade, "pnl", 0.0)),
        "outcome": "win" if getattr(trade, "is_win", False) else "loss",
        "pressure_at_entry": pressure, "alphas_active": active, "n_active": len(active),
        "aligned_with_consensus": bool(aligned),
        "block_saliency": block_saliency,
    }
