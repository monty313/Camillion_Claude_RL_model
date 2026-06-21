# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Collapse the 64 alpha slots into 4 scale-stable PERCENTAGES so the bot
#      is not confused when the number of strategies changes.
# WHERE src/signals/signal_summary.py
# HOW  buy%/sell%/net% are over ACTIVE (firing) alphas; active% is over
#      ASSIGNED strategies (occupancy mask). Empty slots never count.
# DEPENDS_ON: numpy
# USED_BY: src/observation/builder.py, src/signals/signal_memory.py,
#          tests/test_signal_percentages.py
# CHANGE_NOTES(IRAC): I: raw counts shift as strategies are added. R: spec
#   percentages buy/sell/active/net. A: fractions w/ safe zero-division.
#   C: scale-stable summary -> stable policy inputs across alpha growth.
# =====================================================================
"""Alpha summary as 4 percentages: buy%, sell%, active%, net signal%."""
from __future__ import annotations
import numpy as np


def summarize(alpha_values, occupancy_mask) -> np.ndarray:
    """Return float32 [buy_pct, sell_pct, active_pct, net_signal_pct].

    buy/sell/net are over ACTIVE (firing) alphas; active_pct is over ASSIGNED
    strategies. All are 0.0 when their denominator is 0 (nothing active/assigned).
    net_signal_pct in [-1, +1]: +1 all-buy, -1 all-sell, 0 balanced/none.
    """
    av = np.asarray(alpha_values, dtype=np.float32).ravel()
    om = np.asarray(occupancy_mask, dtype=np.float32).ravel()
    assigned = float(om.sum())
    buy = float(np.count_nonzero(av == 1))
    sell = float(np.count_nonzero(av == -1))
    active = buy + sell
    buy_pct = buy / active if active > 0 else 0.0
    sell_pct = sell / active if active > 0 else 0.0
    active_pct = active / assigned if assigned > 0 else 0.0
    net_pct = (buy - sell) / active if active > 0 else 0.0
    return np.array([buy_pct, sell_pct, active_pct, net_pct], dtype=np.float32)


def net_balance(alpha_values) -> float:
    """Just the net signal balance in [-1,+1] (for the last-5 signal memory)."""
    av = np.asarray(alpha_values, dtype=np.float32).ravel()
    buy = float(np.count_nonzero(av == 1))
    sell = float(np.count_nonzero(av == -1))
    active = buy + sell
    return (buy - sell) / active if active > 0 else 0.0
