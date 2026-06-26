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


def summarize(alpha_values, occupancy_mask, directional_mask=None) -> np.ndarray:
    """Return float32 [buy_pct, sell_pct, active_pct, net_signal_pct].

    buy/sell/net are over ACTIVE (firing) alphas; active_pct is over ASSIGNED
    strategies. All are 0.0 when their denominator is 0 (nothing active/assigned).
    net_signal_pct in [-1, +1]: +1 all-buy, -1 all-sell, 0 balanced/none.

    This is the DIRECTIONAL consensus. If `directional_mask` is given, ONLY directional
    slots are counted (numerator AND denominator), so a non-directional 1/0 gate (e.g. a
    movement filter) is excluded -- its 1 is not a buy. directional_mask=None = count all
    occupied slots (legacy behaviour; identical when every alpha is directional).
    """
    av = np.asarray(alpha_values, dtype=np.float32).ravel()
    om = np.asarray(occupancy_mask, dtype=np.float32).ravel()
    keep = (np.asarray(directional_mask, dtype=np.float32).ravel() > 0
            if directional_mask is not None else om > 0)
    avd = av[keep]
    assigned = float(keep.sum())
    buy = float(np.count_nonzero(avd == 1))
    sell = float(np.count_nonzero(avd == -1))
    active = buy + sell
    buy_pct = buy / active if active > 0 else 0.0
    sell_pct = sell / active if active > 0 else 0.0
    active_pct = active / assigned if assigned > 0 else 0.0
    net_pct = (buy - sell) / active if active > 0 else 0.0
    return np.array([buy_pct, sell_pct, active_pct, net_pct], dtype=np.float32)


def net_balance(alpha_values, directional_mask=None) -> float:
    """Net DIRECTIONAL signal balance in [-1,+1] (for signal memory + accuracy).

    If `directional_mask` is given, only directional slots count -- a non-directional
    gate's 1/0 never moves the balance (it is not a buy/sell vote).
    """
    av = np.asarray(alpha_values, dtype=np.float32).ravel()
    if directional_mask is not None:
        av = av[np.asarray(directional_mask, dtype=np.float32).ravel() > 0]
    buy = float(np.count_nonzero(av == 1))
    sell = float(np.count_nonzero(av == -1))
    active = buy + sell
    return (buy - sell) / active if active > 0 else 0.0
