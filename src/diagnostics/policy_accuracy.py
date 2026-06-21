# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Policy's OWN directional accuracy at 1/3/10 bars, defined the same leak-
#      free way as alpha accuracy so we can compare the policy to the alphas.
# WHERE src/diagnostics/policy_accuracy.py
# HOW  Map each action to a directional call: BUY->+1, SELL->-1, HOLD/CLOSE->0
#      (0 = no directional bet, not graded). Grade vs sign(close[t+h]-close[t]);
#      at bar t only calls X<=t-h count. Reuses the leak-free primitive.
# DEPENDS_ON: config/constants.py, config/variables.py, src/signals/signal_accuracy.py
# USED_BY: src/barbershop/policy_doctor.py, tests.
"""Policy directional accuracy at 1/3/10 bars (leak-free; mirrors alpha accuracy)."""
from __future__ import annotations
import numpy as np
from config import constants as C
from config import variables as V
from src.signals.signal_accuracy import rolling_accuracy_counts

HORIZONS = (1, 3, 10)


def action_to_direction(actions) -> np.ndarray:
    """BUY->+1, SELL->-1, HOLD/CLOSE->0 (0 = no directional bet)."""
    a = np.asarray(actions).ravel()
    d = np.zeros(a.shape[0], dtype=np.float64)
    d[a == C.ACTION_BUY] = 1.0
    d[a == C.ACTION_SELL] = -1.0
    return d


def policy_directional_accuracy(actions, close, window: int | None = None,
                                horizons=HORIZONS) -> dict:
    """Return {horizon: (accuracy(T,), valid_count(T,))}, leak-free."""
    w = V.SIGNAL_ACCURACY_WINDOW if window is None else int(window)
    d = action_to_direction(actions)
    return {h: rolling_accuracy_counts(d, close, w, h) for h in horizons}
