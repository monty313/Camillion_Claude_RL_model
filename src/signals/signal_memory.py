# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Give the bot the net signal balance from the LAST 5 BARS so it can see
#      pressure building / fading / flipping. Always exactly 5 values.
# WHERE src/signals/signal_memory.py
# HOW  A deque (lag0 = current bar, lag4 = 4 bars ago). Plus a pure-function
#      helper to slice last-5 from a precomputed net-signal series.
# DEPENDS_ON: config/constants.py, numpy
# USED_BY: src/observation/builder.py, tests/test_signal_memory_last5.py
# CHANGE_NOTES(IRAC): I: need short-term pressure memory of fixed width. R:
#   spec "signal_balance_lag_0..4, each -1..+1". A: deque + slice helper.
#   C: fixed 5-wide memory keeps obs shape constant and shows momentum.
# =====================================================================
"""Last-5-bar net signal balance memory (always exactly 5 values, -1..+1)."""
from __future__ import annotations
from collections import deque
import numpy as np
from config import constants as C


class SignalMemory:
    """Rolling buffer of the last N net-balance values (lag0 = newest)."""

    def __init__(self, lags: int = C.SIGNAL_MEMORY_LAGS) -> None:
        self.lags = int(lags)
        self._buf: deque[float] = deque([0.0] * self.lags, maxlen=self.lags)

    def reset(self) -> None:
        self._buf = deque([0.0] * self.lags, maxlen=self.lags)

    def push(self, net_balance: float) -> None:
        """Add the current bar's net balance; oldest falls off."""
        self._buf.appendleft(float(net_balance))

    def as_vector(self) -> np.ndarray:
        """float32 [lag0, lag1, ..., lag_{N-1}] -- always length N."""
        return np.array(list(self._buf), dtype=np.float32)


def last5_from_series(net_series, t: int, lags: int = C.SIGNAL_MEMORY_LAGS) -> np.ndarray:
    """Slice [net[t], net[t-1], ..., net[t-lags+1]] (zeros before the start)."""
    arr = np.zeros(int(lags), dtype=np.float32)
    for k in range(int(lags)):
        idx = t - k
        if idx >= 0:
            arr[k] = float(net_series[idx])
    return arr
