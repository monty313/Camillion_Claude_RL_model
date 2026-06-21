# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Fixed-size alpha slot system: MAX_STRATEGIES (64) slots that never
#      change length, so adding strategies NEVER changes the observation.
# WHERE src/strategies/registry.py
# HOW  A list of 64 slots (None = empty). collect_signals() returns a length-64
#      vector of +1/-1/0; occupancy_mask() returns 1=assigned / 0=empty so the
#      bot can tell an EMPTY slot apart from an assigned-but-INACTIVE one.
# DEPENDS_ON: config/constants.py, src/strategies/base.py, numpy
# USED_BY: src/signals/signal_summary.py, src/observation/builder.py,
#          tests/test_strategy_registry_shape.py
# CHANGE_NOTES(IRAC): I: strategy count must not change obs shape; empty != 0.
#   R: spec fixed 64 slots + Monty "empty slot = no strategy assigned".
#   A: fixed list + parallel occupancy mask. C: scale-stable observation ->
#   one policy keeps training as the alpha library grows.
# =====================================================================
"""StrategyRegistry: 64 fixed alpha slots + occupancy mask (empty vs inactive)."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.strategies.base import BaseStrategy


class StrategyRegistry:
    """Holds up to MAX_STRATEGIES alphas in FIXED slots. Length never changes."""

    def __init__(self, max_slots: int = C.MAX_STRATEGIES) -> None:
        self.max_slots = int(max_slots)
        self._slots: list[BaseStrategy | None] = [None] * self.max_slots

    # --- registration ---------------------------------------------------
    def register(self, strategy: BaseStrategy, slot: int | None = None) -> int:
        """Assign `strategy` to a slot (first free slot if `slot` is None)."""
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must be a BaseStrategy instance")
        if slot is None:
            slot = self._first_free_slot()
        if not (0 <= slot < self.max_slots):
            raise IndexError(f"slot {slot} out of range [0,{self.max_slots})")
        if self._slots[slot] is not None:
            raise ValueError(f"slot {slot} already occupied by {self._slots[slot].name}")
        self._slots[slot] = strategy
        return slot

    def unregister(self, slot: int) -> None:
        self._slots[slot] = None

    def _first_free_slot(self) -> int:
        for i, s in enumerate(self._slots):
            if s is None:
                return i
        raise RuntimeError(f"all {self.max_slots} strategy slots are full")

    # --- introspection --------------------------------------------------
    @property
    def assigned_count(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    def occupancy_mask(self) -> np.ndarray:
        """Length-64 float32: 1.0 if a strategy is assigned, else 0.0."""
        return np.array([1.0 if s is not None else 0.0 for s in self._slots],
                        dtype=np.float32)

    # --- signal collection ---------------------------------------------
    def collect_signals(self, ctx) -> np.ndarray:
        """Length-64 float32 of +1/-1/0. EMPTY slots are 0 (mask tells them apart)."""
        out = np.zeros(self.max_slots, dtype=np.float32)
        for i, s in enumerate(self._slots):
            if s is not None:
                out[i] = float(s.signal(ctx))
        return out
