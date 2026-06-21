# =====================================================================
# WHEN 2026-06-21 (Phase 0; Phase 1 renamed to lock strategy/alpha split)
# WHO Claude for Monty
# WHY  The fixed-size ALPHA SLOT system. Strategies (internal logic) are
#      registered into 64 fixed slots; the policy only ever sees each slot's
#      ALPHA OUTPUT + the occupancy mask -- never the strategy internals.
# WHERE src/strategies/registry.py
# HOW  64 slots (None = empty). collect_alphas() -> +1/-1/0 per slot;
#      occupancy_mask() -> 1 assigned / 0 empty.
# DEPENDS_ON: config/constants.py, src/strategies/base.py, numpy
# USED_BY: src/signals/signal_summary.py, src/observation/builder.py, tests.
# CHANGE_NOTES(IRAC): I: 'strategy' and 'alpha' were used loosely. R: operator
#   contract 2026-06-21 (strategy=logic, alpha=exposed output, policy=RL agent).
#   A: renamed StrategyRegistry->AlphaRegistry, collect_signals->collect_alphas
#   (aliases kept); locked the semantics in the docstring. C: the policy can
#   never accidentally 'see the strategy' -> clean meta-learning over alphas.
# =====================================================================
"""AlphaRegistry: 64 fixed ALPHA SLOTS filled by strategies; exposes alpha outputs."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.strategies.base import BaseStrategy


class AlphaRegistry:
    """Up to MAX_STRATEGIES strategies in FIXED alpha slots. Length never changes.

    CONTRACT (locked 2026-06-21):
      * strategy = the INTERNAL logic (any indicators/rules); a BaseStrategy.
        The policy NEVER sees a strategy's internals.
      * alpha    = a strategy's EXPOSED OUTPUT for its slot:
                     +1 = active BUY alpha
                     -1 = active SELL alpha
                      0 = alpha assigned but currently inactive / no setup
                   empty slot = no alpha assigned yet (occupancy mask = 0)
        These four states are distinct; NONE of them is the action-space HOLD
        (alpha 0 lives in alpha-space; ACTION HOLD lives in action-space).
      * The observation receives only collect_alphas() + occupancy_mask()
        (+ the alpha_summary percentages) -- never strategy logic.
    """

    def __init__(self, max_slots: int = C.MAX_STRATEGIES) -> None:
        self.max_slots = int(max_slots)
        self._slots: list[BaseStrategy | None] = [None] * self.max_slots

    # --- registration: put a STRATEGY into an alpha slot --------------
    def register(self, strategy: BaseStrategy, slot: int | None = None) -> int:
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("register() takes a BaseStrategy (the internal logic)")
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
        raise RuntimeError(f"all {self.max_slots} alpha slots are full")

    @property
    def assigned_count(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    def occupancy_mask(self) -> np.ndarray:
        """Length-64 float32: 1.0 if an alpha is assigned to the slot, else 0.0."""
        return np.array([1.0 if s is not None else 0.0 for s in self._slots],
                        dtype=np.float32)

    def collect_alphas(self, ctx) -> np.ndarray:
        """Length-64 float32 of alpha OUTPUTS (+1/-1/0). Empty slots are 0
        (the occupancy mask tells empty apart from assigned-but-inactive)."""
        out = np.zeros(self.max_slots, dtype=np.float32)
        for i, s in enumerate(self._slots):
            if s is not None:
                out[i] = float(s.signal(ctx))
        return out

    # backward-compatible alias
    collect_signals = collect_alphas


# backward-compatible alias (old name)
StrategyRegistry = AlphaRegistry
