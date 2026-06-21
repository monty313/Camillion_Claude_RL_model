# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Sample random training windows (not always sequential) for robustness.
# WHERE src/training/random_window_sampler.py | HOW Phase-1 picks [t0,t0+W]
#      with W=WINDOW_LENGTH_BARS after MIN_WARMUP_BARS of history.
# DEPENDS_ON config/training_speed_config.py | USED_BY src/training/trainer.py.
"""Random-window sampler (Phase-0 placeholder)."""
from __future__ import annotations

def sample_window(*args, **kwargs):
    raise NotImplementedError("Phase 1: random-window sampling over cached data.")
