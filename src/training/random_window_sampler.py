# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Sample random training windows so episodes aren't always sequential.
# WHERE src/training/random_window_sampler.py
# DEPENDS_ON config/training_speed_config.py, numpy | USED_BY trainer/vec factory.
"""Random-window sampler over a cached series."""
from __future__ import annotations
import numpy as np
from config import training_speed_config as TS


def sample_window(n_bars: int, window: int | None = None, warmup: int | None = None,
                  rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Return [start, end) with start >= warmup and length ~window."""
    rng = rng or np.random.default_rng()
    window = window or TS.WINDOW_LENGTH_BARS
    warmup = TS.MIN_WARMUP_BARS if warmup is None else warmup
    hi = max(warmup + 1, n_bars - window - 1)
    start = int(rng.integers(warmup, hi)) if hi > warmup else warmup
    return start, min(n_bars - 1, start + window)
