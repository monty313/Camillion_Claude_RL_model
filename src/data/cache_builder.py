# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Precompute ALL indicators ONCE and cache to parquet/numpy memmap so
#      env.step() only reads cached float32 -- the core training-speed trick.
# WHERE src/data/cache_builder.py | HOW Phase-1 computes per-TF indicators via
#      src/indicators/base.py and writes float32 caches.
# DEPENDS_ON src/indicators/base.py, config/training_speed_config.py
# USED_BY src/training/*, src/env/trading_env.py (Phase 1).
"""Indicator cache builder (Phase-0 placeholder; the heart of fast training)."""
from __future__ import annotations

def build_cache(*args, **kwargs):
    """Precompute + persist indicator caches. PHASE-0 STUB."""
    raise NotImplementedError("Phase 1: precompute indicators -> parquet/memmap float32.")
