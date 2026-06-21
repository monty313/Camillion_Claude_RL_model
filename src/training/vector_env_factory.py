# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Build vectorised parallel envs (SubprocVecEnv) for CPU-bound speed.
# WHERE src/training/vector_env_factory.py | HOW Phase-1 spins N_ENVS workers.
# DEPENDS_ON config/training_speed_config.py, src/env/trading_env.py
# USED_BY src/training/trainer.py.
"""Vectorised env factory (Phase-0 placeholder)."""
from __future__ import annotations

def make_vec_env(*args, **kwargs):
    raise NotImplementedError("Phase 1: SubprocVecEnv with N_ENVS workers.")
