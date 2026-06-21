# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Train / resume the PPO policy over the trading env, fast (cached data,
#      parallel envs, random windows, float32, CPU-first).
# WHERE src/training/trainer.py | HOW Phase-1 wires stable-baselines3 PPO.
# DEPENDS_ON src/training/{vector_env_factory,random_window_sampler}.py
# USED_BY notebooks/Camillion_One_Click_Train.ipynb (Phase 2).
"""Trainer (Phase-0 placeholder)."""
from __future__ import annotations

def train(*args, **kwargs):
    raise NotImplementedError("Phase 1: PPO training loop over cached features.")

def resume(*args, **kwargs):
    raise NotImplementedError("Phase 1: resume from a saved checkpoint.")
