# =====================================================================
# WHEN:   2026-06-21 (created, Phase 0)
# WHO:    Claude (Camillion build agent) for Monty
# WHY:    Central knobs for the #1 priority: training speed (caching,
#         parallel envs, random windows, dtype, GPU policy).
# WHERE:  config/training_speed_config.py
# HOW:    Plain constants consumed by the (Phase 1) cache builder, trainer,
#         and vector-env factory. The hot-loop rules here are asserted later.
# DEPENDS_ON: (os only)
# USED_BY: src/data/cache_builder.py, src/training/*, src/env/trading_env.py
# CHANGE_NOTES (IRAC):
#   I: env.step() must never call TA-Lib/MT5/pandas or training is slow.
#   R: Spec "Precompute indicators once; env.step only reads cached values;
#      vectorized envs; random windows; float32 everywhere".
#   A: Encode cache format, env count, window length, dtype, GPU thresholds.
#   C: Fast CPU-bound stepping = more FTMO challenge episodes per GPU-hour.
# =====================================================================
"""Training-speed configuration (caching / parallelism / dtype / device)."""
from __future__ import annotations
import os

# --- Cache: precompute indicators ONCE; env.step() only reads this. ---
CACHE_DIR: str = os.environ.get("CAMILLION_CACHE_DIR", "data_cache")
CACHE_FORMAT: str = "parquet"     # "parquet" | "memmap"
USE_MEMMAP: bool = True           # numpy memmap for zero-copy hot-loop reads
FLOAT_DTYPE: str = "float32"

# --- Parallel environments ---
N_ENVS: int = 8                   # SubprocVecEnv workers (auto-tuned in Phase 1)
VEC_ENV_BACKEND: str = "subproc"  # "subproc" | "dummy"

# --- Random-window training ---
RANDOM_WINDOW_TRAINING: bool = True
WINDOW_LENGTH_BARS: int = 5_000   # bars per sampled training window
MIN_WARMUP_BARS: int = 200        # history needed before t=0 (SMA200 / BB200)

# --- Device policy (tiny MLP -> CPU usually wins; mirrors Quantra) ---
PREFER_CPU: bool = True
GPU_SPEEDUP_THRESHOLD: float = 1.30   # only use GPU if >= this much faster
TARGET_DEVICE_UTILISATION: float = 0.80

# --- Hot-loop invariants (enforced by design; asserted in Phase 1 tests) ---
NO_TALIB_IN_STEP: bool = True
NO_MT5_IN_STEP: bool = True
