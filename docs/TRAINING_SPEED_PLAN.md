# TRAINING SPEED PLAN  (top priority)

Goal: maximize challenge episodes per wall-clock minute. The policy is a tiny MLP, so
the bottleneck is **CPU env-stepping**, not the GPU (this matched Quantra).

## 1. Precompute indicators ONCE → cache
`src/data/cache_builder.py` (Phase 1) computes every indicator for every timeframe via
`src/indicators/base.py` and writes **float32** to **parquet** and/or a **numpy memmap**
(`config/training_speed_config.py: CACHE_FORMAT/USE_MEMMAP`). MT5/TA-Lib run here, once.

## 2. env.step() only reads the cache
The hot loop assembles the 357-vector from **cached float32** via
`src/observation/builder.py`. **No TA-Lib, no MT5, no pandas in step()**
(`NO_TALIB_IN_STEP`, `NO_MT5_IN_STEP`; asserted in Phase-1 tests).

## 3. Random-window training
`src/training/random_window_sampler.py` samples `[t0, t0+WINDOW_LENGTH_BARS]` after
`MIN_WARMUP_BARS` (≥200, enough for SMA200/BB200) so episodes aren't always sequential.

## 4. Vectorized parallel envs
`src/training/vector_env_factory.py` builds **SubprocVecEnv** with `N_ENVS` workers,
auto-scaled toward ~80% device utilization.

## 5. Device policy
`PREFER_CPU=True`; only use GPU if it beats CPU by ≥ `GPU_SPEEDUP_THRESHOLD` (1.30×).
A near-tie does not justify a paid accelerator.

## 6. float32 everywhere
`config.variables.FLOAT = np.float32`; observations are float32; caches are float32.

## 7. No Python loops in indicator math
SMA uses cumulative sums; Phase-1 CCI/RSI/Bollinger use vectorized TA-Lib over arrays.

## What happens during env.step() (Phase 1)
1. advance bar pointer → 2. read cached indicators (190) + cached alpha signals (64) →
3. compute summary/memory/accuracy/account/portfolio/time blocks (cheap numpy) →
4. `builder.build_from_blocks()` → 5. risk/breach check → 6. reward → 7. return obs.
