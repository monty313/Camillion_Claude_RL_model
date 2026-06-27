# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Vectorised parallel envs (SubprocVecEnv) for CPU-bound speed.
# WHERE src/training/vector_env_factory.py | HOW lazy SB3 import; N_ENVS workers.
# DEPENDS_ON config/training_speed_config.py, src/training/gym_adapter.py
# USED_BY src/training/trainer.py.
"""SubprocVecEnv factory (Colab-runnable; lazy SB3 import)."""
from __future__ import annotations
from config import training_speed_config as TS
from src.training.gym_adapter import make_gym_env


def make_vec_env(indicators, close, time_ns, registry_factory, n_envs: int | None = None,
                 **env_kwargs):
    """registry_factory() -> a fresh AlphaRegistry per worker (no shared state)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    n = n_envs or TS.N_ENVS
    random_window = bool(env_kwargs.pop("random_window", TS.RANDOM_WINDOW_TRAINING))

    def _thunk(seed):
        def _f():
            return make_gym_env(indicators, close, time_ns, registry_factory(),
                                seed=seed, random_window=random_window,
                                **env_kwargs)
        return _f

    backend = SubprocVecEnv if TS.VEC_ENV_BACKEND == "subproc" else DummyVecEnv
    return backend([_thunk(i) for i in range(n)])


def make_multi_symbol_vec_env(symbol_data: dict, registry_factory, n_envs: int | None = None,
                              **env_kwargs):
    """ONE policy across MANY symbols. Spread N workers ROUND-ROBIN over
    `symbol_data = {symbol: (indicators, close, time_ns)}`. Each worker is tagged with its
    symbol (so the cross-asset features + per-asset calibrated size are correct), and rewards
    stay comparable across symbols because each asset is sized to ~2.5%/day. This is the
    'one bot trades everything' training path -- the bridge toward the portfolio."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from config import asset_specs as A
    n = n_envs or TS.N_ENVS
    random_window = bool(env_kwargs.pop("random_window", TS.RANDOM_WINDOW_TRAINING))
    syms = list(symbol_data.keys())
    if not syms:
        raise ValueError("symbol_data is empty -- pass {symbol: (indicators, close, time_ns)}")

    def _thunk(seed, sym):
        ind, close, tm = symbol_data[sym]
        kw = dict(env_kwargs)
        if "position_size" not in kw and sym in A.SPECS:
            kw["position_size"] = A.calibrated_position_size(sym)   # per-asset calibrated size
        def _f():
            return make_gym_env(ind, close, tm, registry_factory(), symbol=sym,
                                seed=seed, random_window=random_window, **kw)
        return _f

    backend = SubprocVecEnv if TS.VEC_ENV_BACKEND == "subproc" else DummyVecEnv
    return backend([_thunk(i, syms[i % len(syms)]) for i in range(n)])


def make_portfolio_vec_env(symbol_data: dict, registry_factory, n_envs: int | None = None, **env_kwargs):
    """ONE bot, ONE shared pot, ALL symbols at once -- the true portfolio trainer. Every worker is a
    full PortfolioEnv over `symbol_data = {symbol: (indicators, close, time_ns)}` (time-aligned), so the
    single policy learns to BALANCE risk across simultaneous positions in one account, and scales to the
    full FTMO broker list without changing the locked 479 observation."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.training.gym_adapter import make_portfolio_gym_env
    from src.env.portfolio_env import build_portfolio_subs
    # IMPORTANT: the aligned portfolio arrays are LARGE (every symbol x the full history -> gigabytes).
    # SubprocVecEnv would PICKLE the whole dataset to EACH worker (gigabytes x N_ENVS) -> on Colab that
    # OOM/thrashes and HANGS before training starts. DummyVecEnv runs the envs in ONE process so they
    # SHARE the arrays BY REFERENCE (one copy total) and start immediately.
    n = n_envs or min(4, TS.N_ENVS)
    # EPISODE DIVERSITY: each worker gets a DIFFERENT seed and a random training window, so the N
    # DummyVecEnv workers explore DIFFERENT stretches of history instead of replaying the SAME
    # trajectory (identical workers = no exploration diversity = wasted parallel envs).
    random_window = bool(env_kwargs.pop("random_window", TS.RANDOM_WINDOW_TRAINING))
    window = env_kwargs.pop("window", TS.WINDOW_LENGTH_BARS)
    warmup = env_kwargs.get("warmup", 200)
    cfg = env_kwargs.get("cfg", None)
    # BUILD ONCE, SHARE ACROSS WORKERS: the per-symbol precompute (alphas/streaks over the whole history)
    # is the slow + memory-heavy part. Doing it ONCE and sharing the read-only result across the N workers
    # cuts build time AND memory ~N-fold -- the fix for the multi-hour "stuck building" hang (was rebuilding
    # 4 symbols x N workers = 16 times). The shared sub-envs are read-only, so this is safe in one process.
    print(f"      building shared features for {len(symbol_data)} symbols ONCE (shared by all {n} workers)...",
          flush=True)
    subs = build_portfolio_subs(symbol_data, registry_factory, cfg=cfg, warmup=warmup, progress=True)

    def _thunk(seed):
        def _f():
            return make_portfolio_gym_env(None, registry_factory, subs=subs, seed=seed,
                                          random_window=random_window, window=window, **env_kwargs)
        return _f

    return DummyVecEnv([_thunk(i) for i in range(n)])
