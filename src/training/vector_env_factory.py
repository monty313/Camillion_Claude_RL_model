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


def make_portfolio_vec_env(symbol_data: dict, registry_factory, n_envs: int | None = None,
                           feature_cache_dir: str | None = None, data_cache_dir: str | None = None,
                           symbols=None, use_subproc: bool = False, **env_kwargs):
    """ONE bot, ONE shared pot, ALL symbols at once -- the true portfolio trainer. Every worker is a
    full PortfolioEnv over `symbol_data = {symbol: (indicators, close, time_ns)}` (time-aligned), so the
    single policy learns to BALANCE risk across simultaneous positions in one account, and scales to the
    full FTMO broker list without changing the locked 479 observation."""
    import os
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.training.gym_adapter import make_portfolio_gym_env
    from src.env.portfolio_env import build_portfolio_subs
    n = n_envs or min(4, TS.N_ENVS)
    # EPISODE DIVERSITY: each worker gets a DIFFERENT seed and a random training window, so the workers
    # explore DIFFERENT stretches of history instead of replaying the SAME trajectory.
    random_window = bool(env_kwargs.pop("random_window", TS.RANDOM_WINDOW_TRAINING))
    window = env_kwargs.pop("window", TS.WINDOW_LENGTH_BARS)
    warmup = env_kwargs.pop("warmup", 200)
    cfg = env_kwargs.pop("cfg", None)
    # BUILD ONCE: do the slow per-symbol precompute a single time in the parent. This both warms the env
    # AND (if feature_cache_dir) SAVES the features to disk -- so multi-core workers can LOAD them rather
    # than each rebuilding (or each receiving a gigabyte pickle, the original OOM hang).
    print(f"      building shared features for {len(symbol_data)} symbols ONCE...", flush=True)
    subs = build_portfolio_subs(symbol_data, registry_factory, cfg=cfg, warmup=warmup, progress=True,
                                feature_cache_dir=feature_cache_dir)

    # TRUE MULTI-CORE: separate worker PROCESSES each step the market in parallel. To avoid pickling the
    # gigabyte dataset to each worker, the workers LOAD their data (indicator cache) + features (feature
    # cache) FROM DISK -- so only small strings are pickled. Requires both caches on disk + posix (fork,
    # which is Colab-notebook-safe). Falls back to single-process DummyVecEnv otherwise.
    can_subproc = bool(use_subproc and n >= 2 and data_cache_dir and feature_cache_dir and symbols
                       and os.name == "posix")
    if can_subproc:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            from src.training.gym_adapter import make_portfolio_gym_env_from_disk
            del subs   # free the parent copy; each worker loads its own from disk
            syms = list(symbols)

            def _disk_thunk(seed):
                def _f():
                    return make_portfolio_gym_env_from_disk(
                        data_cache_dir, syms, feature_cache_dir=feature_cache_dir, cfg=cfg, warmup=warmup,
                        window=window, random_window=random_window, seed=seed, **env_kwargs)
                return _f

            print(f"      starting {n} worker processes (multi-core; each loads from disk)...", flush=True)
            return SubprocVecEnv([_disk_thunk(i) for i in range(n)], start_method="fork")
        except Exception as e:   # never let multi-core break a run -> fall back to single process
            print(f"      (multi-core unavailable: {e}; using single process)", flush=True)
            subs = build_portfolio_subs(symbol_data, registry_factory, cfg=cfg, warmup=warmup,
                                        progress=False, feature_cache_dir=feature_cache_dir)

    # SINGLE PROCESS (DummyVecEnv): workers SHARE the one read-only `subs` (no pickling, one copy total).
    def _thunk(seed):
        def _f():
            return make_portfolio_gym_env(None, registry_factory, subs=subs, seed=seed, cfg=cfg,
                                          warmup=warmup, random_window=random_window, window=window,
                                          **env_kwargs)
        return _f

    return DummyVecEnv([_thunk(i) for i in range(n)])
