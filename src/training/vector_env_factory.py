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
