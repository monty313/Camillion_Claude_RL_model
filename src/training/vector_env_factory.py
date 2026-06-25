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
