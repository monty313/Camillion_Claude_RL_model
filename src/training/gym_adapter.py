# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Wrap TradingEnv as a gymnasium.Env so SB3 can train on it. Import-safe:
#      gymnasium is imported lazily inside the factory (Colab installs it).
# WHERE src/training/gym_adapter.py
# DEPENDS_ON src/env/trading_env.py, config/constants.py | USED_BY vec factory.
"""gymnasium.Env adapter around TradingEnv (lazy import; Colab-runnable)."""
from __future__ import annotations
import numpy as np
from config import constants as C


def make_gym_env(indicators, close, time_ns, alpha_registry, *, aux=None, **kwargs):
    """Build a gymnasium.Env wrapping TradingEnv (call in Colab where gym exists).
    `aux` (v1.6.0 OHLC obs block + ADX-DI side-channel) is an EXPLICIT param so no caller can
    silently drop it -> a None default = OHLC block zeros + the two ADX-DI alphas inactive."""
    import gymnasium as gym
    from gymnasium import spaces
    from src.env.trading_env import TradingEnv

    class GymTradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._env = TradingEnv(indicators, close, time_ns, alpha_registry, aux=aux, **kwargs)
            self.observation_space = spaces.Box(-np.inf, np.inf, C.OBS_SHAPE, np.float32)
            self.action_space = spaces.Discrete(C.N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(int(action))

    return GymTradingEnv()


def make_portfolio_gym_env(symbol_data, registry_factory, **kwargs):
    """Build a gymnasium.Env wrapping the shared-pot PortfolioEnv (one bot, all symbols, one pot).

    Same obs(499)/action(4) interface as the single-symbol env, so the existing MlpPolicy +
    VecNormalize + trainer apply unchanged."""
    import gymnasium as gym
    from gymnasium import spaces
    from src.env.portfolio_env import PortfolioEnv

    class GymPortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._env = PortfolioEnv(symbol_data, registry_factory, **kwargs)
            self.observation_space = spaces.Box(-np.inf, np.inf, C.OBS_SHAPE, np.float32)
            self.action_space = spaces.Discrete(C.N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(int(action))

    return GymPortfolioEnv()


def _default_registry():
    """Module-level (picklable) registry factory so SubprocVecEnv workers can build it themselves."""
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    r = AlphaRegistry(); register_all(r); return r


def make_portfolio_gym_env_from_disk(data_cache_dir, symbols, *, feature_cache_dir=None, cfg=None,
                                     warmup=200, window=None, random_window=False, seed=None, **env_kwargs):
    """Build a PortfolioEnv gym env by LOADING its data (indicator cache) + features (feature cache) FROM
    DISK -- nothing large is pickled. This is what each SubprocVecEnv worker runs, so true multi-core
    training works WITHOUT the gigabyte-pickling that caused the original OOM hang. The feature cache must
    already be populated by the parent (build_portfolio_subs) so workers LOAD rather than rebuild."""
    import gymnasium as gym
    from gymnasium import spaces
    from src.data.cache_builder import load_cache, load_aux
    from src.env.portfolio_env import PortfolioEnv, align_symbol_data, build_portfolio_subs

    # 4-tuple per symbol so the v1.6.0 aux (OHLC obs block + ADX-DI side-channel) flows into the subs.
    sd = align_symbol_data({s: (*load_cache(data_cache_dir, s), load_aux(data_cache_dir, s)) for s in symbols})
    subs = build_portfolio_subs(sd, _default_registry, cfg=cfg, warmup=warmup, progress=False,
                                feature_cache_dir=feature_cache_dir)   # loads from the cache (fast)

    class GymPortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._env = PortfolioEnv(subs=subs, cfg=cfg, warmup=warmup, window=window,
                                     random_window=random_window, seed=seed, **env_kwargs)
            self.observation_space = spaces.Box(-np.inf, np.inf, C.OBS_SHAPE, np.float32)
            self.action_space = spaces.Discrete(C.N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(int(action))

    return GymPortfolioEnv()
