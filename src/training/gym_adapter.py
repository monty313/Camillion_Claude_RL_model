# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Wrap TradingEnv as a gymnasium.Env so SB3 can train on it. Import-safe:
#      gymnasium is imported lazily inside the factory (Colab installs it).
# WHERE src/training/gym_adapter.py
# DEPENDS_ON src/env/trading_env.py, config/constants.py | USED_BY vec factory.
"""gymnasium.Env adapter around TradingEnv (lazy import; Colab-runnable)."""
from __future__ import annotations
import numpy as np
from config import constants as C


def make_gym_env(indicators, close, time_ns, alpha_registry, **kwargs):
    """Build a gymnasium.Env wrapping TradingEnv (call in Colab where gym exists)."""
    import gymnasium as gym
    from gymnasium import spaces
    from src.env.trading_env import TradingEnv

    class GymTradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._env = TradingEnv(indicators, close, time_ns, alpha_registry, **kwargs)
            self.observation_space = spaces.Box(-np.inf, np.inf, C.OBS_SHAPE, np.float32)
            self.action_space = spaces.Discrete(C.N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(int(action))

    return GymTradingEnv()
