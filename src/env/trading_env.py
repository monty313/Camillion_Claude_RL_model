# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  The RL environment. Observation = the locked 357 contract; actions =
#      {HOLD,BUY,SELL,CLOSE}. step() must ONLY read cached float32 (no TA-Lib /
#      MT5 / pandas) so training stays fast.
# WHERE src/env/trading_env.py | HOW Phase-1 makes this a gymnasium.Env that
#      assembles observations via src/observation/builder.py from cached data.
# DEPENDS_ON config/constants.py, src/observation/builder.py, src/risk/* (Phase 1)
# USED_BY src/training/* (Phase 1), tests (Phase 1).
"""Trading environment (Phase-0 placeholder; imports without gymnasium)."""
from __future__ import annotations
from config import constants as C

class TradingEnv:
    """RL env stub. observation_shape is the locked contract; logic is Phase 1."""
    observation_shape = C.OBS_SHAPE
    n_actions = C.N_ACTIONS

    def __init__(self, *args, **kwargs) -> None:
        self._built = False

    def reset(self, *args, **kwargs):
        raise NotImplementedError("Phase 1: cached-data reset + first observation.")

    def step(self, action):
        # HOT LOOP: cached float32 reads only. No TA-Lib/MT5/pandas here.
        raise NotImplementedError("Phase 1: env step over cached features.")
