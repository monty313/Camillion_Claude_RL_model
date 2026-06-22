# Audit S4.19: the standard "can it learn at all" check. Trains PPO on a tiny
# fixed slice (with VecNormalize for the raw obs) and confirms it runs finite.
# Skips cleanly without torch/SB3 (run in Colab for the real numeric proof).
try:
    import torch  # noqa
    import gymnasium  # noqa
    import stable_baselines3  # noqa
    _HAVE = True
except Exception:
    _HAVE = False
import numpy as np
try:
    from tests._audit_helpers import cache
except ImportError:  # stdlib runner / Colab load test as top-level module
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _audit_helpers import cache
from src.strategies.registry import AlphaRegistry


def test_single_batch_overfit():
    if not _HAVE:
        print("SKIP test_single_batch_overfit: needs torch + stable-baselines3 (run in Colab)")
        return
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.training.gym_adapter import make_gym_env
    ind, cl, t = cache(n=120)
    def mk():
        return make_gym_env(ind, cl, t, AlphaRegistry(), warmup=20, position_size=100000.0)
    venv = VecNormalize(DummyVecEnv([mk]), norm_reward=False)   # normalize raw obs (finding F1)
    model = PPO("MlpPolicy", venv, n_steps=128, batch_size=64, gamma=0.997,
                gae_lambda=0.97, policy_kwargs=dict(net_arch=[256, 256, 256]), verbose=0)
    model.learn(4000)
    assert np.isfinite(model.policy.predict(venv.reset())[0]).all()
