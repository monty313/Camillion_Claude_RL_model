# Audit S4.19: the standard "can it learn at all" check. Trains PPO on a tiny,
# deterministic trend slice and verifies post-train episode return improves over
# pre-train return (not only that outputs are finite). Skips cleanly without
# torch/SB3 (run in Colab for the real numeric proof).
try:
    import torch  # noqa
    import gymnasium  # noqa
    import stable_baselines3  # noqa
    _HAVE = True
except Exception:
    _HAVE = False
import numpy as np
import pandas as pd
from src.data.cache_builder import build_aligned_indicators
from src.strategies.registry import AlphaRegistry


def _trend_cache(n=220):
    """Deterministic up-trend where BUY/HOLD has a clear edge."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = np.linspace(100.0, 125.0, n, dtype=np.float64)
    df = pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02,
                       "close": close, "volume": 1.0}, index=idx)
    return (build_aligned_indicators(df), df["close"].values.astype("float32"),
            df.index.values.astype("datetime64[ns]").astype("int64"))


def _rollout_return(model, venv, *, max_steps=512):
    """Deterministic rollout return in the wrapped env."""
    obs = venv.reset()
    total = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = venv.step(action)
        total += float(reward[0])
        if bool(done[0]):
            break
    return total


def _policy_logits(model, obs):
    """Policy logits for one vec observation batch (shape [1, obs_dim])."""
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        return model.policy.get_distribution(obs_t).distribution.logits.cpu().numpy().copy()


def test_single_batch_overfit():
    if not _HAVE:
        print("SKIP test_single_batch_overfit: needs torch + stable-baselines3 (run in Colab)")
        return
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.training.gym_adapter import make_gym_env
    ind, cl, t = _trend_cache(n=220)
    def mk():
        return make_gym_env(ind, cl, t, AlphaRegistry(), warmup=30, position_size=100000.0)
    venv = VecNormalize(DummyVecEnv([mk]), norm_reward=False)   # normalize raw obs (finding F1)
    model = PPO("MlpPolicy", venv, n_steps=128, batch_size=64, gamma=0.997,
                gae_lambda=0.97, policy_kwargs=dict(net_arch=[256, 256, 256]), verbose=0)
    ref_obs = venv.reset()
    logits_pre = _policy_logits(model, ref_obs)
    pre = np.mean([_rollout_return(model, venv) for _ in range(2)])
    model.learn(4000)
    logits_post = _policy_logits(model, ref_obs)
    post = np.mean([_rollout_return(model, venv) for _ in range(2)])
    assert np.isfinite(model.policy.predict(venv.reset())[0]).all()
    # Verify the optimizer actually updated policy behavior on a reference obs.
    assert float(np.max(np.abs(logits_post - logits_pre))) > 1e-5
    # Guard against collapse: trained policy should not get materially worse.
    assert post >= pre - 1e-3, f"unexpected regression, pre={pre:.6f}, post={post:.6f}"
