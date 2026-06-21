# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 Colab-runnable) | WHO Claude for Monty
# WHY  Train / resume a PPO policy over the cached env, fast. Mirrors Quantra's
#      locked PPO (gamma=0.997, lambda=0.97, 3x256 MLP). An eval callback runs
#      the READ-ONLY Policy Doctor; it never changes training.
# WHERE src/training/trainer.py
# HOW  Lazy SB3 import (Colab installs torch+SB3). REWARD comes only from the
#      env (equity change) -- the trainer adds no reward shaping over alphas.
# DEPENDS_ON src/training/{vector_env_factory,evaluate}.py
# USED_BY notebooks/Camillion_One_Click_Train.ipynb.
# CHANGE_NOTES(IRAC): I: need fast, reproducible PPO with clean eval separation.
#   R: operator guardrails + Quantra locked PPO. A: SB3 PPO + introspection eval
#   callback (read-only). C: a trainable meta-learner over alphas, FTMO-aligned.
# =====================================================================
"""PPO trainer (Colab-runnable; lazy torch/SB3 import). Reward stays env-defined."""
from __future__ import annotations

PPO_HPARAMS = dict(gamma=0.997, gae_lambda=0.97, n_steps=2048, batch_size=256,
                   ent_coef=0.0, learning_rate=3e-4,
                   policy_kwargs=dict(net_arch=[256, 256, 256]))


def train(indicators, close, time_ns, registry_factory, *, total_timesteps=1_000_000,
          n_envs=None, save_path="models/camillion_ppo", eval_env=None, **env_kwargs):
    """Train PPO. SB3/torch are imported here (install them in Colab)."""
    from stable_baselines3 import PPO
    from src.training.vector_env_factory import make_vec_env

    venv = make_vec_env(indicators, close, time_ns, registry_factory, n_envs, **env_kwargs)
    model = PPO("MlpPolicy", venv, verbose=1, **PPO_HPARAMS)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model


def resume(save_path, indicators, close, time_ns, registry_factory, *,
           total_timesteps=500_000, n_envs=None, **env_kwargs):
    from stable_baselines3 import PPO
    from src.training.vector_env_factory import make_vec_env
    venv = make_vec_env(indicators, close, time_ns, registry_factory, n_envs, **env_kwargs)
    model = PPO.load(save_path, env=venv)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model


def sb3_policy_fn(model):
    """Wrap an SB3 model as policy_fn(obs)->(logits, value) for the introspector."""
    import numpy as np, torch
    def policy_fn(obs):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy().ravel()
            value = model.policy.predict_values(obs_t).cpu().numpy().ravel()[0]
        return logits, float(value)
    return policy_fn
