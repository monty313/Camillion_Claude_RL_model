# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 Colab-runnable; F1 normalization)
# WHO  Claude for Monty
# WHY  Train / resume a PPO policy over the cached env, fast. Mirrors Quantra's
#      locked PPO (gamma=0.997, lambda=0.97, 3x256 MLP). An eval callback runs
#      the READ-ONLY Policy Doctor; it never changes training.
# WHERE src/training/trainer.py
# HOW  Lazy SB3 import (Colab installs torch+SB3). REWARD comes only from the
#      env (equity change) -- the trainer adds no reward shaping over alphas.
#      F1: training wraps the vec env in VecNormalize(norm_obs=True,
#      norm_reward=False). The raw 367 cache/contract is UNTOUCHED; only what the
#      policy sees at train time is standardized. The running mean/std are SAVED
#      next to the model and MUST be reloaded (training=False) for eval/walk-
#      forward, or the policy sees mis-scaled inputs (and eval would leak into
#      the stats). Use load_for_eval() to do this correctly.
# DEPENDS_ON src/training/{vector_env_factory,evaluate,gym_adapter}.py
# USED_BY notebooks/Camillion_One_Click_Train.ipynb.
# CHANGE_NOTES(IRAC): I: raw obs spans ~[-414,+382] across mixed scales -> PPO
#   learns poorly (audit finding F1). R: normalize the policy input, not the
#   cache. A: VecNormalize on the training path + save/load of the stats; eval
#   loads frozen stats. C: same 367 contract, far better-conditioned learning.
# =====================================================================
"""PPO trainer (Colab-runnable; lazy torch/SB3 import). Reward stays env-defined.

Normalization (F1): obs are standardized by a VecNormalize wrapper at TRAIN time
only. The stats live in `<save_path>_vecnorm.pkl`. Always pair a saved model with
its saved stats; eval/walk-forward must load them with training=False.
"""
from __future__ import annotations

# ent_coef = the "keep exploring" bonus. At 0.0 the policy can collapse to ALWAYS-HOLD: every trade
# pays a cost (immediate negative reward) while HOLD is exactly 0, so doing nothing is a stable trap.
# A small bonus (~0.005-0.02; 0.01 here) keeps it trying real trades long enough to learn. Watch the
# heartbeat's action-mix early in training -- if it's ~HOLD 100%, raise this; if it never settles, lower it.
PPO_HPARAMS = dict(gamma=0.997, gae_lambda=0.97, n_steps=2048, batch_size=256,
                   ent_coef=0.01, learning_rate=3e-4,
                   policy_kwargs=dict(net_arch=[256, 256, 256]))

VECNORM_KW = dict(norm_obs=True, norm_reward=False, clip_obs=10.0)


def _vecnorm_path(save_path: str) -> str:
    return save_path + "_vecnorm.pkl"


def _make_heartbeat(total_timesteps):
    """A tiny SB3 callback that prints LIVE progress after each rollout, so a long run is NEVER silently
    stuck AND you can WATCH IT LEARN: steps/s + ETA, the ACTION MIX (% HOLD/BUY/SELL/CLOSE -> is it
    actually trading or stuck on HOLD?), and the mean step reward (is it making money?). This is the
    plain-language 'is it working?' read-out. Built lazily so this module imports without SB3."""
    from stable_baselines3.common.callbacks import BaseCallback
    import time as _time

    _NAMES = ["HOLD", "BUY", "SELL", "CLOSE"]

    class _HB(BaseCallback):
        def _on_training_start(self) -> None:
            self._t0 = _time.time()

        def _on_rollout_end(self) -> None:
            el = max(1e-9, _time.time() - self._t0)
            n = self.num_timesteps
            rate = n / el
            eta_min = (total_timesteps - n) / max(1e-9, rate) / 60.0
            extra = ""
            try:    # read the just-collected rollout: what did it DO, and did it make money?
                import numpy as _np
                buf = self.model.rollout_buffer
                acts = _np.asarray(buf.actions).astype(int).ravel()
                if acts.size:                                  # skip cleanly if a rollout is somehow empty
                    counts = _np.bincount(acts, minlength=len(_NAMES))[:len(_NAMES)]
                    tot = max(1, int(counts.sum()))
                    mix = " ".join(f"{_NAMES[k]} {100.0 * counts[k] / tot:.0f}%" for k in range(len(_NAMES)))
                    mean_r = float(_np.asarray(buf.rewards).mean())
                    extra = f"  | trading: {mix}  | mean reward {mean_r:+.2e}"
            except Exception:
                pass
            print(f"      ...{n:,}/{total_timesteps:,} steps  ({rate:.0f} steps/s, ~{eta_min:.1f} min left){extra}",
                  flush=True)

        def _on_step(self) -> bool:
            return True

    return _HB()


def train(indicators, close, time_ns, registry_factory, *, total_timesteps=1_000_000,
        n_envs=None, save_path="models/camillion_ppo", eval_env=None,
        eval_freq: int | None = None, **env_kwargs):
    """Train PPO with obs normalization (F1). SB3/torch imported here (Colab)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from src.training.vector_env_factory import make_vec_env

    venv = make_vec_env(indicators, close, time_ns, registry_factory, n_envs, **env_kwargs)
    venv = VecNormalize(venv, **VECNORM_KW)            # F1: standardize policy input
    model = PPO("MlpPolicy", venv, verbose=1, **PPO_HPARAMS)
    cb = None
    if eval_env is not None:
        # Eval callback is optional and read-only: it never mutates training env/reward.
        freq = int(eval_freq or PPO_HPARAMS["n_steps"])
        cb = EvalCallback(eval_env, eval_freq=max(1, freq),
                          n_eval_episodes=3, deterministic=True,
                          best_model_save_path=None, log_path=None)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(save_path)
    venv.save(_vecnorm_path(save_path))                # F1: persist running mean/std
    return model


def train_multi_symbol(symbol_data, registry_factory, *, total_timesteps=1_000_000,
                       n_envs=None, save_path="models/camillion_ppo", eval_env=None,
                       eval_freq: int | None = None, **env_kwargs):
    """Train ONE PPO policy across MANY symbols at once -- the 'one bot trades everything' path.
    `symbol_data = {symbol: (indicators, close, time_ns)}`. Workers are spread round-robin over
    the symbols (each tagged with its symbol + per-asset calibrated size), so a single policy
    learns to generalise across pairs/indices/metals using the cross-asset observation features.
    Rewards are comparable across symbols because each is sized to ~2.5%/day."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from src.training.vector_env_factory import make_multi_symbol_vec_env

    venv = make_multi_symbol_vec_env(symbol_data, registry_factory, n_envs, **env_kwargs)
    venv = VecNormalize(venv, **VECNORM_KW)
    model = PPO("MlpPolicy", venv, verbose=1, **PPO_HPARAMS)
    cb = None
    if eval_env is not None:
        freq = int(eval_freq or PPO_HPARAMS["n_steps"])
        cb = EvalCallback(eval_env, eval_freq=max(1, freq), n_eval_episodes=3,
                          deterministic=True, best_model_save_path=None, log_path=None)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(save_path)
    venv.save(_vecnorm_path(save_path))
    return model


def train_portfolio(symbol_data, registry_factory, *, total_timesteps=2_000_000,
                    n_envs=None, save_path="models/camillion_portfolio_ppo", eval_env=None,
                    eval_freq: int | None = None, feature_cache_dir: str | None = None,
                    data_cache_dir: str | None = None, symbols=None, **env_kwargs):
    """Train ONE policy that trades the WHOLE book from ONE shared pot -- the portfolio bot.

    `symbol_data = {symbol: (indicators, close, time_ns)}` (time-aligned across symbols). Every worker
    is a full PortfolioEnv: the policy decides one symbol at a time while seeing the shared pot's
    exposure, so it learns to BALANCE risk and generalises to the full FTMO universe live. Obs (479),
    actions, VecNormalize and the MlpPolicy are identical to single-symbol training."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    from stable_baselines3.common.vec_env import VecNormalize
    from src.training.vector_env_factory import make_portfolio_vec_env
    from src.training.autotune import autotune

    tuned = autotune()                                  # detect cores/RAM/GPU -> memory-safe settings + report
    if n_envs is None:
        n_envs = tuned["n_envs"]
    print("      building the training environment (can take a minute on a big history)...", flush=True)
    venv = make_portfolio_vec_env(symbol_data, registry_factory, n_envs,
                                  feature_cache_dir=feature_cache_dir, data_cache_dir=data_cache_dir,
                                  symbols=symbols, use_subproc=tuned["use_subproc"], **env_kwargs)
    venv = VecNormalize(venv, **VECNORM_KW)
    model = PPO("MlpPolicy", venv, verbose=0, device=tuned["device"], **PPO_HPARAMS)
    print("      environment ready; training now (you'll see a heartbeat each update)...", flush=True)
    cbs = [_make_heartbeat(total_timesteps)]
    if eval_env is not None:
        freq = int(eval_freq or PPO_HPARAMS["n_steps"])
        cbs.append(EvalCallback(eval_env, eval_freq=max(1, freq), n_eval_episodes=3,
                                deterministic=True, best_model_save_path=None, log_path=None))
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(cbs))
    model.save(save_path)
    venv.save(_vecnorm_path(save_path))
    return model


def resume(save_path, indicators, close, time_ns, registry_factory, *,
        total_timesteps=500_000, n_envs=None, eval_env=None,
        eval_freq: int | None = None, **env_kwargs):
    """Resume training, restoring the saved normalization stats (keeps stats updating)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from src.training.vector_env_factory import make_vec_env

    venv = make_vec_env(indicators, close, time_ns, registry_factory, n_envs, **env_kwargs)
    venv = VecNormalize.load(_vecnorm_path(save_path), venv)   # restore mean/std
    venv.training = True; venv.norm_reward = False
    model = PPO.load(save_path, env=venv)
    cb = None
    if eval_env is not None:
        freq = int(eval_freq or PPO_HPARAMS["n_steps"])
        cb = EvalCallback(eval_env, eval_freq=max(1, freq),
                          n_eval_episodes=3, deterministic=True,
                          best_model_save_path=None, log_path=None)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(save_path)
    venv.save(_vecnorm_path(save_path))
    return model


def load_for_eval(save_path, indicators, close, time_ns, registry_factory, **env_kwargs):
    """Load model + FROZEN normalization stats for leakage-free eval / walk-forward.

    training=False so (1) the policy sees the same scaling it trained on, and
    (2) eval observations do NOT update the running stats. Returns (model, venv).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.training.gym_adapter import make_gym_env

    venv = DummyVecEnv([lambda: make_gym_env(indicators, close, time_ns,
                                             registry_factory(), **env_kwargs)])
    venv = VecNormalize.load(_vecnorm_path(save_path), venv)
    venv.training = False        # freeze stats (no eval-time leakage)
    venv.norm_reward = False
    model = PPO.load(save_path, env=venv)
    return model, venv


def sb3_policy_fn(model, vecnorm=None):
    """Wrap an SB3 model as policy_fn(obs)->(logits, value) for the introspector.

    If `vecnorm` (a loaded VecNormalize) is given, raw obs are normalized with the
    SAME frozen stats the model trained on before the forward pass.
    """
    import numpy as np, torch

    def policy_fn(obs):
        x = np.asarray(obs, dtype=np.float32)
        if vecnorm is not None:
            x = vecnorm.normalize_obs(x)               # apply frozen train-time stats
        obs_t = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy().ravel()
            value = model.policy.predict_values(obs_t).cpu().numpy().ravel()[0]
        return logits, float(value)
    return policy_fn
