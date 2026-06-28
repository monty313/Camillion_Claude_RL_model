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
# gamma=0.9995 -> effective horizon ~1/(1-gamma)=2000 steps ~ a FULL trading day (was 0.997 ~1.4h, which
# discounted the midnight +2.5% target and the 4% wall to near-zero, so they were only avoided REACTIVELY).
PPO_HPARAMS = dict(gamma=0.9995, gae_lambda=0.97, n_steps=2048, batch_size=256,
                   ent_coef=0.01, learning_rate=3e-4,
                   policy_kwargs=dict(net_arch=[256, 256, 256]))

VECNORM_KW = dict(norm_obs=True, norm_reward=False, clip_obs=10.0)


def _vecnorm_path(save_path: str) -> str:
    return save_path + "_vecnorm.pkl"


def _make_heartbeat(total_timesteps):
    """A SPARSE 'still working' tick: prints only when crossing each ~10% of training, so a long run is
    never a silent freeze WITHOUT spamming the screen. The day-by-day pass metrics come from the separate
    progress-check callback below. Built lazily so this module imports without SB3."""
    from stable_baselines3.common.callbacks import BaseCallback
    import time as _time

    class _HB(BaseCallback):
        def _on_training_start(self) -> None:
            self._t0 = _time.time(); self._next_pct = 10

        def _on_rollout_end(self) -> None:
            n = self.num_timesteps
            pct = int(100 * n / max(1, total_timesteps))
            if pct >= self._next_pct:
                el = max(1e-9, _time.time() - self._t0)
                eta = (total_timesteps - n) / max(1e-9, n / el) / 60.0
                print(f"      training… {min(pct, 100)}% done  (~{eta:.1f} min left)", flush=True)
                self._next_pct = (pct // 10) * 10 + 10

        def _on_step(self) -> bool:
            return True

    return _HB()


def _make_entropy_anneal(start, total_timesteps):
    """Linearly fade the exploration bonus (ent_coef) from `start` -> ~0 over training, so the bot explores
    early but ends fully DECISIVE/dynamic (nothing forces the finished policy's action mix)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _EA(BaseCallback):
        def _on_rollout_start(self) -> None:
            frac = max(0.0, 1.0 - self.num_timesteps / max(1, total_timesteps))
            try:
                self.model.ent_coef = float(start) * frac
            except Exception:
                pass

        def _on_step(self) -> bool:
            return True

    return _EA()


def _make_day_report_callback(report_env, report_envs, total_timesteps, *, evals=3, max_days=6):
    """Every ~(1/evals) of training, run the day-by-day FTMO report with the CURRENT policy so you watch it
    improve as it learns. Prints (1) a detailed table on the fixed stretch (now WALKS THROUGH a breach instead
    of stopping at day 2), (2) the chosen-action mix (HOLD-collapse check), and (3) a ROLLING pass-rate averaged
    over several RANDOM windows (fixed seeds) -- the single fixed stretch replays identically and can't reveal
    progress on its own. `report_envs` = list of (env, seed). Best-effort: wrapped so it can never break training."""
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as _np

    class _DR(BaseCallback):
        def _on_training_start(self) -> None:
            self._every = max(1, total_timesteps // max(1, evals))
            self._next = self._every

        def _print(self) -> None:
            from src.training.daily_report import daily_report, format_action_mix
            init = float(report_env.cfg.starting_balance)
            pol = sb3_policy_fn(self.model, self.model.get_vec_normalize_env())
            rows, summary = daily_report(report_env, policy=pol, max_days=max_days)
            print(f"\n      ── progress check on a fixed test stretch (after {self.num_timesteps:,} steps) ──",
                  flush=True)
            for r in rows:
                bal = init * (1.0 + r["cum_pnl_pct"] / 100.0)
                tgt = "YES" if r["passed_target"] else "no"
                wall = "ok" if r["within_trailing"] else "BREACH"
                print(f"        Day {r['day']:>2}  {r['date']}  bal ${bal:>12,.0f}  {r['day_pnl_pct']:+6.2f}%  "
                      f"+2.5%? {tgt:<3}  DD {r['trailing_dd_pct']:>4.1f}% {wall:<6}  breach "
                      f"{'YES' if r['breached'] else 'no'}", flush=True)
            print(f"        -> {summary['days_passed_target']}/{summary['days']} days hit +2.5% · "
                  f"{summary['breaches']} breaches · final {summary['final_cum_pct']:+.2f}%", flush=True)
            mix = format_action_mix(summary.get("action_mix"))
            if mix:
                print(f"        action-mix: {mix}", flush=True)
            # ROLLING multi-window pass-rate: re-seed each env to its FIXED seed so the SAME windows are scored
            # every checkpoint (comparable) -> this number MOVES as the policy learns.
            if report_envs:
                tot_pass = tot_days = tot_breach = 0
                finals = []
                for renv, seed in report_envs:
                    renv.rng = _np.random.default_rng(seed)
                    _, s = daily_report(renv, policy=pol, max_days=max_days)
                    tot_pass += s["days_passed_target"]; tot_days += s["days"]
                    tot_breach += s["breaches"]; finals.append(s["final_cum_pct"])
                mean_final = sum(finals) / max(1, len(finals))
                print(f"        ── across {len(report_envs)} random windows: {tot_pass}/{tot_days} days hit "
                      f"+2.5% · {tot_breach} breaches · mean final {mean_final:+.2f}% ──", flush=True)

        def _on_rollout_end(self) -> None:
            if self.num_timesteps < self._next:
                return
            self._next += self._every
            try:
                self._print()
            except Exception as e:
                print(f"      (progress check skipped: {e})", flush=True)

        def _on_step(self) -> bool:
            return True

    return _DR()


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


def _make_checkpoint_callback(save_path, total_timesteps, *, every=None):
    """PERIODICALLY save (OVERWRITE) the model + its VecNormalize stats to save_path, so stopping the run or a
    Colab disconnect never loses progress -- the next run resumes from here. Saves every `every` steps (default
    ~5% of the run, min 50k). Always the SAME path -> one rolling checkpoint, not a pile of files."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _CK(BaseCallback):
        def _on_training_start(self) -> None:
            self._every = int(every or max(50_000, total_timesteps // 20))
            self._next = self._every

        def _save(self) -> None:
            self.model.save(save_path)                              # overwrites save_path(.zip)
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(_vecnorm_path(save_path))                  # overwrites the stats too

        def _on_rollout_end(self) -> None:
            if self.num_timesteps >= self._next:
                self._next += self._every
                try:
                    self._save()
                    print(f"      [checkpoint] progress saved -> {save_path} (after {self.num_timesteps:,} steps)",
                          flush=True)
                except Exception as e:
                    print(f"      [checkpoint] save skipped: {e}", flush=True)

        def _on_step(self) -> bool:
            return True

    return _CK()


def train_portfolio(symbol_data, registry_factory, *, total_timesteps=2_000_000,
                    n_envs=None, save_path="models/camillion_portfolio_ppo", eval_env=None,
                    eval_freq: int | None = None, feature_cache_dir: str | None = None,
                    data_cache_dir: str | None = None, symbols=None, resume: bool = True,
                    checkpoint_every: int | None = None, **env_kwargs):
    """Train ONE policy that trades the WHOLE book from ONE shared pot -- the portfolio bot.

    `symbol_data = {symbol: (indicators, close, time_ns)}` (time-aligned across symbols). Every worker
    is a full PortfolioEnv: the policy decides one symbol at a time while seeing the shared pot's
    exposure, so it learns to BALANCE risk and generalises to the full FTMO universe live. Obs (479),
    actions, VecNormalize and the MlpPolicy are identical to single-symbol training.

    resume=True (default): if a matching past model already exists at save_path, CONTINUE its training
    (warm-start); a non-matching/absent model falls back to a fresh bot. Pass resume=False to force scratch."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    from stable_baselines3.common.vec_env import VecNormalize
    from src.training.vector_env_factory import make_portfolio_vec_env
    from src.training.autotune import autotune

    tuned = autotune()                                  # detect cores/RAM/GPU -> memory-safe settings + report
    if n_envs is None:
        n_envs = tuned["n_envs"]
    print("      building the training environment (can take a minute on a big history)...", flush=True)
    raw = make_portfolio_vec_env(symbol_data, registry_factory, n_envs,
                                 feature_cache_dir=feature_cache_dir, data_cache_dir=data_cache_dir,
                                 symbols=symbols, use_subproc=tuned["use_subproc"], **env_kwargs)
    # RESUME (warm-start): if a past model + its saved stats already exist at save_path AND their obs/action
    # spaces MATCH this run, LOAD and CONTINUE training. SB3's PPO.load raises on any mismatch (e.g. a changed
    # 479-obs contract), so a non-matching/old/absent model cleanly falls back to a FRESH bot. resume=False forces fresh.
    import os
    model = None
    _zip = save_path if save_path.endswith(".zip") else save_path + ".zip"
    _vp = _vecnorm_path(save_path)
    if resume and os.path.exists(_zip) and os.path.exists(_vp):
        try:
            venv = VecNormalize.load(_vp, raw)                  # restore the saved mean/std ...
            venv.training = True; venv.norm_reward = False      # ... and KEEP updating them while we train more
            model = PPO.load(_zip, env=venv, device=tuned["device"])   # explicit .zip (ignore any sibling file)
            print(f"      [resume] MATCH -> loaded past model '{_zip}' (+ stats); CONTINUING its training", flush=True)
        except Exception as e:
            print(f"      [resume] a past model exists but does NOT match this setup ({e}); training a FRESH bot", flush=True)
            model = None
    elif resume:
        print("      [resume] no past model at save_path; training a FRESH bot", flush=True)
    if model is None:                                            # fresh: no match, nothing saved, or resume=False
        venv = VecNormalize(raw, **VECNORM_KW)
        model = PPO("MlpPolicy", venv, verbose=0, device=tuned["device"], **PPO_HPARAMS)
    print("      environment ready; training now...", flush=True)
    cbs = [_make_heartbeat(total_timesteps),
           _make_entropy_anneal(PPO_HPARAMS["ent_coef"], total_timesteps),
           _make_checkpoint_callback(save_path, total_timesteps, every=checkpoint_every)]  # periodic auto-save
    # LIVE day-by-day progress check on a fixed test stretch (loads features from the cache -> fast).
    # Best-effort: if the cache is off or anything fails, training still runs (just without the live table).
    if feature_cache_dir:
        try:
            from src.env.portfolio_env import PortfolioEnv, build_portfolio_subs
            _cfg = env_kwargs.get("cfg"); _warm = env_kwargs.get("warmup", 200)
            rsubs = build_portfolio_subs(symbol_data, registry_factory, cfg=_cfg, warmup=_warm,
                                         progress=False, feature_cache_dir=feature_cache_dir)
            report_env = PortfolioEnv(subs=rsubs, cfg=_cfg, warmup=_warm)
            # ROLLING multi-window eval: a few RANDOM stretches (fixed seeds, share the same precomputed subs)
            # so the averaged pass-rate MOVES as the policy learns -- the single fixed stretch above replays
            # identically every checkpoint and can't reveal progress on its own.
            _win = 10_000   # ~7 trading days of M1 -> max_days=6 binds; clamped to <=half the span by the env
            report_envs = [(PortfolioEnv(subs=rsubs, cfg=_cfg, warmup=_warm,
                                         window=_win, random_window=True, seed=s), s)
                           for s in (11, 23, 37, 53, 71)]
            cbs.append(_make_day_report_callback(report_env, report_envs, total_timesteps))
        except Exception as e:
            print(f"      (live day-by-day check disabled: {e})", flush=True)
    if eval_env is not None:
        freq = int(eval_freq or PPO_HPARAMS["n_steps"])
        cbs.append(EvalCallback(eval_env, eval_freq=max(1, freq), n_eval_episodes=3,
                                deterministic=True, best_model_save_path=None, log_path=None))
    # Interrupt-safe: a manual STOP (Ctrl-C / Colab "Interrupt execution") still saves what we have so far,
    # then the final save below overwrites with the latest -> stopping never loses progress; just re-run to resume.
    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(cbs))
    except KeyboardInterrupt:
        print("\n      [stopped] interrupt caught — saving progress before exit...", flush=True)
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
