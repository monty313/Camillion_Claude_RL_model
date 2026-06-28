# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The on-device PPO training loop that plays THOUSANDS of trading lifetimes at
#      once on a TPU and keeps going until the policy passes 40 challenges in a row on
#      HELD-OUT data — checkpointing params + details + a progress ledger to Drive every
#      eval so a Colab disconnect never loses progress (operator 2026-06-28).
# WHERE jax_tpu/jax_trainer.py
# HOW   One pmapped train iteration = rollout (lax.scan, auto-reset on done, risk
#       domain-randomized) -> GAE -> N_EPOCHS of minibatch PPO with pmean'd grads. Scale
#       (N_ENVS_PER_CORE x N_DEVICES x N_STEPS) is sized in jax_config to push a v2-8 to
#       70-80% utilization. Every EVAL_EVERY iters: held-out pass-rate + best streak ->
#       checkpoint + ledger; stop at TARGET_CONSECUTIVE_PASSES.
# DEPENDS_ON: jax, optax, numpy, jax_tpu.{jax_env, jax_ppo, jax_eval, jax_checkpoint, jax_config}
# USED_BY: notebooks/Camillion_JAX_TPU_Train.ipynb, run-from-python
# CHANGE_NOTES(IRAC): I: CPU PPO can't reach the scale/consistency we need. R: blueprint
#   §8 loop + operator "70-80% TPU, train to 40-in-a-row, save to Colab". A: pmapped
#   rollout+update at scale, domain-randomized risk, held-out 40-in-a-row stop, Drive
#   checkpoints + ledger + resume. C: a saturated TPU marching, documented, to consistency.
# =====================================================================
"""On-device PPO trainer: pmap rollout+update, domain-randomized risk, 40-in-a-row stop, Drive saves."""
from __future__ import annotations
from functools import partial
import signal
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from jax_tpu import jax_env as JE
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_eval as EVAL
from jax_tpu import jax_checkpoint as CKPT
from jax_tpu import jax_config as JC
from jax_tpu import jax_progress as PROG
from src.training.env_fingerprint import env_fingerprint


def _sample_starts_risk(key, n, warmup, train_end, window):
    """Random training-window starts + domain-randomized (daily_target, trailing_dd) per env."""
    k1, k2, k3 = jax.random.split(key, 3)
    hi = jnp.maximum(warmup + 1, train_end - window)
    starts = jax.random.randint(k1, (n,), warmup, jnp.maximum(warmup + 1, hi)).astype(jnp.int32)
    if JC.DOMAIN_RANDOMIZE_RISK:
        dtf = jax.random.uniform(k2, (n,), minval=JC.DAILY_TARGET_MIN, maxval=JC.DAILY_TARGET_MAX)
        trf = jax.random.uniform(k3, (n,), minval=JC.TRAILING_DD_MIN, maxval=JC.TRAILING_DD_MAX)
    else:
        dtf = jnp.full((n,), JC.EVAL_DAILY_TARGET); trf = jnp.full((n,), JC.EVAL_TRAILING_DD)
    ends = jnp.minimum(starts + window, train_end).astype(jnp.int32)
    return starts, ends, dtf, trf


def _fresh_states(key, n, static, params, warmup, train_end, window, env=JE):
    starts, ends, dtf, trf = _sample_starts_risk(key, n, warmup, train_end, window)
    st = jax.vmap(env.init_state, in_axes=(None, None, 0, 0, 0, 0))(static, params, starts, ends, dtf, trf)
    obs = jax.vmap(env.reset_obs, in_axes=(0, None, None))(st, static, params)
    return st, obs


def _restart_continue(s, static, params, env):
    """Fail = START OVER and KEEP GOING: a fresh account at the CURRENT bar (same window, same risk), so the
    bot never trains a dead/breached account -- it restarts and tries again on the same timeline."""
    rs = jax.vmap(env.init_state, in_axes=(None, None, 0, 0, 0, 0))(
        static, params, s.t, s.end, s.daily_target_frac, s.trailing_dd_frac)
    ro = jax.vmap(env.reset_obs, in_axes=(0, None, None))(rs, static, params)
    return rs, ro


def _reset_split(trunc, term_only, fresh_s, fresh_o, restart_s, restart_o, cont_s, cont_o):
    """truncation (window end) -> a NEW random window (diversity); breach/pass -> RESTART a fresh account at
    the current bar (continue the timeline); otherwise keep stepping."""
    def pick(fw, rc, kp):
        tw = (trunc > 0.5).reshape((-1,) + (1,) * (fw.ndim - 1))
        to = (term_only > 0.5).reshape((-1,) + (1,) * (fw.ndim - 1))
        return jnp.where(tw, fw, jnp.where(to, rc, kp))
    state = jax.tree_util.tree_map(pick, fresh_s, restart_s, cont_s)
    obs = jnp.where((trunc > 0.5)[:, None], fresh_o,
                    jnp.where((term_only > 0.5)[:, None], restart_o, cont_o))
    return state, obs


def make_train_iter(static, params, warmup, train_end, window, env=JE):
    """Build the pmapped one-iteration function (rollout + GAE + PPO epochs). `env` is the env
    module (jax_env for single-symbol, jax_portfolio_env for the shared pot)."""
    step_v = jax.vmap(env.step, in_axes=(0, 0, None, None))
    model = PPO.CamillionPolicy()
    optimizer = PPO.make_optimizer()
    n_per_core = JC.N_ENVS_PER_CORE
    n_steps = JC.N_STEPS
    total = n_per_core * n_steps
    mb = min(JC.MINIBATCH_SIZE, total)
    n_mb = max(1, total // mb)

    def rollout(net_params, norm, state, obs, key):
        def body(carry, _):
            state, obs, key = carry
            key, ak = jax.random.split(key)
            nobs = PPO.norm_apply(norm, obs.astype(jnp.float32))
            logits, value = model.apply(net_params, nobs)
            actions = jax.random.categorical(ak, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1)[:, 0]
            nstate, nobs2, reward, term, trunc = step_v(state, actions.astype(jnp.int32), static, params)
            done = jnp.maximum(term, trunc)
            key, rk = jax.random.split(key)
            # FAIL -> START OVER and KEEP GOING: a breach restarts a fresh account at the current bar (continue
            # the timeline -> the bot never trains dead); a window-end draws a NEW random window (diversity).
            term_only = term * (1.0 - trunc)
            fresh_s, fresh_o = _fresh_states(rk, obs.shape[0], static, params, warmup, train_end, window, env)
            restart_s, restart_o = _restart_continue(nstate, static, params, env)
            nstate, nobs2 = _reset_split(trunc, term_only, fresh_s, fresh_o, restart_s, restart_o, nstate, nobs2)
            trans = (obs, actions.astype(jnp.int32), logp, value, reward, done)
            return (nstate, nobs2, key), trans

        (state, obs, key), trans = jax.lax.scan(body, (state, obs, key), None, length=n_steps)
        nobs = PPO.norm_apply(norm, obs.astype(jnp.float32))
        _, last_value = model.apply(net_params, nobs)
        return state, obs, key, trans, last_value

    @partial(jax.pmap, axis_name="i", static_broadcasted_argnums=())
    def train_iter(net_params, opt_state, norm, state, obs, key, ent_coef):
        state, obs, key, trans, last_value = rollout(net_params, norm, state, obs, key)
        obs_t, act_t, logp_t, val_t, rew_t, done_t = trans
        adv, ret = PPO.compute_gae(rew_t, val_t, done_t, last_value)
        # flatten (T, n) -> (T*n, ...)
        flat = lambda x: x.reshape((total,) + x.shape[2:])
        obs_f, act_f, logp_f = flat(obs_t), flat(act_t), flat(logp_t)
        adv_f, ret_f = flat(adv), flat(ret)
        mask_f = jnp.ones((total,), jnp.float32)

        keys = jax.random.split(key, JC.N_EPOCHS + 1)
        key = keys[0]
        perms = jax.vmap(lambda k: jax.random.permutation(k, total))(keys[1:])      # (epochs, total)
        mb_index = perms[:, :n_mb * mb].reshape(JC.N_EPOCHS * n_mb, mb)

        def upd(carry, idx):
            net_params, opt_state = carry
            o = PPO.norm_apply(norm, obs_f[idx].astype(jnp.float32))
            (loss, aux), grads = jax.value_and_grad(PPO.ppo_loss, has_aux=True)(
                net_params, model.apply, o, act_f[idx], logp_f[idx],
                adv_f[idx], ret_f[idx], mask_f[idx], ent_coef)
            grads = jax.lax.pmean(grads, axis_name="i")
            updates, opt_state = optimizer.update(grads, opt_state, net_params)
            net_params = optax.apply_updates(net_params, updates)
            return (net_params, opt_state), aux["entropy"]

        (net_params, opt_state), ents = jax.lax.scan(upd, (net_params, opt_state), mb_index)
        norm = PPO.norm_update(norm, obs_f, axis_name="i")   # sync the obs-normalizer across devices
        metrics = {"mean_reward": jax.lax.pmean(rew_t.mean(), "i"),
                   "entropy": jax.lax.pmean(ents.mean(), "i"),
                   "passes": jax.lax.psum(state.episode_passed.sum(), "i")}
        return net_params, opt_state, norm, state, obs, key, metrics

    return train_iter, optimizer


def _replicate(tree, n):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.asarray(x).shape), tree)


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def train(static_data, *, save_dir=JC.SAVE_DIR, resume=True, total_updates=None,
          eval_every=JC.EVAL_EVERY, target_streak=JC.TARGET_WON_DAY_STREAK,
          n_envs_per_core=None, n_steps=None, verbose=True, max_iters=None, env_param_kwargs=None,
          on_eval=None, log_every=10, env=JE, anneal_updates=None):
    """Train until `target_streak` consecutive held-out challenge passes; save everything to Drive.

    `static_data` is a jax_static_features.StaticData (one symbol). `env_param_kwargs` overrides
    the FTMO knobs in params_from_static. `on_eval(row, rows)` is called every eval with the latest
    progress row + the full ledger (pass a jax_progress.LiveDashboard() to watch P(pass) + the
    40-in-a-row streak live). `log_every` controls the light heartbeat between evals. Returns details."""
    if n_envs_per_core is not None:
        JC.N_ENVS_PER_CORE = int(n_envs_per_core)
    if n_steps is not None:
        JC.N_STEPS = int(n_steps)
    static = env.make_device_static(static_data)
    params = env.params_from_static(static_data, **(env_param_kwargs or {}))
    n_dev = jax.local_device_count()
    n_per_core = JC.N_ENVS_PER_CORE
    T = static_data.T
    warmup = static_data.warmup
    symbols = list(getattr(static_data, "symbols", []))   # for the per-symbol exposure diagnostic
    train_end = int(T * 0.8)                              # last 20% held out for eval
    window = int(min(JC.MAX_BARS, max(2, (train_end - warmup) // 2)))
    fp = env_fingerprint()

    key = jax.random.PRNGKey(JC.SEED)
    model, net_params = PPO.init_params(key)
    norm = PPO.norm_init()
    train_iter, optimizer = make_train_iter(static, params, warmup, train_end, window, env)
    opt_state = optimizer.init(net_params)

    start_update = 0
    best_streak = 0
    if resume:
        latest = CKPT.find_latest(save_dir)
        if latest is not None:
            tag, start_update = latest
            net_params, norm, details = CKPT.load_policy(save_dir, tag, net_params, PPO.RunningNorm)
            best_streak = int(details.get("best_streak_global", 0))
            if verbose:
                print(f"[resume] loaded {tag} @ update {start_update} (best_streak={best_streak})")

    # replicate across devices; shard envs as (n_dev, n_per_core)
    p_params = _replicate(net_params, n_dev)
    p_opt = _replicate(opt_state, n_dev)
    p_norm = _replicate(norm, n_dev)
    key, sk = jax.random.split(key)
    dev_keys = jax.random.split(sk, n_dev)
    init_keys = jax.random.split(key, n_dev)

    def _init_shard(k):
        return _fresh_states(k, n_per_core, static, params, warmup, train_end, window, env)
    p_state, p_obs = jax.vmap(_init_shard)(init_keys)

    if verbose:
        print(f"[train] devices={n_dev} envs/core={n_per_core} n_steps={JC.N_STEPS} "
              f"-> batch/iter={n_dev * n_per_core * JC.N_STEPS:,} | fp={fp} | window={window}")

    details = {"contract_version": __import__("config.constants", fromlist=["x"]).OBSERVATION_CONTRACT_VERSION,
               "obs_size": JC.OBS_SIZE, "env_fingerprint": fp, "n_devices": n_dev,
               "n_envs_per_core": n_per_core, "n_steps": JC.N_STEPS, "best_streak_global": best_streak}
    it = 0
    t0 = time.time()
    update = start_update
    rows = []           # the full progress ledger (for the live dashboard)
    prev_row = None

    # ---- CRASH-SAFE CHECKPOINTING (operator 2026-06-28) ---------------------------------------------
    # Progress is saved (a) at EVERY eval checkpoint, (b) every CHECKPOINT_EVERY updates (lightweight, so a
    # hard kill loses at most that many updates), and (c) on INTERRUPT (Ctrl-C, or a Colab-disconnect SIGTERM)
    # -> a graceful save before exit. Re-run with resume=True and it picks up from the latest checkpoint.
    def _save(upd, *, named=None, why=""):
        npar = _unreplicate(p_params); nnorm = _unreplicate(p_norm)
        d = {**details, "update": int(upd), "saved_reason": why, "best_streak_global": best_streak}
        CKPT.checkpoint(save_dir, int(upd), npar, nnorm, d)     # jax_ckpt_<upd> + latest_step.txt (resume anchor)
        if named:
            CKPT.save_named(save_dir, named, npar, nnorm, d)
        return d

    interrupted = {"flag": False, "why": ""}
    def _on_signal(signum, _frame):
        interrupted["flag"] = True; interrupted["why"] = f"signal-{signum}"
    _old_term = None
    try:                                                       # SIGTERM handler only works in the main thread
        _old_term = signal.signal(signal.SIGTERM, _on_signal)
    except Exception:
        _old_term = None

    # UNBOUNDED by default: keep training until `target_streak` in a row is hit (the ONLY stop). A cap is
    # opt-in (total_updates) + a max_iters backstop for tests. Entropy anneals over `anneal_updates`,
    # decoupled from the (open-ended) stop so the schedule is well-defined when training is unbounded.
    anneal = anneal_updates or JC.ANNEAL_UPDATES
    if verbose:
        cap = "until 40-in-a-row (unbounded)" if total_updates is None else f"<= {total_updates:,} updates"
        print(f"[stop] training {cap}; stop ONLY when {target_streak} won days in a row on held-out data "
              f"| checkpoints -> {save_dir} (every eval + every {JC.CHECKPOINT_EVERY} updates + on interrupt)")
    try:
      while total_updates is None or update < total_updates:
        if interrupted["flag"]:                                # Colab disconnect / SIGTERM -> save & stop
            _save(update, named="interrupted", why=interrupted["why"])
            if verbose:
                print(f"[interrupted] {interrupted['why']} -> saved progress at update {update} -> {save_dir}. "
                      f"Re-run the cell (resume=True) to continue.")
            return details
        progress = update / max(1, anneal)
        ent = float(PPO.entropy_coef(progress))
        p_ent = jnp.full((n_dev,), ent)
        p_params, p_opt, p_norm, p_state, p_obs, dev_keys, metrics = train_iter(
            p_params, p_opt, p_norm, p_state, p_obs, dev_keys, p_ent)
        update += 1
        it += 1

        is_eval = (update % eval_every == 0 or update == start_update + 1)
        if (update % JC.CHECKPOINT_EVERY == 0) and not is_eval:   # lightweight crash-safe save between evals
            _save(update, why="periodic")
        if verbose and log_every and (update % log_every == 0) and not is_eval:
            print(PROG.heartbeat(update, update * n_dev * n_per_core * JC.N_STEPS,
                                 float(np.asarray(metrics["mean_reward"])[0]),
                                 it / max(1e-6, time.time() - t0), best_streak, target_streak), flush=True)

        if is_eval:
            net_params = _unreplicate(p_params)
            norm = _unreplicate(p_norm)
            ev = EVAL.evaluate(net_params, norm, static, params, env=env,
                               daily_target_frac=JC.EVAL_DAILY_TARGET, trailing_dd_frac=JC.EVAL_TRAILING_DD)
            pass_streak = ev["best_streak"]              # challenge-pass streak (HEALTH metric)
            # STOP metric = WON DAYS IN A ROW on a continuous held-out walk (breach = start over). Portfolio only;
            # single-symbol has no daily streak so it falls back to the challenge-pass streak.
            is_portfolio = hasattr(static, "alpha_matrix")
            if is_portfolio:
                wd = EVAL.evaluate_won_day_streak(net_params, norm, static, params, env=env,
                                                  daily_target_frac=JC.EVAL_DAILY_TARGET, trailing_dd_frac=JC.EVAL_TRAILING_DD)
                won_streak = wd["best_won_day_streak"]
                stop_streak = won_streak
            else:
                won_streak = None
                stop_streak = pass_streak
                wd = {}
            best_streak = max(best_streak, stop_streak)
            mr = float(np.asarray(metrics["mean_reward"])[0])
            iters_per_s = it / max(1e-6, time.time() - t0)
            row = {"update": int(update), "timesteps": int(update * n_dev * n_per_core * JC.N_STEPS),
                   "mean_reward": mr, "entropy": float(np.asarray(metrics["entropy"])[0]),
                   "eval_pass_rate": ev["pass_rate"], "consecutive_passes": pass_streak,
                   "won_day_streak": won_streak, "stop_streak": stop_streak,
                   "best_streak_global": best_streak, "eval_mean_return": ev["mean_return"],
                   "eval_breach_rate": ev["breach_rate"], "eval_mean_trades": ev["mean_trades"],
                   "eval_beats_alphas": ev.get("beats_alphas"), "eval_beat_margin": ev.get("beat_margin"),
                   "eval_alpha_return": ev.get("alpha_mean_return"),
                   "action_mix": wd.get("action_mix"), "symbol_exposure": wd.get("symbol_exposure"),
                   "symbol_concentration": wd.get("symbol_concentration"), "symbols": symbols,
                   "iters_per_s": round(iters_per_s, 3), "ent_coef": ent}
            details.update({"update": int(update), "best_streak_global": best_streak,
                            "last_pass_rate": ev["pass_rate"], "last_won_day_streak": won_streak,
                            "last_pass_streak": pass_streak, "mean_reward": mr, "timesteps": row["timesteps"]})
            CKPT.append_progress(save_dir, row, JC.PROGRESS_JSONL)
            CKPT.checkpoint(save_dir, update, net_params, norm, details)
            if stop_streak >= best_streak:
                CKPT.save_named(save_dir, JC.BEST_DIR, net_params, norm, details)
            rows.append(row)
            if verbose:
                print(PROG.format_eval(row, prev_row, target_streak), flush=True)
            prev_row = row
            if on_eval is not None:
                try:
                    on_eval(row, rows)
                except Exception as e:   # a dashboard hiccup must never stop training
                    print(f"   (on_eval callback skipped: {e})")
            if stop_streak >= target_streak:
                metric = "winning days in a row" if is_portfolio else "challenge passes in a row"
                CKPT.save_named(save_dir, JC.PASSED_DIR, net_params, norm,
                                {**details, "passed_streak": stop_streak, "passed_at_update": int(update)})
                if verbose:
                    print(f"[DONE] reached {stop_streak} {metric} on held-out data at update {update}. "
                          f"Saved -> {save_dir}/{JC.PASSED_DIR}")
                return details
        if max_iters is not None and it >= max_iters:
            break
    except KeyboardInterrupt:                                  # Ctrl-C -> save before exit
        _save(update, named="interrupted", why="KeyboardInterrupt")
        if verbose:
            print(f"\n[interrupted] Ctrl-C -> saved progress at update {update} -> {save_dir}. "
                  f"Re-run the cell (resume=True) to continue from here.")
        return details
    finally:
        if _old_term is not None:                              # restore the previous SIGTERM handler
            try:
                signal.signal(signal.SIGTERM, _old_term)
            except Exception:
                pass
    return details


def train_portfolio(portfolio_static_data, **kwargs):
    """Train the SHARED-POT portfolio bot (the core goal) to `target_streak` consecutive held-out
    passes. Same API as train(), but uses the jax_portfolio_env. `portfolio_static_data` is a
    jax_static_features.PortfolioStaticData (build it with build_portfolio_static(subs))."""
    from jax_tpu import jax_portfolio_env as JPE
    return train(portfolio_static_data, env=JPE, **kwargs)
