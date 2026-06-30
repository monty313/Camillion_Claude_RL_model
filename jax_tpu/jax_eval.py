# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The required "tell me the chance of passing" readout AND the stop condition.
#      Runs many DETERMINISTIC challenge windows on HELD-OUT data; a window passes if
#      it reaches +10% without breaching. pass_rate = passes/M = P(pass). The longest
#      run of consecutive passes is the operator's "40 in a row" gate (config knob).
# WHERE jax_tpu/jax_eval.py
# HOW   vmap distinct start bars across the held-out region; lax.scan window_len steps
#       with argmax (deterministic) actions and a FROZEN obs-normalizer; record whether
#       each window passed before its first termination. All on-device.
# DEPENDS_ON: jax, numpy, jax_tpu.{jax_env, jax_ppo, jax_config}
# USED_BY: jax_tpu/jax_trainer.py (stop gate + ledger), the notebook (P(pass) grid)
# CHANGE_NOTES(IRAC): I: need generalization, not memorization, and a real consistency
#   bar. R: blueprint §7/§8 (held-out pass-rate + consecutive-eval gate) + operator
#   "40 straight passes". A: deterministic held-out windows -> pass_rate + best streak.
#   C: an honest P(pass) and a 40-in-a-row stop that means real consistency.
# =====================================================================
"""Held-out walk-forward pass-rate (= P(pass)) + consecutive-pass streak for the 40-in-a-row gate."""
from __future__ import annotations
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax_tpu import jax_env as JE
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_config as JC

# v1.12.0 Stage 4: eval drives the brackets from the policy's continuous head MEANS (deterministic), routed
# through the same freeze/unlock curriculum as training.
_HEAD_MASK = jnp.asarray(JC.curriculum_head_mask(), jnp.float32)
_FROZEN_CONT = jnp.asarray(JC.FROZEN_CONT, jnp.float32)


def _eval_brackets(cont_mean):
    """Deterministic bracket heads at eval: LIVE heads use the mean, FROZEN heads use their default. (B,3)."""
    cu = _HEAD_MASK * cont_mean + (1.0 - _HEAD_MASK) * _FROZEN_CONT
    return cu[:, 0], cu[:, 1], cu[:, 2]


def _longest_run(passed: np.ndarray) -> int:
    """Longest run of consecutive 1s in an ordered 0/1 array (the 'N in a row' metric)."""
    best = cur = 0
    for p in passed.astype(int).tolist():
        cur = cur + 1 if p else 0
        best = max(best, cur)
    return int(best)


def _consensus_actions(state, static):
    """The 'blindly follow the alpha consensus' baseline action for each env (portfolio only): BUY if the
    firing alphas net long, SELL if net short, else HOLD. The bot 'beats the alphas' when it out-performs THIS."""
    am = static.alpha_matrix[state.j, state.t]          # (M, 64) per-env alpha row for the current symbol/bar
    occ = static.occupancy[state.j]                     # (M, 64)
    fired = (am != 0.0) & (occ > 0.5)
    buys = jnp.sum((am > 0.0) & fired, axis=-1)
    sells = jnp.sum((am < 0.0) & fired, axis=-1)
    return jnp.where(buys > sells, 1, jnp.where(sells > buys, 2, 0)).astype(jnp.int32)


@partial(jax.jit, static_argnums=(3, 6, 7, 8))
def _run_windows(net_params, norm, starts, env_params, static, ends_dtf_trf, window_len, env, consensus=False):
    """vmap M windows (one per start bar), scan window_len deterministic steps.
    Returns (passed[M], breached[M], final_return[M], trades[M]). `env` is the env module; `consensus`=True
    runs the follow-the-alphas baseline instead of the trained policy (portfolio only)."""
    ends, dtf, trf = ends_dtf_trf
    model = PPO.CamillionPolicy()
    init_v = jax.vmap(env.init_state, in_axes=(None, None, 0, 0, 0, 0))
    state = init_v(static, env_params, starts, ends, dtf, trf)
    obs0 = jax.vmap(env.reset_obs, in_axes=(0, None, None))(state, static, env_params)
    M = starts.shape[0]
    active0 = jnp.ones((M,), jnp.float32)
    passed0 = jnp.zeros((M,), jnp.float32)

    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0, None, None))   # +tp01/sl01/lot01 bracket heads (v1.12.0)

    def body(carry, _):
        state, obs, active, passed = carry
        if consensus:
            actions = _consensus_actions(state, static)          # follow-the-alphas baseline
            cm = jnp.zeros((state.equity.shape[0], JC.N_CONT_ACTIONS), jnp.float32)   # consensus -> no brackets
        else:
            nobs = PPO.norm_apply(norm, obs.astype(jnp.float32))
            logits, _, cm, _ = model.apply(net_params, nobs)
            actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)   # the trained policy (deterministic)
        tp01, sl01, lot01 = _eval_brackets(cm)
        state, obs, _r, term, trunc = step_v(state, actions, tp01, sl01, lot01, static, env_params)
        done = jnp.maximum(term, trunc)
        newly_done = active * done
        # a window PASSES only if it reached +10% AND never breached (the FTMO contract). episode_passed
        # is monotonic; gating by (1 - episode_breached) means a breach (even AFTER a +10% touch) does NOT
        # count as a pass -> the 40-in-a-row consistency gate can't be inflated by post-pass blow-ups.
        won = state.episode_passed * (1.0 - state.episode_breached)
        passed = jnp.where(newly_done > 0.5, won, passed)
        active = active * (1.0 - done)
        return (state, obs, active, passed), None

    (state, _obs, _active, passed), _ = jax.lax.scan(
        body, (state, obs0, active0, passed0), None, length=window_len)
    final_return = (state.equity - env_params.starting_balance) / env_params.starting_balance
    return passed, state.episode_breached, final_return, state.episode_trades


@partial(jax.jit, static_argnums=(3, 6, 7))
def _won_day_walk(net_params, norm, starts, env_params, static, ends_dtf_trf, walk_len, env):
    """Continuous held-out walk(s): step the policy over the whole held-out region; a BREACH resets the
    account (start over) and keeps walking; track the longest run of WINNING days (state.daily_pass_streak).
    Returns the max won-day streak per walk. (Portfolio env: state has daily_pass_streak.)"""
    ends, dtf, trf = ends_dtf_trf
    model = PPO.CamillionPolicy()
    init_v = jax.vmap(env.init_state, in_axes=(None, None, 0, 0, 0, 0))
    reset_v = jax.vmap(env.reset_obs, in_axes=(0, None, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0, None, None))   # +tp01/sl01/lot01 bracket heads (v1.12.0)
    state = init_v(static, env_params, starts, ends, dtf, trf)
    obs0 = reset_v(state, static, env_params)
    M = starts.shape[0]
    Nsym = static.position_size.shape[0]

    def body(carry, _):
        state, obs, maxstreak, amix, sexp, dead = carry
        prev_days = state.days_elapsed
        nobs = PPO.norm_apply(norm, obs.astype(jnp.float32))
        logits, _, cm, _ = model.apply(net_params, nobs)
        raw = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        # FAIL -> STOP TRADING FOR THE DAY: while "dead" (breached today) force CLOSE so the bot is FLAT and
        # OUT until the next day, then it restarts fresh (operator: "stops trading for that day, restarts the
        # following day"). This also keeps a partial breach-day from being mis-counted as a winning day.
        actions = jnp.where(dead > 0.5, jnp.int32(3), raw)                 # 3 = CLOSE (flatten + stay out)
        amix = amix + jnp.sum(jax.nn.one_hot(raw, 4), axis=0)             # count the policy's INTENT (not forced closes)
        sexp = sexp + jnp.sum((jnp.abs(state.position) > 0.0).astype(jnp.float32), axis=0)  # per-symbol exposure
        tp01, sl01, lot01 = _eval_brackets(cm)                             # bracket heads (deterministic means)
        state, obs, _r, term, trunc = step_v(state, actions, tp01, sl01, lot01, static, env_params)
        maxstreak = jnp.maximum(maxstreak, state.daily_pass_streak)        # longest won-day run so far
        dead = jnp.maximum(dead, term)                                    # a breach -> done for the day
        crossed = (state.days_elapsed > prev_days).astype(jnp.float32)    # a NEW day started this step
        do_reset = dead * crossed                                         # new day while dead -> START OVER fresh
        fresh = init_v(static, env_params, state.t, ends, dtf, trf)
        fresh_obs = reset_v(fresh, static, env_params)
        rm = (do_reset > 0.5)
        state = jax.tree_util.tree_map(
            lambda f, c: jnp.where(rm.reshape((-1,) + (1,) * (f.ndim - 1)), f, c), fresh, state)
        obs = jnp.where(rm[:, None], fresh_obs, obs)
        dead = dead * (1.0 - do_reset)                                    # cleared on the next-day reset
        return (state, obs, maxstreak, amix, sexp, dead), None

    init_carry = (state, obs0, jnp.zeros((M,), jnp.float32), jnp.zeros((4,), jnp.float32),
                  jnp.zeros((Nsym,), jnp.float32), jnp.zeros((M,), jnp.float32))
    (state, _o, maxstreak, amix, sexp, _d), _ = jax.lax.scan(body, init_carry, None, length=walk_len)
    return maxstreak, amix, sexp


def evaluate_won_day_streak(net_params, norm, static, env_params, *, env, n_walks=JC.WON_DAY_N_WALKS,
                            walk_len=None, holdout_start=None, holdout_end=None,
                            daily_target_frac=JC.EVAL_DAILY_TARGET, trailing_dd_frac=JC.EVAL_TRAILING_DD):
    """Best run of consecutive WINNING days (each >= +2.5%) on a continuous held-out walk, breach = start over.
    This is the STOP metric: stop training when this reaches TARGET_WON_DAY_STREAK (40). Portfolio env only
    (needs state.daily_pass_streak). Returns {best_won_day_streak, n_walks, walk_len_steps}."""
    T = int(env_params.T)
    if holdout_start is None:
        holdout_start = int(T * 0.8)
    if holdout_end is None:
        holdout_end = T - 1
    N = int(getattr(env_params, "N", 1))                       # symbol-steps per bar (portfolio cycles symbols)
    # one continuous walk covers the held-out region (bars * N steps); won days need >=40 days of held-out data
    if walk_len is None:
        walk_len = max(2, (holdout_end - holdout_start) * N)
    span = max(1, holdout_end - holdout_start)
    starts = (holdout_start + (np.arange(n_walks) * span // max(1, n_walks))).astype(np.int32)
    starts = np.clip(starts, holdout_start, holdout_end - 1).astype(np.int32)
    ends = jnp.full((n_walks,), holdout_end, jnp.int32)
    dtf = jnp.full((n_walks,), daily_target_frac, jnp.float32)
    trf = jnp.full((n_walks,), trailing_dd_frac, jnp.float32)
    maxstreak, amix, sexp = _won_day_walk(net_params, norm, jnp.asarray(starts), env_params, static,
                                          (ends, dtf, trf), int(walk_len), env)
    amix = np.asarray(amix); sexp = np.asarray(sexp)
    a_tot = max(1.0, float(amix.sum())); s_tot = max(1e-9, float(sexp.sum()))
    return {
        "best_won_day_streak": int(np.asarray(maxstreak).max()),
        "n_walks": int(n_walks), "walk_len_steps": int(walk_len),
        # action mix (HOLD/BUY/SELL/CLOSE fractions) -> confirms/kills the HOLD-collapse question
        "action_mix": (amix / a_tot).tolist(),
        # per-symbol exposure SHARE -> confirms it trades ALL symbols, not just one
        "symbol_exposure": (sexp / s_tot).tolist(),
        "symbol_concentration": float(sexp.max() / s_tot),   # 1/N = even, ~1.0 = all on one symbol
    }


def evaluate(net_params, norm, static, env_params, *, env=JE, n_windows=JC.EVAL_N_WINDOWS,
             window_len=None, holdout_start=None, holdout_end=None,
             daily_target_frac=JC.EVAL_DAILY_TARGET, trailing_dd_frac=JC.EVAL_TRAILING_DD):
    """Deterministic held-out evaluation at a fixed (target, risk).

    Returns dict: pass_rate, best_streak (longest consecutive passes), n_windows, mean_return,
    and the raw per-window passed array. `static`/`env_params` are the device statics; the held-out
    region is [holdout_start, holdout_end] (defaults: last 20% of bars). `env` is the env module
    (jax_env for single-symbol, jax_portfolio_env for the shared pot)."""
    T = int(env_params.T)
    # A held-out CHALLENGE ends at +10% (a real challenge stops when you pass). If this env has a
    # continue_after_pass knob (portfolio), force it OFF for eval so a pass terminates the window.
    if hasattr(env_params, "continue_after_pass"):
        env_params = env_params._replace(continue_after_pass=0.0)
    if holdout_start is None:
        holdout_start = int(T * 0.8)
    if holdout_end is None:
        holdout_end = T - 1
    if window_len is None:
        window_len = min(JC.MAX_BARS, max(2, holdout_end - holdout_start - 2))
    window_len = int(min(window_len, max(2, holdout_end - holdout_start)))   # never exceed the held-out region
    span = holdout_end - window_len - holdout_start
    if span <= 1:
        starts = np.full((n_windows,), holdout_start, np.int32)
    else:
        starts = (holdout_start + (np.arange(n_windows) * span // max(1, n_windows - 1))).astype(np.int32)
    starts = np.clip(starts, holdout_start, max(holdout_start, holdout_end - window_len)).astype(np.int32)
    starts_j = jnp.asarray(starts)
    ends = jnp.minimum(starts_j + window_len, T - 1).astype(jnp.int32)
    dtf = jnp.full((n_windows,), daily_target_frac, jnp.float32)
    trf = jnp.full((n_windows,), trailing_dd_frac, jnp.float32)
    passed, breached, fret, trades = _run_windows(
        net_params, norm, starts_j, env_params, static, (ends, dtf, trf), int(window_len), env, False)
    passed = np.asarray(passed)
    out = {
        "pass_rate": float(passed.mean()),
        "best_streak": _longest_run(passed),
        "n_windows": int(n_windows),
        "mean_return": float(np.asarray(fret).mean()),
        "breach_rate": float(np.asarray(breached).mean()),
        "mean_trades": float(np.asarray(trades).mean()),   # activity (closed trades/window) -> detects "hiding"
        "passed": passed.astype(int).tolist(),
        "daily_target": float(daily_target_frac),
        "trailing_dd": float(trailing_dd_frac),
    }
    # BEAT-THE-ALPHAS: on the SAME held-out windows, compare the bot to a "blindly follow the alpha
    # consensus" baseline. The bot BEATS the alphas when it out-returns / out-passes that baseline.
    if hasattr(static, "alpha_matrix"):
        b_pass, _b_breach, b_ret, _bt = _run_windows(
            net_params, norm, starts_j, env_params, static, (ends, dtf, trf), int(window_len), env, True)
        bot_ret = out["mean_return"]; alpha_ret = float(np.asarray(b_ret).mean())
        out["alpha_pass_rate"] = float(np.asarray(b_pass).mean())
        out["alpha_mean_return"] = alpha_ret
        out["beats_alphas"] = bool(bot_ret > alpha_ret)            # bot out-returns following the alphas
        out["beat_margin"] = bot_ret - alpha_ret                   # how much it beats them by (return frac)
    return out
