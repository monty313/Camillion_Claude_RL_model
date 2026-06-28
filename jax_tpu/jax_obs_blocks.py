# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The 40 DYNAMIC observation floats (account_daily 7 + account_episode 7 +
#      portfolio 8 + sizing 10 + recent_context 8) rebuilt in jnp. These are the
#      ONLY obs values the JAX hot loop recomputes each step (they depend on the
#      evolving account/position state); the other 459 floats are indexed from the
#      shared precomputed static tensor (see jax_static_features.py). This is a 1:1
#      branchless port of src/account/win_loss_features.py.
# WHERE jax_tpu/jax_obs_blocks.py
# HOW   Pure jnp functions, broadcast over the env batch (vmap-safe). Risk limits
#       arrive as FRACTIONS (per-env, so domain randomization feeds these % features
#       exactly as the CPU env feeds cfg). Fixed denominators come from config.
# DEPENDS_ON: jax, config.constants (SIZING_LOTS_LADDER), config.variables (MAX_* denoms)
# USED_BY: jax_tpu/jax_env.py, jax_tpu/tests/test_jax_parity.py
# CHANGE_NOTES(IRAC): I: the account/sizing/context features evolve every step and
#   must match the CPU obs exactly or the policy sees a different world. R: CLAUDE.md
#   "never change the observation" + win_loss_features as the reference. A: port each
#   of the 5 functions to clipped jnp math, same bases/denominators. C: byte-parity on
#   the 40 dynamic floats -> the full 499 obs matches the CPU env.
# =====================================================================
"""The 40 dynamic obs floats in jnp — 1:1 with src/account/win_loss_features.py."""
from __future__ import annotations
import jax.numpy as jnp
from config import constants as C
from config import variables as V

# Fixed denominators / ladder (host constants -> baked into the jitted graph).
_MAX_DAILY_TRADES = float(max(1, V.MAX_DAILY_TRADES))           # 30
_MAX_CONSEC_LOSSES = float(max(1, V.MAX_CONSECUTIVE_LOSSES))    # 10
_MAX_OPEN_POSITIONS = float(max(1, V.MAX_OPEN_POSITIONS))       # 5
_LADDER = tuple(float(x) for x in C.SIZING_LOTS_LADDER)         # (0.01,0.1,0.5,1.0,2.0,4.0)
_LADDER_MAX = float(max(_LADDER))                              # 4.0
_F32 = jnp.float32


def _safe_div(num, den):
    """num/den with den==0 -> 0.0 (matches the CPU `(x/d) if d else 0.0` idiom)."""
    den = jnp.asarray(den, dtype=_F32)
    return jnp.where(den != 0.0, jnp.asarray(num, dtype=_F32) / jnp.where(den != 0.0, den, 1.0), 0.0)


def daily_features(equity, day_start_balance, starting_balance, daily_realized_pnl,
                   daily_wins, daily_trades, daily_consecutive_losses,
                   daily_target_frac, daily_dd_frac):
    """account_daily block (7). CPU ref: win_loss_features.daily_features."""
    bal0 = day_start_balance          # always set (>0) in our env
    init = starting_balance
    win_rate = _safe_div(daily_wins, daily_trades)
    pnl_frac = _safe_div(daily_realized_pnl, bal0)
    daily_gain_frac = _safe_div(equity - bal0, init)
    target_progress = _safe_div(daily_gain_frac, daily_target_frac)
    daily_live_loss = jnp.maximum(0.0, bal0 - equity)
    daily_loss_frac = _safe_div(daily_live_loss, bal0)
    dd_used = _safe_div(daily_loss_frac, daily_dd_frac)
    risk_remaining = 1.0 - dd_used
    trades_pct = daily_trades / _MAX_DAILY_TRADES
    streak_pct = daily_consecutive_losses / _MAX_CONSEC_LOSSES
    return jnp.stack([
        jnp.clip(win_rate, 0, 1),
        jnp.clip(pnl_frac, -1, 1),
        jnp.clip(dd_used, 0, 2),
        jnp.clip(target_progress, -2, 2),
        jnp.clip(risk_remaining, 0, 1),
        jnp.clip(trades_pct, 0, 2),
        jnp.clip(streak_pct, 0, 2),
    ], axis=-1).astype(_F32)


def episode_features(equity, episode_start_balance, starting_balance, episode_realized_pnl,
                     episode_wins, episode_trades, episode_consecutive_losses, episode_breached,
                     episode_peak_equity, trailing_dd_frac, total_dd_frac, trailing_enabled,
                     profit_target_frac):
    """account_episode block (7). CPU ref: win_loss_features.episode_features."""
    bal0 = episode_start_balance
    win_rate = _safe_div(episode_wins, episode_trades)
    pnl_frac = _safe_div(episode_realized_pnl, bal0)
    # binding wall: trailing if enabled else total (branchless select)
    wall = trailing_enabled * trailing_dd_frac + (1.0 - trailing_enabled) * total_dd_frac
    peak = episode_peak_equity
    dd_frac = jnp.maximum(0.0, _safe_div(peak - equity, peak))
    dd_used = _safe_div(dd_frac, wall)
    risk_remaining = 1.0 - dd_used
    target_progress = _safe_div(pnl_frac, profit_target_frac)
    # pass_progress: 0 if breached else clip(target_progress,0,1)
    pass_progress = (1.0 - episode_breached) * jnp.clip(target_progress, 0, 1)
    streak_pct = episode_consecutive_losses / _MAX_CONSEC_LOSSES
    return jnp.stack([
        jnp.clip(win_rate, 0, 1),
        jnp.clip(pnl_frac, -1, 1),
        jnp.clip(dd_used, 0, 2),
        jnp.clip(target_progress, -2, 2),
        jnp.clip(pass_progress, 0, 1),
        jnp.clip(risk_remaining, 0, 1),
        jnp.clip(streak_pct, 0, 2),
    ], axis=-1).astype(_F32)


def portfolio_features(equity, balance, starting_balance, position):
    """portfolio block (8). CPU ref: win_loss_features.portfolio_features fed by
    TradingEnv._portfolio_block (single-symbol): open=1 if pos!=0; net=pos; gross=|pos|;
    unreal=equity-balance; largest_dir=sign(pos); avg_position_age_bars stays 0 -> age_pct=0."""
    bal0 = starting_balance
    position = jnp.asarray(position, dtype=_F32)
    equity = jnp.asarray(equity, dtype=_F32)
    balance = jnp.asarray(balance, dtype=_F32)
    open_positions = (position != 0.0).astype(_F32)
    net_exposure = position
    gross_exposure = jnp.abs(position)
    unreal = equity - balance
    largest_dir = jnp.sign(position)
    open_pct = open_positions / _MAX_OPEN_POSITIONS
    equity_ratio = _safe_div(equity, bal0)
    balance_ratio = _safe_div(balance, bal0)
    unreal_frac = _safe_div(unreal, bal0)
    age_pct = 0.0  # avg_position_age_bars never set in single-symbol env
    return jnp.stack([
        jnp.clip(open_pct, 0, 1),
        jnp.clip(net_exposure, -1, 1),
        jnp.clip(gross_exposure, 0, 1),
        jnp.clip(unreal_frac, -1, 1),
        jnp.clip(jnp.broadcast_to(_F32(age_pct), jnp.shape(equity)), 0, 1),
        jnp.clip(largest_dir, -1, 1),
        jnp.clip(equity_ratio, 0, 5),
        jnp.clip(balance_ratio, 0, 5),
    ], axis=-1).astype(_F32)


def portfolio_features_agg(equity, balance, starting_balance, open_positions,
                           net_exposure, gross_exposure, largest_dir):
    """portfolio block (8) for the SHARED-POT env. CPU ref: win_loss_features.portfolio_features fed by
    PortfolioEnv._set_aggregates (aggregates already computed across ALL symbols). avg_position_age=0."""
    bal0 = starting_balance
    open_pct = open_positions / _MAX_OPEN_POSITIONS
    unreal_frac = _safe_div(equity - balance, bal0)
    equity_ratio = _safe_div(equity, bal0)
    balance_ratio = _safe_div(balance, bal0)
    zero = jnp.zeros_like(jnp.asarray(equity, _F32))
    return jnp.stack([
        jnp.clip(open_pct, 0, 1),
        jnp.clip(net_exposure, -1, 1),
        jnp.clip(gross_exposure, 0, 1),
        jnp.clip(unreal_frac, -1, 1),
        jnp.clip(zero, 0, 1),                       # age_pct = 0 (not tracked)
        jnp.clip(largest_dir, -1, 1),
        jnp.clip(equity_ratio, 0, 5),
        jnp.clip(balance_ratio, 0, 5),
    ], axis=-1).astype(_F32)


def sizing_features(equity, day_start_balance, starting_balance, episode_peak_equity,
                    value_per_point, ref_move, position_size,
                    daily_target_frac, trailing_dd_frac, total_dd_frac, trailing_enabled):
    """sizing block (10). CPU ref: win_loss_features.sizing_features."""
    init = starting_balance
    one_lot_move = value_per_point * ref_move
    ladder = [jnp.clip(one_lot_move * L / init, 0.0, 1.0) for L in _LADDER]   # 6
    day0 = day_start_balance
    day_gain_frac = _safe_div(equity - day0, init)
    target_remaining = jnp.clip(daily_target_frac - day_gain_frac, 0.0, 1.0)
    wall = trailing_enabled * trailing_dd_frac + (1.0 - trailing_enabled) * total_dd_frac
    peak = episode_peak_equity
    dd_frac = jnp.maximum(0.0, _safe_div(peak - equity, peak))
    dd_room = jnp.clip(wall - dd_frac, 0.0, 1.0)
    active_lots = _safe_div(position_size, value_per_point)
    active_norm = jnp.clip(active_lots / _LADDER_MAX, 0.0, 1.0)
    active_move = jnp.clip(one_lot_move * active_lots / init, 0.0, 1.0)
    return jnp.stack([*ladder, target_remaining, dd_room, active_norm, active_move],
                     axis=-1).astype(_F32)


def recent_context_features(equity, starting_balance, week_avg, prev_day, prev2, today_sofar,
                            typical_range, days_elapsed, daily_target_frac, profit_target_frac):
    """recent_context block (8). CPU ref: win_loss_features.recent_context_features.
    `typical_range` is the symbol's long-run daily range, or <=0 to mean 'None' (-> use week_avg)."""
    eps = 1e-9
    wk = jnp.maximum(week_avg, eps)
    # CPU: typ = typical_range if typical_range else wk  -> 0/None means use wk
    typ = jnp.where(jnp.asarray(typical_range) > 0.0, typical_range, wk)
    week_vs_typical = jnp.clip(wk / jnp.maximum(typ, eps) / 2.0, 0.0, 1.0)
    prev_vs_week = jnp.clip(prev_day / wk / 2.0, 0.0, 1.0)
    prev2_vs_week = jnp.clip(prev2 / wk / 2.0, 0.0, 1.0)
    today_vs_week = jnp.clip(today_sofar / wk / 2.0, 0.0, 1.0)
    init = starting_balance
    ret = _safe_div(equity - init, init)
    target = profit_target_frac
    daily_t = daily_target_frac
    d = jnp.maximum(1.0, days_elapsed)
    days_norm = jnp.clip(days_elapsed / 20.0, 0.0, 1.0)
    return_so_far = jnp.clip(ret, -1.0, 1.0)
    pace = jnp.where(daily_t > 0.0, jnp.clip(_safe_div(ret, daily_t * d) / 2.0, 0.0, 1.0), 0.0)
    remaining = jnp.clip(target - ret, 0.0, 1.0)
    return jnp.stack([week_vs_typical, prev_vs_week, prev2_vs_week, today_vs_week,
                      days_norm, return_so_far, pace, remaining], axis=-1).astype(_F32)
