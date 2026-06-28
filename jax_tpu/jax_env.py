# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The trading environment as a BRANCHLESS, fixed-shape jnp state machine so
#      thousands of copies step in lockstep on a TPU. It is a 1:1 reimplementation
#      of src/env/trading_env.py (the CPU reference): same action effects, same
#      P&L/cost model, same equity-change reward, same day rollover, same NY-index
#      bonus, same breach + two-phase banking. The 459 STATIC obs floats are indexed
#      from the shared precomputed tensor; only the 40 DYNAMIC floats are recomputed.
# WHERE jax_tpu/jax_env.py
# HOW   EnvState = a NamedTuple pytree (one set of scalars per env -> vmap replicates
#       it). step_env() does every FTMO decision with jnp.where/masks (Rule 3). Money
#       math follows the array dtype (float64 when jax_enable_x64 is on for the parity
#       test; float32 on TPU). Obs assembled by scattering the dynamic blocks into the
#       static row, then nan_to_num (exactly like the CPU builder).
# DEPENDS_ON: jax, jax_tpu/{jax_ftmo, jax_obs_blocks, jax_static_features}, config.constants
# USED_BY: jax_tpu/jax_trainer.py, jax_tpu/jax_eval.py, jax_tpu/tests/test_jax_parity.py
# CHANGE_NOTES(IRAC): I: the CPU env's Python if/else can't run vectorized on a TPU.
#   R: blueprint Rules 2+3 (fixed shapes + branchless) + step-parity invariant. A: a
#   pytree state + a masked step that reproduces trading_env.step line-for-line. C: the
#   same observations/rewards as the CPU env, now for thousands of lives at once.
# =====================================================================
"""Branchless jnp trading env (EnvState pytree + step_env) — 1:1 with src/env/trading_env.py."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from config import constants as C
from jax_tpu import jax_ftmo, jax_obs_blocks
from jax_tpu.jax_static_features import BLOCK_RANGES, DYNAMIC_BLOCKS

_HOLD, _BUY, _SELL, _CLOSE = 0, 1, 2, 3
# dynamic-block (start,end) indices in the 499 vector (static Python ints)
_SL = {b: BLOCK_RANGES[b] for b in DYNAMIC_BLOCKS}


class EnvParams(NamedTuple):
    """Static per-symbol + reward config (Python scalars -> hashable -> jit static arg)."""
    starting_balance: float
    position_size: float
    value_per_point: float
    cost_frac: float
    typical_range: float        # 0.0 means "None"
    is_index: float             # 1.0/0.0
    daily_dd_frac: float        # 0.05
    total_dd_frac: float        # 0.10
    profit_target_frac: float   # 0.10
    trailing_enabled: float     # 1.0/0.0
    two_phase_enabled: float    # 1.0/0.0
    phase2_continue: float      # 1.0/0.0
    phase2_trailing_frac: float # 0.01
    breach_penalty: float       # 1.0
    pass_bonus: float           # 1.0
    reward_scale: float         # 1.0
    ny_half_bonus: float        # 0.15
    ny_full_bonus: float        # 0.45
    ny_daily_target_frac: float # 0.025 (fixed target the NY session is measured against)
    open_gate: float            # 1.0/0.0 — block new directional opens when the 5m CCI is neutral
    max_bars: int
    T: int


class DeviceStatic(NamedTuple):
    """Shared, read-only device arrays the env indexes by bar. Same for ALL envs (vmap in_axes=None)."""
    static_obs: jnp.ndarray     # (T, 499)
    close: jnp.ndarray          # (T,)
    is_new_day: jnp.ndarray     # (T,)
    open_gate_blocked: jnp.ndarray  # (T,)
    minute_of_day: jnp.ndarray  # (T,)
    ref_move: jnp.ndarray       # (T,)
    week_avg: jnp.ndarray       # (T,)
    prev_day: jnp.ndarray       # (T,)
    prev2_day: jnp.ndarray      # (T,)
    today_sofar: jnp.ndarray    # (T,)


class EnvState(NamedTuple):
    ptr: jnp.ndarray
    end: jnp.ndarray
    active: jnp.ndarray
    balance: jnp.ndarray
    equity: jnp.ndarray
    starting_balance: jnp.ndarray
    day_start_balance: jnp.ndarray
    day_peak_equity: jnp.ndarray
    daily_realized_pnl: jnp.ndarray
    daily_wins: jnp.ndarray
    daily_losses: jnp.ndarray
    daily_trades: jnp.ndarray
    daily_consecutive_losses: jnp.ndarray
    episode_start_balance: jnp.ndarray
    episode_peak_equity: jnp.ndarray
    episode_realized_pnl: jnp.ndarray
    episode_wins: jnp.ndarray
    episode_losses: jnp.ndarray
    episode_trades: jnp.ndarray
    episode_consecutive_losses: jnp.ndarray
    episode_breached: jnp.ndarray
    episode_passed: jnp.ndarray
    position: jnp.ndarray
    entry_price: jnp.ndarray
    days_elapsed: jnp.ndarray
    phase2_active: jnp.ndarray
    phase2_peak: jnp.ndarray
    day_locked: jnp.ndarray
    ny_started: jnp.ndarray
    ny_start_realized: jnp.ndarray
    ny_half: jnp.ndarray
    ny_full: jnp.ndarray
    daily_target_frac: jnp.ndarray
    trailing_dd_frac: jnp.ndarray


def make_device_static(sd) -> DeviceStatic:
    """numpy StaticData -> device arrays (dtype follows the active x64 setting)."""
    return DeviceStatic(
        static_obs=jnp.asarray(sd.static_obs),
        close=jnp.asarray(sd.close),
        is_new_day=jnp.asarray(sd.is_new_day),
        open_gate_blocked=jnp.asarray(sd.open_gate_blocked),
        minute_of_day=jnp.asarray(sd.minute_of_day),
        ref_move=jnp.asarray(sd.ref_move),
        week_avg=jnp.asarray(sd.week_avg),
        prev_day=jnp.asarray(sd.prev_day),
        prev2_day=jnp.asarray(sd.prev2_day),
        today_sofar=jnp.asarray(sd.today_sofar),
    )


def params_from_static(sd, *, daily_target_frac=0.025, trailing_dd_frac=0.04,
                       daily_dd_frac=0.05, total_dd_frac=0.10, profit_target_frac=0.10,
                       trailing_enabled=1.0, two_phase_enabled=1.0, phase2_continue=0.0,
                       phase2_trailing_frac=0.01, breach_penalty=1.0, pass_bonus=1.0,
                       reward_scale=1.0, ny_half_bonus=0.15, ny_full_bonus=0.45,
                       ny_daily_target_frac=0.025, open_gate=0.0, max_bars=None) -> EnvParams:
    """Build EnvParams from a StaticData + the FTMO knobs (defaults = the locked numbers)."""
    return EnvParams(
        starting_balance=float(sd.starting_balance), position_size=float(sd.position_size),
        value_per_point=float(sd.value_per_point), cost_frac=float(sd.cost_frac),
        typical_range=float(sd.typical_range), is_index=float(sd.is_index),
        daily_dd_frac=float(daily_dd_frac), total_dd_frac=float(total_dd_frac),
        profit_target_frac=float(profit_target_frac), trailing_enabled=float(trailing_enabled),
        two_phase_enabled=float(two_phase_enabled), phase2_continue=float(phase2_continue),
        phase2_trailing_frac=float(phase2_trailing_frac), breach_penalty=float(breach_penalty),
        pass_bonus=float(pass_bonus), reward_scale=float(reward_scale),
        ny_half_bonus=float(ny_half_bonus), ny_full_bonus=float(ny_full_bonus),
        ny_daily_target_frac=float(ny_daily_target_frac), open_gate=float(open_gate),
        max_bars=int(max_bars if max_bars is not None else sd.T), T=int(sd.T),
    )


def init_state(static: DeviceStatic, params: EnvParams, start, end,
               daily_target_frac, trailing_dd_frac) -> EnvState:
    """Initial state at bar `start` (mirrors TradingEnv.reset; money fields follow array dtype)."""
    bal = jnp.asarray(static.close[start] * 0 + params.starting_balance)  # dtype of close
    z = bal * 0.0
    one = jnp.ones_like(z)
    return EnvState(
        ptr=jnp.asarray(start, jnp.int32), end=jnp.asarray(end, jnp.int32), active=jnp.float32(1.0),
        balance=bal, equity=bal, starting_balance=bal,
        day_start_balance=bal, day_peak_equity=bal, daily_realized_pnl=z,
        daily_wins=jnp.float32(0.0), daily_losses=jnp.float32(0.0),
        daily_trades=jnp.float32(0.0), daily_consecutive_losses=jnp.float32(0.0),
        episode_start_balance=bal, episode_peak_equity=bal, episode_realized_pnl=z,
        episode_wins=jnp.float32(0.0), episode_losses=jnp.float32(0.0),
        episode_trades=jnp.float32(0.0), episode_consecutive_losses=jnp.float32(0.0),
        episode_breached=jnp.float32(0.0), episode_passed=jnp.float32(0.0),
        position=z, entry_price=jnp.asarray(static.close[start]),
        days_elapsed=jnp.float32(0.0),
        phase2_active=jnp.float32(0.0), phase2_peak=z, day_locked=jnp.float32(0.0),
        ny_started=jnp.float32(0.0), ny_start_realized=z,
        ny_half=jnp.float32(0.0), ny_full=jnp.float32(0.0),
        daily_target_frac=jnp.float32(daily_target_frac), trailing_dd_frac=jnp.float32(trailing_dd_frac),
    )


def _dynamic_obs(s: EnvState, static: DeviceStatic, params: EnvParams, t) -> dict:
    """The 5 dynamic blocks from state `s` + per-bar scalars at bar `t`."""
    daily = jax_obs_blocks.daily_features(
        s.equity, s.day_start_balance, s.starting_balance, s.daily_realized_pnl,
        s.daily_wins, s.daily_trades, s.daily_consecutive_losses,
        s.daily_target_frac, params.daily_dd_frac)
    epi = jax_obs_blocks.episode_features(
        s.equity, s.episode_start_balance, s.starting_balance, s.episode_realized_pnl,
        s.episode_wins, s.episode_trades, s.episode_consecutive_losses, s.episode_breached,
        s.episode_peak_equity, s.trailing_dd_frac, params.total_dd_frac,
        params.trailing_enabled, params.profit_target_frac)
    port = jax_obs_blocks.portfolio_features(s.equity, s.balance, s.starting_balance, s.position)
    size = jax_obs_blocks.sizing_features(
        s.equity, s.day_start_balance, s.starting_balance, s.episode_peak_equity,
        params.value_per_point, static.ref_move[t], params.position_size,
        s.daily_target_frac, s.trailing_dd_frac, params.total_dd_frac, params.trailing_enabled)
    ctx = jax_obs_blocks.recent_context_features(
        s.equity, s.starting_balance, static.week_avg[t], static.prev_day[t], static.prev2_day[t],
        static.today_sofar[t], params.typical_range, s.days_elapsed,
        s.daily_target_frac, params.profit_target_frac)
    return {"account_daily": daily, "account_episode": epi, "portfolio": port,
            "sizing": size, "recent_context": ctx}


def _assemble_obs(s: EnvState, static: DeviceStatic, params: EnvParams, t) -> jnp.ndarray:
    """static row at bar t with the 5 dynamic blocks scattered in, then nan_to_num (== CPU builder)."""
    obs = static.static_obs[t].astype(jnp.float32)
    blocks = _dynamic_obs(s, static, params, t)
    for name in DYNAMIC_BLOCKS:
        a, b = _SL[name]
        obs = obs.at[a:b].set(blocks[name].astype(jnp.float32))
    return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def reset_obs(s: EnvState, static: DeviceStatic, params: EnvParams) -> jnp.ndarray:
    """Observation at the reset bar (no step taken yet)."""
    return _assemble_obs(s, static, params, s.ptr)


def step_env(s: EnvState, action, static: DeviceStatic, params: EnvParams):
    """One branchless step. Returns (new_state, obs, reward, terminated, truncated).
    Mirrors src/env/trading_env.py step() exactly (see that file's numbered comments)."""
    t = s.ptr
    a = jnp.asarray(action, jnp.int32)
    equity_before = s.equity
    close_t = static.close[t]
    size = params.position_size
    cost = params.cost_frac

    # --- 1) act at close[t]: pick target, apply day-lock, realize close, charge open cost ---
    target = jnp.where(a == _HOLD, s.position,
              jnp.where(a == _BUY, 1.0, jnp.where(a == _SELL, -1.0, 0.0)))
    # 5m CCI open-gate: forbid establishing a NEW direction when the 5m market is neutral (CPU order:
    # gate first, then day-lock). Both just force target=0 on a new open, so order doesn't change the result.
    new_open = ((target != 0.0) & (target != s.position)).astype(s.position.dtype)
    gate_blk = params.open_gate * static.open_gate_blocked[t] * new_open
    target = jnp.where(gate_blk > 0.5, 0.0, target)
    # two-phase day-lock: block a NEW open (force close) once the day is locked
    blocked = s.day_locked * ((target != 0.0) & (target != s.position)).astype(s.position.dtype)
    target = jnp.where(blocked > 0.5, 0.0, target)

    changed = (target != s.position)
    close_mask = (changed & (s.position != 0.0)).astype(close_t.dtype)
    open_mask = (changed & (target != 0.0)).astype(close_t.dtype)

    realized = s.position * (close_t - s.entry_price) * size - cost * close_t * size
    # record_close effects (gated by close_mask)
    balance = s.balance + realized * close_mask
    daily_realized = s.daily_realized_pnl + realized * close_mask
    episode_realized = s.episode_realized_pnl + realized * close_mask
    is_win = ((realized > 0.0) & (close_mask > 0.5)).astype(jnp.float32)
    is_loss = ((realized <= 0.0) & (close_mask > 0.5)).astype(jnp.float32)
    daily_trades = s.daily_trades + close_mask.astype(jnp.float32)
    episode_trades = s.episode_trades + close_mask.astype(jnp.float32)
    daily_wins = s.daily_wins + is_win
    episode_wins = s.episode_wins + is_win
    daily_losses = s.daily_losses + is_loss
    episode_losses = s.episode_losses + is_loss
    daily_consec = jnp.where(close_mask > 0.5,
                             jnp.where(is_win > 0.5, 0.0, s.daily_consecutive_losses + 1.0),
                             s.daily_consecutive_losses)
    episode_consec = jnp.where(close_mask > 0.5,
                               jnp.where(is_win > 0.5, 0.0, s.episode_consecutive_losses + 1.0),
                               s.episode_consecutive_losses)
    # mark_equity(balance) inside record_close updates peaks (gated by close_mask)
    day_peak = jnp.where(close_mask > 0.5, jnp.maximum(s.day_peak_equity, balance), s.day_peak_equity)
    epi_peak = jnp.where(close_mask > 0.5, jnp.maximum(s.episode_peak_equity, balance), s.episode_peak_equity)
    # open cost (gated by open_mask)
    ecost = cost * close_t * size
    balance = balance - ecost * open_mask
    daily_realized = daily_realized - ecost * open_mask
    episode_realized = episode_realized - ecost * open_mask
    entry_price = jnp.where(open_mask > 0.5, close_t, s.entry_price)
    position = target

    # --- 2) advance one bar, mark to market at close[t+1] ---
    t1 = jnp.minimum(t + 1, params.T - 1)
    close_t1 = static.close[t1]
    unrealized = position * (close_t1 - entry_price) * size
    equity = balance + unrealized
    day_peak = jnp.maximum(day_peak, equity)
    epi_peak = jnp.maximum(epi_peak, equity)

    # --- 3) reward = equity change / starting balance ---
    reward = ((equity - equity_before) / params.starting_balance) * params.reward_scale

    # --- 4) day boundary: pay NY bonus if the ending day passed, then reset day/phase2/NY ---
    new_day = static.is_new_day[t1].astype(jnp.float32)
    ny_target = params.ny_daily_target_frac * params.starting_balance
    day_passed_ny = (daily_realized >= ny_target).astype(jnp.float32)
    ny_bonus = params.is_index * day_passed_ny * (s.ny_half * params.ny_half_bonus
                                                  + s.ny_full * params.ny_full_bonus)
    reward = reward + new_day * ny_bonus

    def _reset(val, cur):  # apply reset value on a new day, else keep
        return jnp.where(new_day > 0.5, val, cur)

    day_start_balance = _reset(balance, s.day_start_balance)
    day_peak = _reset(equity, day_peak)
    daily_realized = _reset(jnp.zeros_like(daily_realized), daily_realized)
    daily_wins = _reset(0.0, daily_wins); daily_losses = _reset(0.0, daily_losses)
    daily_trades = _reset(0.0, daily_trades); daily_consec = _reset(0.0, daily_consec)
    phase2_active = _reset(0.0, s.phase2_active)
    phase2_peak = _reset(jnp.zeros_like(s.phase2_peak), s.phase2_peak)
    day_locked = _reset(0.0, s.day_locked)
    ny_started = _reset(0.0, s.ny_started)
    ny_start_realized = _reset(jnp.zeros_like(s.ny_start_realized), s.ny_start_realized)
    ny_half = _reset(0.0, s.ny_half); ny_full = _reset(0.0, s.ny_full)
    days_elapsed = s.days_elapsed + new_day

    # --- 4b) NY qualify (post-reset), index only, using minute_of_day[t1] ---
    mod = static.minute_of_day[t1].astype(jnp.float32)
    set_start = params.is_index * (mod >= 810.0).astype(jnp.float32) * (1.0 - ny_started)
    ny_start_realized = jnp.where(set_start > 0.5, daily_realized, ny_start_realized)
    ny_started = jnp.maximum(ny_started, set_start)
    session_closed = daily_realized - ny_start_realized
    qual_gate = params.is_index * ny_started * (session_closed > 0.0).astype(jnp.float32)
    half_win = ((mod >= 810.0) & (mod < 930.0)).astype(jnp.float32) * (session_closed >= 0.5 * ny_target).astype(jnp.float32)
    full_win = ((mod >= 810.0) & (mod < 990.0)).astype(jnp.float32) * (session_closed >= ny_target).astype(jnp.float32)
    ny_half = jnp.maximum(ny_half, qual_gate * half_win)
    ny_full = jnp.maximum(ny_full, qual_gate * full_win)

    # --- 5) breach check (post day-reset) -> terminate; else +10% pass ---
    br = jax_ftmo.breach(equity, day_start_balance, params.starting_balance, epi_peak,
                         params.daily_dd_frac, params.total_dd_frac, s.trailing_dd_frac,
                         params.trailing_enabled)
    any_breach = br["any_breach"]
    pass_hit = (1.0 - any_breach) * (equity >= params.starting_balance * (1.0 + params.profit_target_frac)).astype(jnp.float32)
    terminated = jnp.maximum(any_breach, pass_hit)
    reward = reward - params.breach_penalty * any_breach + params.pass_bonus * pass_hit
    episode_breached = jnp.maximum(s.episode_breached, any_breach)
    episode_passed = jnp.maximum(s.episode_passed, pass_hit)

    # --- 6) two-phase daily engine (gated: not terminated AND two_phase_enabled) ---
    gate = (1.0 - terminated) * params.two_phase_enabled
    flat_target = jax_ftmo.daily_target_hit(equity, day_start_balance, params.starting_balance, s.daily_target_frac)
    condA = gate * flat_target * (1.0 - phase2_active) * (1.0 - day_locked)        # bank +2.5%
    # phase-2 peak update + give-back trip (only when already in phase 2)
    phase2_peak_upd = jnp.where(phase2_active > 0.5, jnp.maximum(phase2_peak, equity), phase2_peak)
    give_back = phase2_peak_upd - equity
    trip = (give_back >= phase2_peak_upd * params.phase2_trailing_frac).astype(jnp.float32)
    condB = gate * phase2_active * trip                                            # gave back the 1%
    flat = jnp.maximum(condA, condB)

    # flatten(): realize the open position at close[t1], bank, zero position, mark
    realized_flat = position * (close_t1 - entry_price) * size - cost * close_t1 * size
    balance = balance + realized_flat * flat
    daily_realized = daily_realized + realized_flat * flat
    episode_realized = episode_realized + realized_flat * flat
    fis_win = ((realized_flat > 0.0) & (flat > 0.5)).astype(jnp.float32)
    fis_loss = ((realized_flat <= 0.0) & (flat > 0.5)).astype(jnp.float32)
    daily_trades = daily_trades + flat
    episode_trades = episode_trades + flat
    daily_wins = daily_wins + fis_win; episode_wins = episode_wins + fis_win
    daily_losses = daily_losses + fis_loss; episode_losses = episode_losses + fis_loss
    daily_consec = jnp.where(flat > 0.5, jnp.where(fis_win > 0.5, 0.0, daily_consec + 1.0), daily_consec)
    episode_consec = jnp.where(flat > 0.5, jnp.where(fis_win > 0.5, 0.0, episode_consec + 1.0), episode_consec)
    position = jnp.where(flat > 0.5, 0.0, position)
    equity = jnp.where(flat > 0.5, balance, equity)
    day_peak = jnp.where(flat > 0.5, jnp.maximum(day_peak, balance), day_peak)
    epi_peak = jnp.where(flat > 0.5, jnp.maximum(epi_peak, balance), epi_peak)
    # post-flatten phase2/day_lock state
    phase2_peak_new = jnp.where(condA > 0.5, equity, phase2_peak_upd)
    phase2_active_new = jnp.where(condA > 0.5, params.phase2_continue,
                                  jnp.where(condB > 0.5, 0.0, phase2_active))
    day_locked_new = jnp.where(condA > 0.5, 1.0 - params.phase2_continue,
                               jnp.where(condB > 0.5, 1.0, day_locked))

    truncated = (t1 >= s.end).astype(jnp.float32)

    ns = EnvState(
        ptr=t1, end=s.end, active=s.active,
        balance=balance, equity=equity, starting_balance=s.starting_balance,
        day_start_balance=day_start_balance, day_peak_equity=day_peak, daily_realized_pnl=daily_realized,
        daily_wins=daily_wins, daily_losses=daily_losses, daily_trades=daily_trades,
        daily_consecutive_losses=daily_consec,
        episode_start_balance=s.episode_start_balance, episode_peak_equity=epi_peak,
        episode_realized_pnl=episode_realized, episode_wins=episode_wins, episode_losses=episode_losses,
        episode_trades=episode_trades, episode_consecutive_losses=episode_consec,
        episode_breached=episode_breached, episode_passed=episode_passed,
        position=position, entry_price=entry_price, days_elapsed=days_elapsed,
        phase2_active=phase2_active_new, phase2_peak=phase2_peak_new, day_locked=day_locked_new,
        ny_started=ny_started, ny_start_realized=ny_start_realized, ny_half=ny_half, ny_full=ny_full,
        daily_target_frac=s.daily_target_frac, trailing_dd_frac=s.trailing_dd_frac,
    )
    obs = _assemble_obs(ns, static, params, t1)
    return ns, obs, reward, terminated, truncated


# Uniform env-interface alias (so the trainer/eval can be env-agnostic): init_state, reset_obs, step.
step = step_env
