# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The CORE goal: ONE bot trading the WHOLE book from ONE shared pot, on a TPU.
#      This is a branchless jnp reimplementation of src/env/portfolio_env.py: per-symbol
#      decisions cycling symbol-by-symbol, a shared equity/DD pot, the alpha-shaping
#      reward (USE+BEAT the alphas, PnL-capped), midnight day-scoring (won/failed day +
#      4-in-a-row), pot-level breach/+10% pass, and two-phase banking that flattens the
#      WHOLE book. Same 499 obs, same fingerprint -> ranked head-to-head with CPU policies.
# WHERE jax_tpu/jax_portfolio_env.py
# HOW   PortfolioState pytree (shared-pot scalars + per-symbol position/entry/alpha-entry
#       arrays + symbol cursor j). step decides symbol j at bar t; when j wraps, the bar
#       advances and the pot-level day/breach/two-phase logic runs. Per-symbol sequential
#       parts (flatten-all) unroll over the static symbol count for exact ordering parity.
# DEPENDS_ON: jax, jax_tpu.{jax_ftmo, jax_obs_blocks, jax_static_features}, config.constants
# USED_BY: jax_tpu/jax_trainer.py (portfolio mode), tests/test_jax_portfolio_parity.py
# CHANGE_NOTES(IRAC): I: single-symbol can't learn portfolio risk allocation across one pot.
#   R: portfolio_management-is-core-goal + portfolio_env.py reward model (alpha-shaping,
#   day-scoring, two-phase). A: port portfolio_env.step to branchless jnp, symbol-cycling,
#   shared pot. C: the actual product — one TPU bot balancing the whole FTMO book toward a
#   consistent pass — verified bar-for-bar against the CPU PortfolioEnv.
# =====================================================================
"""Branchless shared-pot PortfolioEnv in jnp (symbol-cycling) — 1:1 with src/env/portfolio_env.py."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax_tpu import jax_ftmo, jax_obs_blocks
from jax_tpu.jax_static_features import BLOCK_RANGES, DYNAMIC_BLOCKS

_HOLD, _BUY, _SELL, _CLOSE = 0, 1, 2, 3
_SL = {b: BLOCK_RANGES[b] for b in DYNAMIC_BLOCKS}


class PortfolioParams(NamedTuple):
    """Static config (Python scalars -> hashable jit static arg). Per-symbol arrays live in DeviceStatic."""
    N: int
    T: int
    max_bars: int
    starting_balance: float
    daily_dd_frac: float
    total_dd_frac: float
    profit_target_frac: float
    trailing_enabled: float
    two_phase_enabled: float
    phase2_continue: float
    phase2_trailing_frac: float
    breach_penalty: float
    pass_bonus: float
    reward_scale: float
    continue_after_pass: float
    alpha_on: float
    alpha_agree: float
    alpha_against: float
    alpha_beat: float
    day_pass_reward: float
    day_fail_penalty: float
    streak_bonus: float          # +per ADDITIONAL consecutive won day (escalation; replaces the every-4th jackpot)
    streak_bonus_cap: float
    target_seek_weight: float
    idle_day_penalty: float
    dd_proximity_coef: float
    # v1.7.0 trade-risk behaviours (default 0 = off -> identical to v1.6.0)
    bb_stop_enabled: float
    risk_based: float           # 1.0 -> risk-based per-trade sizing
    risk_frac: float            # risk_per_trade_pct / 100
    band_stack_bonus: float
    reentry_bonus: float


class PortfolioDeviceStatic(NamedTuple):
    static_obs: jnp.ndarray     # (N, T, 499)
    close: jnp.ndarray          # (N, T)
    is_new_day: jnp.ndarray     # (T,)
    ref_move: jnp.ndarray       # (N, T)
    week_avg: jnp.ndarray       # (N, T)
    prev_day: jnp.ndarray       # (N, T)
    prev2_day: jnp.ndarray      # (N, T)
    today_sofar: jnp.ndarray    # (N, T)
    alpha_matrix: jnp.ndarray   # (N, T, 64)
    occupancy: jnp.ndarray      # (N, 64)
    position_size: jnp.ndarray  # (N,)
    value_per_point: jnp.ndarray# (N,)
    typical_range: jnp.ndarray  # (N,)
    cost_frac: jnp.ndarray      # (N,)
    # v1.7.0 trade-risk per-bar refs (per symbol)
    atr1m: jnp.ndarray          # (N, T)
    bb200_1m_up: jnp.ndarray    # (N, T)
    bb200_1m_lo: jnp.ndarray    # (N, T)
    bb200_5m_up: jnp.ndarray    # (N, T)
    bb200_5m_lo: jnp.ndarray    # (N, T)
    bb10_1m_up: jnp.ndarray     # (N, T)
    bb10_1m_lo: jnp.ndarray     # (N, T)
    bb10_5m_up: jnp.ndarray     # (N, T)
    bb10_5m_lo: jnp.ndarray     # (N, T)


class PortfolioState(NamedTuple):
    t: jnp.ndarray
    j: jnp.ndarray
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
    position: jnp.ndarray        # (N,)
    entry: jnp.ndarray           # (N,)
    entry_agreed: jnp.ndarray    # (N,)
    entry_alpha_dir: jnp.ndarray # (N,)
    # v1.7.0 per-trade risk state (per symbol)
    trade_size: jnp.ndarray      # (N,) actual (risk-based) size of the open trade
    entry_bar: jnp.ndarray       # (N,)
    entry_atr: jnp.ndarray       # (N,)
    entry_stop_band: jnp.ndarray # (N,)
    mfe_atr: jnp.ndarray         # (N,)
    mae_atr: jnp.ndarray         # (N,)
    last_close_bar: jnp.ndarray  # (N,)
    last_close_dir: jnp.ndarray  # (N,)
    last_exit_px: jnp.ndarray    # (N,)
    entry_band_long: jnp.ndarray # (N,)
    entry_band_short: jnp.ndarray# (N,)
    entry_reentry: jnp.ndarray   # (N,)
    days_elapsed: jnp.ndarray
    daily_pass_streak: jnp.ndarray
    phase2_active: jnp.ndarray
    phase2_peak: jnp.ndarray
    day_locked: jnp.ndarray
    day_progress_hwm: jnp.ndarray
    day_had_exposure: jnp.ndarray
    daily_target_frac: jnp.ndarray
    trailing_dd_frac: jnp.ndarray


def make_portfolio_device_static(psd) -> PortfolioDeviceStatic:
    j = jnp.asarray
    return PortfolioDeviceStatic(
        static_obs=j(psd.static_obs), close=j(psd.close), is_new_day=j(psd.is_new_day),
        ref_move=j(psd.ref_move), week_avg=j(psd.week_avg), prev_day=j(psd.prev_day),
        prev2_day=j(psd.prev2_day), today_sofar=j(psd.today_sofar),
        alpha_matrix=j(psd.alpha_matrix), occupancy=j(psd.occupancy),
        position_size=j(psd.position_size), value_per_point=j(psd.value_per_point),
        typical_range=j(psd.typical_range), cost_frac=j(psd.cost_frac),
        atr1m=j(psd.atr1m), bb200_1m_up=j(psd.bb200_1m_up), bb200_1m_lo=j(psd.bb200_1m_lo),
        bb200_5m_up=j(psd.bb200_5m_up), bb200_5m_lo=j(psd.bb200_5m_lo),
        bb10_1m_up=j(psd.bb10_1m_up), bb10_1m_lo=j(psd.bb10_1m_lo),
        bb10_5m_up=j(psd.bb10_5m_up), bb10_5m_lo=j(psd.bb10_5m_lo))


def portfolio_params(psd, *, daily_target_frac=0.025, trailing_dd_frac=0.04, daily_dd_frac=0.05,
                     total_dd_frac=0.10, profit_target_frac=0.10, trailing_enabled=1.0,
                     two_phase_enabled=1.0, phase2_continue=1.0, phase2_trailing_frac=0.01,
                     breach_penalty=20.0, pass_bonus=1.0, reward_scale=1.0, continue_after_pass=1.0,
                     alpha_on=1.0, alpha_agree=0.01, alpha_against=0.01, alpha_beat=0.05,
                     day_pass_reward=10.0, day_fail_penalty=5.0, streak_bonus=1.0, streak_bonus_cap=10.0,
                     target_seek_weight=3.0, idle_day_penalty=0.02, dd_proximity_coef=2.0,
                     bb_stop_enabled=0.0, risk_based=0.0, risk_frac=0.0,
                     band_stack_bonus=0.0, reentry_bonus=0.0,
                     max_bars=None) -> PortfolioParams:
    """Build PortfolioParams from a PortfolioStaticData + the FTMO/alpha knobs (defaults = FTMOConfig)."""
    return PortfolioParams(
        N=int(psd.N), T=int(psd.T), max_bars=int(max_bars if max_bars is not None else psd.T),
        starting_balance=float(psd.starting_balance), daily_dd_frac=float(daily_dd_frac),
        total_dd_frac=float(total_dd_frac), profit_target_frac=float(profit_target_frac),
        trailing_enabled=float(trailing_enabled), two_phase_enabled=float(two_phase_enabled),
        phase2_continue=float(phase2_continue), phase2_trailing_frac=float(phase2_trailing_frac),
        breach_penalty=float(breach_penalty), pass_bonus=float(pass_bonus),
        reward_scale=float(reward_scale), continue_after_pass=float(continue_after_pass),
        alpha_on=float(alpha_on), alpha_agree=float(alpha_agree), alpha_against=float(alpha_against),
        alpha_beat=float(alpha_beat), day_pass_reward=float(day_pass_reward),
        day_fail_penalty=float(day_fail_penalty), streak_bonus=float(streak_bonus),
        streak_bonus_cap=float(streak_bonus_cap), target_seek_weight=float(target_seek_weight),
        idle_day_penalty=float(idle_day_penalty), dd_proximity_coef=float(dd_proximity_coef),
        bb_stop_enabled=float(bb_stop_enabled), risk_based=float(risk_based), risk_frac=float(risk_frac),
        band_stack_bonus=float(band_stack_bonus), reentry_bonus=float(reentry_bonus))


def init_state(static: PortfolioDeviceStatic, params: PortfolioParams, start, end,
               daily_target_frac, trailing_dd_frac) -> PortfolioState:
    N = params.N
    bal = jnp.asarray(static.close[0, start] * 0 + params.starting_balance)
    z = bal * 0.0
    entry0 = static.close[:, start].astype(bal.dtype)   # entry[s]=close[s,start] (mirrors reset)
    zf = jnp.zeros((N,), jnp.float32)
    zN = jnp.zeros((N,), bal.dtype)                      # (N,) money-dtype zeros
    bar0 = jnp.full((N,), start, jnp.int32)
    return PortfolioState(
        t=jnp.asarray(start, jnp.int32), j=jnp.int32(0), end=jnp.asarray(end, jnp.int32),
        active=jnp.float32(1.0),
        balance=bal, equity=bal, starting_balance=bal,
        day_start_balance=bal, day_peak_equity=bal, daily_realized_pnl=z,
        daily_wins=jnp.float32(0.0), daily_losses=jnp.float32(0.0), daily_trades=jnp.float32(0.0),
        daily_consecutive_losses=jnp.float32(0.0),
        episode_start_balance=bal, episode_peak_equity=bal, episode_realized_pnl=z,
        episode_wins=jnp.float32(0.0), episode_losses=jnp.float32(0.0), episode_trades=jnp.float32(0.0),
        episode_consecutive_losses=jnp.float32(0.0),
        episode_breached=jnp.float32(0.0), episode_passed=jnp.float32(0.0),
        position=jnp.zeros((N,), bal.dtype), entry=entry0,
        entry_agreed=zf, entry_alpha_dir=zf,
        trade_size=jnp.asarray(static.position_size).astype(bal.dtype),   # base size until an open sets it
        entry_bar=bar0, entry_atr=zN, entry_stop_band=zN, mfe_atr=zN, mae_atr=zN,
        last_close_bar=bar0, last_close_dir=zN, last_exit_px=entry0,
        entry_band_long=zN, entry_band_short=zN, entry_reentry=zN,
        days_elapsed=jnp.float32(0.0), daily_pass_streak=jnp.float32(0.0),
        phase2_active=jnp.float32(0.0), phase2_peak=z, day_locked=jnp.float32(0.0),
        # day_progress_hwm is derived from equity (money dtype) each step -> init in the MONEY dtype so the
        # lax.scan training carry stays dtype-stable under x64 too (float32 on TPU is unchanged).
        day_progress_hwm=z, day_had_exposure=jnp.float32(0.0),
        daily_target_frac=jnp.float32(daily_target_frac), trailing_dd_frac=jnp.float32(trailing_dd_frac))


def _consensus(am_row, occ_row, d):
    """(agree, disagree, net_dir) among FIRING, UNMASKED alphas at this bar for direction d.
    1:1 with PortfolioEnv._alpha_consensus."""
    fired = (am_row != 0.0) & (occ_row > 0.5)
    nf = jnp.sum(fired.astype(jnp.float32))
    buys = jnp.sum(((am_row > 0.0) & fired).astype(jnp.float32))
    sells = jnp.sum(((am_row < 0.0) & fired).astype(jnp.float32))
    nf_safe = jnp.maximum(nf, 1.0)
    has = (nf > 0.0).astype(jnp.float32)
    net_dir = has * jnp.where(buys > sells, 1.0, jnp.where(sells > buys, -1.0, 0.0))
    dpos = (d > 0.0).astype(jnp.float32); dneg = (d < 0.0).astype(jnp.float32)
    agree = has * (dpos * buys + dneg * sells) / nf_safe
    disagree = has * (dpos * sells + dneg * buys) / nf_safe
    return agree, disagree, net_dir


def _dynamic_obs(s: PortfolioState, static: PortfolioDeviceStatic, params: PortfolioParams, j, t):
    daily = jax_obs_blocks.daily_features(
        s.equity, s.day_start_balance, s.starting_balance, s.daily_realized_pnl,
        s.daily_wins, s.daily_trades, s.daily_consecutive_losses, s.daily_target_frac, params.daily_dd_frac)
    epi = jax_obs_blocks.episode_features(
        s.equity, s.episode_start_balance, s.starting_balance, s.episode_realized_pnl,
        s.episode_wins, s.episode_trades, s.episode_consecutive_losses, s.episode_breached,
        s.episode_peak_equity, s.trailing_dd_frac, params.total_dd_frac, params.trailing_enabled,
        params.profit_target_frac)
    # pot aggregates across ALL symbols (PortfolioEnv._set_aggregates)
    pos = s.position
    Nf = float(params.N)
    open_positions = jnp.sum((pos != 0.0).astype(jnp.float32))
    net_exposure = jnp.clip(jnp.sum(pos) / Nf, -1.0, 1.0)
    gross_exposure = jnp.clip(jnp.sum(jnp.abs(pos)) / Nf, 0.0, 1.0)
    largest_dir = jnp.sign(pos[jnp.argmax(jnp.abs(pos))])
    port = jax_obs_blocks.portfolio_features_agg(
        s.equity, s.balance, s.starting_balance, open_positions, net_exposure, gross_exposure, largest_dir)
    size = jax_obs_blocks.sizing_features(
        s.equity, s.day_start_balance, s.starting_balance, s.episode_peak_equity,
        static.value_per_point[j], static.ref_move[j, t], static.position_size[j],
        s.daily_target_frac, s.trailing_dd_frac, params.total_dd_frac, params.trailing_enabled)
    ctx = jax_obs_blocks.recent_context_features(
        s.equity, s.starting_balance, static.week_avg[j, t], static.prev_day[j, t], static.prev2_day[j, t],
        static.today_sofar[j, t], static.typical_range[j], s.days_elapsed,
        s.daily_target_frac, params.profit_target_frac)
    tr = jax_obs_blocks.trade_risk_features(
        s.position[j], s.entry[j], static.close[j, t], s.trade_size[j], s.equity,
        s.entry_atr[j], static.atr1m[j, t], s.entry_stop_band[j],
        t.astype(jnp.float32) - s.entry_bar[j].astype(jnp.float32), s.mfe_atr[j], s.mae_atr[j],
        t.astype(jnp.float32) - s.last_close_bar[j].astype(jnp.float32), s.last_close_dir[j], s.last_exit_px[j],
        static.bb200_1m_up[j, t], static.bb200_1m_lo[j, t], static.bb200_5m_up[j, t], static.bb200_5m_lo[j, t],
        static.bb10_1m_up[j, t], static.bb10_1m_lo[j, t], static.bb10_5m_up[j, t], static.bb10_5m_lo[j, t])
    return {"account_daily": daily, "account_episode": epi, "portfolio": port,
            "sizing": size, "recent_context": ctx, "trade_risk": tr}


def _assemble_obs(s: PortfolioState, static: PortfolioDeviceStatic, params: PortfolioParams, j, t):
    obs = static.static_obs[j, t].astype(jnp.float32)
    blocks = _dynamic_obs(s, static, params, j, t)
    for name in DYNAMIC_BLOCKS:
        a, b = _SL[name]
        obs = obs.at[a:b].set(blocks[name].astype(jnp.float32))
    return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def reset_obs(s: PortfolioState, static: PortfolioDeviceStatic, params: PortfolioParams):
    return _assemble_obs(s, static, params, s.j, s.t)


def step_portfolio(s: PortfolioState, action, static: PortfolioDeviceStatic, params: PortfolioParams):
    """One branchless step (decides symbol j). 1:1 with src/env/portfolio_env.py step()."""
    N = params.N
    t = s.t; j = s.j
    eq_before = s.equity
    start = params.starting_balance
    close_jt = static.close[j, t]
    psize_j = static.position_size[j]
    cost_j = static.cost_frac[j]
    pos_j = s.position[j]
    entry_j = s.entry[j]

    # --- pick target for symbol j, apply day-lock ---
    target = jnp.where(action == _HOLD, pos_j,
              jnp.where(action == _BUY, 1.0, jnp.where(action == _SELL, -1.0, 0.0)))
    blocked = s.day_locked * ((target != 0.0) & (target != pos_j)).astype(pos_j.dtype)
    target = jnp.where(blocked > 0.5, 0.0, target)

    changed = (target != pos_j)
    close_mask = (changed & (pos_j != 0.0)).astype(close_jt.dtype)
    open_mask = (changed & (target != 0.0)).astype(close_jt.dtype)

    # --- close leg (symbol j): realize + record_close (uses the trade's ACTUAL risk-based size) ---
    ts_old = s.trade_size[j]
    realized = pos_j * (close_jt - entry_j) * ts_old - cost_j * close_jt * ts_old
    balance = s.balance + realized * close_mask
    daily_realized = s.daily_realized_pnl + realized * close_mask
    episode_realized = s.episode_realized_pnl + realized * close_mask
    is_win = ((realized > 0.0) & (close_mask > 0.5)).astype(jnp.float32)
    is_loss = ((realized <= 0.0) & (close_mask > 0.5)).astype(jnp.float32)
    daily_trades = s.daily_trades + close_mask.astype(jnp.float32)
    episode_trades = s.episode_trades + close_mask.astype(jnp.float32)
    daily_wins = s.daily_wins + is_win; episode_wins = s.episode_wins + is_win
    daily_losses = s.daily_losses + is_loss; episode_losses = s.episode_losses + is_loss
    daily_consec = jnp.where(close_mask > 0.5, jnp.where(is_win > 0.5, 0.0, s.daily_consecutive_losses + 1.0),
                             s.daily_consecutive_losses)
    episode_consec = jnp.where(close_mask > 0.5, jnp.where(is_win > 0.5, 0.0, s.episode_consecutive_losses + 1.0),
                               s.episode_consecutive_losses)
    day_peak = jnp.where(close_mask > 0.5, jnp.maximum(s.day_peak_equity, balance), s.day_peak_equity)
    epi_peak = jnp.where(close_mask > 0.5, jnp.maximum(s.episode_peak_equity, balance), s.episode_peak_equity)
    # v1.7.0 re-entry context recorded on the AGENT close leg (the two-phase flatten + hard stop below set
    # their own; the flatten does NOT, matching CPU)
    lc = close_mask > 0.5
    last_close_bar_j = jnp.where(lc, t.astype(jnp.int32), s.last_close_bar[j])
    last_close_dir_j = jnp.where(lc, pos_j, s.last_close_dir[j])
    last_exit_px_j = jnp.where(lc, close_jt, s.last_exit_px[j])

    # shaping at CLOSE (profitable close, day net up, capped at PnL): alpha USE+BEAT + band-stack + re-entry
    day0 = s.day_start_balance
    pnl_frac = realized / start
    move = close_jt - entry_j
    alpha_gross = s.entry_alpha_dir[j] * move * ts_old
    bot_gross = pos_j * move * ts_old
    beat_term = jnp.minimum(params.alpha_beat, (bot_gross - alpha_gross) / start) * (bot_gross > alpha_gross).astype(jnp.float32)
    alpha_part = params.alpha_on * (s.entry_agreed[j] * params.alpha_agree + beat_term)
    band_ind = ((pos_j > 0.0).astype(jnp.float32) * s.entry_band_long[j]
                + (pos_j < 0.0).astype(jnp.float32) * s.entry_band_short[j])
    bonus = alpha_part + params.band_stack_bonus * band_ind + params.reentry_bonus * s.entry_reentry[j]
    close_active = close_mask * (realized > 0.0).astype(jnp.float32) * (balance > day0).astype(jnp.float32)
    alpha_shaping = close_active * jnp.minimum(bonus, pnl_frac)

    # clear entry tracking on close (then open may overwrite below)
    agreed_tmp = jnp.where(close_mask > 0.5, 0.0, s.entry_agreed[j])
    dir_tmp = jnp.where(close_mask > 0.5, 0.0, s.entry_alpha_dir[j])

    # --- open leg (symbol j): RISK-BASED size + cost + consensus + entry stamping ---
    base_j = psize_j
    sb = jnp.where(target > 0.0, static.bb10_1m_lo[j, t], static.bb10_1m_up[j, t])   # the BB(10,1) hard-stop band
    dist = jnp.abs(close_jt - sb)
    risk_dollars = start * params.risk_frac
    ts_risk = jnp.where((params.risk_based > 0.5) & jnp.isfinite(sb) & (dist > 1e-12),
                        jnp.minimum(base_j, risk_dollars / jnp.maximum(dist, 1e-12)), base_j)
    ts_new = jnp.where(open_mask > 0.5, ts_risk, ts_old)            # only an OPEN changes the size
    ecost = cost_j * close_jt * ts_new
    balance = balance - ecost * open_mask
    daily_realized = daily_realized - ecost * open_mask
    episode_realized = episode_realized - ecost * open_mask
    agree, disagree, net_dir = _consensus(static.alpha_matrix[j, t], static.occupancy[j], target)
    against = params.alpha_on * open_mask * (disagree >= 0.5).astype(jnp.float32) * params.alpha_against
    alpha_shaping = alpha_shaping - against
    agreed_j = jnp.where(open_mask > 0.5, (agree >= 0.5).astype(jnp.float32), agreed_tmp)
    dir_j = jnp.where(open_mask > 0.5, net_dir, dir_tmp)
    entry_j_new = jnp.where(open_mask > 0.5, close_jt, entry_j)
    # v1.7.0 per-trade risk state on the open leg
    op = open_mask > 0.5
    entry_bar_j = jnp.where(op, t.astype(jnp.int32), s.entry_bar[j])
    entry_atr_j = jnp.where(op, static.atr1m[j, t], s.entry_atr[j])
    entry_stop_band_j = jnp.where(op, sb, s.entry_stop_band[j])
    mfe_j = jnp.where(op, jnp.zeros_like(s.mfe_atr[j]), s.mfe_atr[j])
    mae_j = jnp.where(op, jnp.zeros_like(s.mae_atr[j]), s.mae_atr[j])
    p_open = close_jt
    bsl = ((p_open > static.bb200_1m_up[j, t]) & (p_open > static.bb10_1m_up[j, t])
           & (p_open > static.bb200_5m_up[j, t]) & (p_open > static.bb10_5m_up[j, t])).astype(jnp.float32)
    bss = ((p_open < static.bb200_1m_lo[j, t]) & (p_open < static.bb10_1m_lo[j, t])
           & (p_open < static.bb200_5m_lo[j, t]) & (p_open < static.bb10_5m_lo[j, t])).astype(jnp.float32)
    entry_band_long_j = jnp.where(op, bsl, s.entry_band_long[j])
    entry_band_short_j = jnp.where(op, bss, s.entry_band_short[j])
    cont = last_close_dir_j * (close_jt - last_exit_px_j)          # with-trend re-entry (uses post-close-leg last_close)
    reentry_raw = ((last_close_dir_j != 0.0) & (target == last_close_dir_j) & (cont > 0.0)).astype(jnp.float32)
    entry_reentry_j = jnp.where(op, reentry_raw, s.entry_reentry[j])

    position = s.position.at[j].set(target)
    entry = s.entry.at[j].set(entry_j_new)
    entry_agreed = s.entry_agreed.at[j].set(agreed_j)
    entry_alpha_dir = s.entry_alpha_dir.at[j].set(dir_j)
    trade_size = s.trade_size.at[j].set(ts_new.astype(s.trade_size.dtype))
    entry_bar = s.entry_bar.at[j].set(entry_bar_j)
    entry_atr = s.entry_atr.at[j].set(entry_atr_j.astype(s.entry_atr.dtype))
    entry_stop_band = s.entry_stop_band.at[j].set(entry_stop_band_j.astype(s.entry_stop_band.dtype))
    mfe_atr = s.mfe_atr.at[j].set(mfe_j); mae_atr = s.mae_atr.at[j].set(mae_j)
    last_close_bar = s.last_close_bar.at[j].set(last_close_bar_j)
    last_close_dir = s.last_close_dir.at[j].set(last_close_dir_j.astype(s.last_close_dir.dtype))
    last_exit_px = s.last_exit_px.at[j].set(last_exit_px_j.astype(s.last_exit_px.dtype))
    entry_band_long = s.entry_band_long.at[j].set(entry_band_long_j)
    entry_band_short = s.entry_band_short.at[j].set(entry_band_short_j)
    entry_reentry = s.entry_reentry.at[j].set(entry_reentry_j)

    # --- advance cursor: next symbol; wrap -> advance the bar ---
    j2 = j + 1
    bar_adv = (j2 >= N).astype(jnp.float32)
    j_new = jnp.where(j2 >= N, 0, j2).astype(jnp.int32)
    t_new = jnp.where(j2 >= N, jnp.minimum(t + 1, params.T - 1), t).astype(jnp.int32)

    # --- mark the pot at the (possibly advanced) bar (uses each trade's ACTUAL risk-based size) ---
    close_col = static.close[:, t_new]
    unreal = jnp.sum(position * (close_col - entry) * trade_size)
    equity = balance + unreal
    day_peak = jnp.maximum(day_peak, equity)
    epi_peak = jnp.maximum(epi_peak, equity)
    # v1.7.0: update each open trade's max favorable / adverse excursion (ATR units) at the new bar
    in_pos = (position != 0.0) & (entry_atr > 0.0)
    exc = jnp.where(in_pos, position * (close_col - entry) / jnp.maximum(entry_atr, 1e-9), 0.0)
    mfe_atr = jnp.where(in_pos, jnp.maximum(mfe_atr, exc), mfe_atr)
    mae_atr = jnp.where(in_pos, jnp.maximum(mae_atr, -exc), mae_atr)

    reward = ((equity - eq_before) / start) * params.reward_scale + alpha_shaping

    # SEEK-THE-TARGET (dense): reward NEW progress toward today's +2.5% target (high-water-mark so it can't
    # be farmed). Uses the pre-reset day_start (s.day_start_balance) + the just-marked equity. 1:1 with CPU.
    target_amt_seek = s.daily_target_frac * start
    day_progress = jnp.clip(jnp.where(target_amt_seek > 0.0, (equity - s.day_start_balance) / target_amt_seek, 0.0), 0.0, 1.0)
    reward = reward + params.target_seek_weight * jnp.maximum(0.0, day_progress - s.day_progress_hwm)
    day_progress_hwm = jnp.maximum(s.day_progress_hwm, day_progress)
    # DRAWDOWN-PROXIMITY (dense): gradual penalty as equity nears the trailing wall (plan AWAY from it). 1:1 CPU.
    wall = s.trailing_dd_frac
    dd_frac = jnp.maximum(0.0, (epi_peak - equity) / epi_peak)
    reward = reward - params.dd_proximity_coef * jnp.where(wall > 0.0, jnp.minimum(dd_frac / wall, 1.0), 0.0) ** 2
    # ANTI-HIDE: track whether the bot was EXPOSED (held any position) at any point today.
    exposed_now = (jnp.sum(jnp.abs(position)) > 0.0).astype(jnp.float32)
    day_had_exposure = jnp.maximum(s.day_had_exposure, exposed_now)

    # ============================ bar-advance block ============================
    new_day = static.is_new_day[t_new].astype(jnp.float32) * bar_adv

    # midnight day-scoring (won/failed day + 4-in-a-row), using pre-reset equity vs day0
    d0 = s.day_start_balance
    won = (equity - d0 >= s.daily_target_frac * start).astype(jnp.float32)
    streak_if_won = s.daily_pass_streak + 1.0
    # ESCALATING streak bonus: every ADDITIONAL consecutive won day pays more, capped (replaces the every-4th
    # jackpot). day N of a streak pays day_pass + streak_bonus*min(N-1, cap). 1:1 with PortfolioEnv.
    streak_amt = params.streak_bonus * jnp.minimum(streak_if_won - 1.0, params.streak_bonus_cap)
    day_reward = won * (params.day_pass_reward + streak_amt) - (1.0 - won) * params.day_fail_penalty
    reward = reward + new_day * day_reward
    # ANTI-HIDE: penalise a day the bot was FLAT all day (no exposure). Uses day_had_exposure accumulated
    # over the day; re-seeds below from the carried position (holding across midnight = exposed).
    idle = (1.0 - day_had_exposure)
    reward = reward - new_day * idle * params.idle_day_penalty
    daily_pass_streak = jnp.where(new_day > 0.5, jnp.where(won > 0.5, streak_if_won, 0.0), s.daily_pass_streak)

    def _rst(val, cur):
        return jnp.where(new_day > 0.5, val, cur)
    day_progress_hwm = _rst(0.0, day_progress_hwm)            # new day -> reset the seek high-water mark
    day_had_exposure = _rst(exposed_now, day_had_exposure)    # new day -> re-seed from carried position
    day_start_balance = _rst(balance, s.day_start_balance)
    day_peak = _rst(equity, day_peak)
    daily_realized = _rst(jnp.zeros_like(daily_realized), daily_realized)
    daily_wins = _rst(0.0, daily_wins); daily_losses = _rst(0.0, daily_losses)
    daily_trades = _rst(0.0, daily_trades); daily_consec = _rst(0.0, daily_consec)
    phase2_active = _rst(0.0, s.phase2_active)
    phase2_peak = _rst(jnp.zeros_like(s.phase2_peak), s.phase2_peak)
    day_locked = _rst(0.0, s.day_locked)
    days_elapsed = s.days_elapsed + new_day

    # breach (post day-reset) -> terminate; else +10% pass
    br = jax_ftmo.breach(equity, day_start_balance, start, epi_peak, params.daily_dd_frac,
                         params.total_dd_frac, s.trailing_dd_frac, params.trailing_enabled)
    any_breach = br["any_breach"] * bar_adv
    pass_raw = (equity >= start * (1.0 + params.profit_target_frac)).astype(jnp.float32)
    pass_hit = (1.0 - any_breach) * bar_adv * pass_raw
    term_on_pass = pass_hit * (1.0 - params.continue_after_pass)
    terminated = jnp.maximum(any_breach, term_on_pass)
    reward = reward - params.breach_penalty * any_breach + params.pass_bonus * term_on_pass
    episode_breached = jnp.maximum(s.episode_breached, any_breach)
    episode_passed = jnp.maximum(s.episode_passed, pass_hit)

    # v1.7.0 BB HARD STOP (protective): close any position past the 1m BB(10,1) opposite band, BEFORE two-phase
    # banking. Unrolled in symbol order for exact tally parity; equity-neutral (closes at the marked bar) so it
    # does not change this step's reward. OFF unless bb_stop_enabled. 1:1 with PortfolioEnv._apply_bb_hard_stop.
    stop_gate = params.bb_stop_enabled * bar_adv * (1.0 - terminated)
    for sidx in range(N):
        p = position[sidx]; px = close_col[sidx]
        lo = static.bb10_1m_lo[sidx, t_new]; up = static.bb10_1m_up[sidx, t_new]
        hit_long = (p > 0.0) & jnp.isfinite(lo) & (px < lo)
        hit_short = (p < 0.0) & jnp.isfinite(up) & (px > up)
        do_s = stop_gate * (hit_long | hit_short).astype(close_col.dtype)
        ts_s = trade_size[sidx]
        r_s = p * (px - entry[sidx]) * ts_s - static.cost_frac[sidx] * px * ts_s
        balance = balance + r_s * do_s
        daily_realized = daily_realized + r_s * do_s
        episode_realized = episode_realized + r_s * do_s
        w = ((r_s > 0.0) & (do_s > 0.5)).astype(jnp.float32)
        l = ((r_s <= 0.0) & (do_s > 0.5)).astype(jnp.float32)
        daily_trades = daily_trades + do_s.astype(jnp.float32)
        episode_trades = episode_trades + do_s.astype(jnp.float32)
        daily_wins = daily_wins + w; episode_wins = episode_wins + w
        daily_losses = daily_losses + l; episode_losses = episode_losses + l
        daily_consec = jnp.where(do_s > 0.5, jnp.where(w > 0.5, 0.0, daily_consec + 1.0), daily_consec)
        episode_consec = jnp.where(do_s > 0.5, jnp.where(w > 0.5, 0.0, episode_consec + 1.0), episode_consec)
        last_close_bar = last_close_bar.at[sidx].set(jnp.where(do_s > 0.5, t_new, last_close_bar[sidx]))
        last_close_dir = last_close_dir.at[sidx].set(jnp.where(do_s > 0.5, p, last_close_dir[sidx]))
        last_exit_px = last_exit_px.at[sidx].set(jnp.where(do_s > 0.5, px, last_exit_px[sidx]))
        position = position.at[sidx].set(jnp.where(do_s > 0.5, jnp.zeros_like(p), p))
        day_peak = jnp.where(do_s > 0.5, jnp.maximum(day_peak, balance), day_peak)
        epi_peak = jnp.where(do_s > 0.5, jnp.maximum(epi_peak, balance), epi_peak)
    # re-mark after the hard stop (equity-neutral; positions/balance changed) — mirrors CPU _mark()
    equity = balance + jnp.sum(position * (close_col - entry) * trade_size)
    day_peak = jnp.maximum(day_peak, equity)
    epi_peak = jnp.maximum(epi_peak, equity)

    # two-phase banking on the SHARED pot (flatten the whole book)
    gate = bar_adv * (1.0 - terminated) * params.two_phase_enabled
    open_mask_all = (position != 0.0).astype(close_col.dtype)
    pending_exit = jnp.sum(static.cost_frac * close_col * trade_size * open_mask_all)
    target_amt = s.daily_target_frac * start
    net_equity = equity - pending_exit
    hit_net = (net_equity - day_start_balance >= target_amt).astype(jnp.float32)
    condA = gate * hit_net * (1.0 - phase2_active) * (1.0 - day_locked)
    phase2_peak_upd = jnp.where(phase2_active > 0.5, jnp.maximum(phase2_peak, equity), phase2_peak)
    give_back = phase2_peak_upd - equity
    trip = (give_back >= phase2_peak_upd * params.phase2_trailing_frac).astype(jnp.float32)
    condB = gate * phase2_active * trip
    flat = jnp.maximum(condA, condB)

    # flatten_all: close EVERY open position at close[:, t_new], in symbol order (exact tally/peak parity)
    for sidx in range(N):
        do_s = flat * (position[sidx] != 0.0).astype(close_col.dtype)
        r_s = position[sidx] * (close_col[sidx] - entry[sidx]) * trade_size[sidx] \
            - static.cost_frac[sidx] * close_col[sidx] * trade_size[sidx]
        balance = balance + r_s * do_s
        daily_realized = daily_realized + r_s * do_s
        episode_realized = episode_realized + r_s * do_s
        w = ((r_s > 0.0) & (do_s > 0.5)).astype(jnp.float32)
        l = ((r_s <= 0.0) & (do_s > 0.5)).astype(jnp.float32)
        daily_trades = daily_trades + do_s.astype(jnp.float32)
        episode_trades = episode_trades + do_s.astype(jnp.float32)
        daily_wins = daily_wins + w; episode_wins = episode_wins + w
        daily_losses = daily_losses + l; episode_losses = episode_losses + l
        daily_consec = jnp.where(do_s > 0.5, jnp.where(w > 0.5, 0.0, daily_consec + 1.0), daily_consec)
        episode_consec = jnp.where(do_s > 0.5, jnp.where(w > 0.5, 0.0, episode_consec + 1.0), episode_consec)
        day_peak = jnp.where(do_s > 0.5, jnp.maximum(day_peak, balance), day_peak)
        epi_peak = jnp.where(do_s > 0.5, jnp.maximum(epi_peak, balance), epi_peak)
    position = jnp.where(flat > 0.5, jnp.zeros_like(position), position)
    equity = jnp.where(flat > 0.5, balance, equity)
    day_peak = jnp.where(flat > 0.5, jnp.maximum(day_peak, balance), day_peak)
    epi_peak = jnp.where(flat > 0.5, jnp.maximum(epi_peak, balance), epi_peak)

    phase2_peak = jnp.where(condA > 0.5, equity, phase2_peak_upd)
    phase2_active = jnp.where(condA > 0.5, params.phase2_continue, jnp.where(condB > 0.5, 0.0, phase2_active))
    day_locked = jnp.where(condA > 0.5, 1.0 - params.phase2_continue, jnp.where(condB > 0.5, 1.0, day_locked))

    truncated = (t_new >= s.end).astype(jnp.float32) * bar_adv

    ns = PortfolioState(
        t=t_new, j=j_new, end=s.end, active=s.active,
        balance=balance, equity=equity, starting_balance=s.starting_balance,
        day_start_balance=day_start_balance, day_peak_equity=day_peak, daily_realized_pnl=daily_realized,
        daily_wins=daily_wins, daily_losses=daily_losses, daily_trades=daily_trades,
        daily_consecutive_losses=daily_consec,
        episode_start_balance=s.episode_start_balance, episode_peak_equity=epi_peak,
        episode_realized_pnl=episode_realized, episode_wins=episode_wins, episode_losses=episode_losses,
        episode_trades=episode_trades, episode_consecutive_losses=episode_consec,
        episode_breached=episode_breached, episode_passed=episode_passed,
        position=position, entry=entry, entry_agreed=entry_agreed, entry_alpha_dir=entry_alpha_dir,
        trade_size=trade_size, entry_bar=entry_bar, entry_atr=entry_atr, entry_stop_band=entry_stop_band,
        mfe_atr=mfe_atr, mae_atr=mae_atr, last_close_bar=last_close_bar, last_close_dir=last_close_dir,
        last_exit_px=last_exit_px, entry_band_long=entry_band_long, entry_band_short=entry_band_short,
        entry_reentry=entry_reentry,
        days_elapsed=days_elapsed, daily_pass_streak=daily_pass_streak,
        phase2_active=phase2_active, phase2_peak=phase2_peak, day_locked=day_locked,
        day_progress_hwm=day_progress_hwm, day_had_exposure=day_had_exposure,
        daily_target_frac=s.daily_target_frac, trailing_dd_frac=s.trailing_dd_frac)
    obs = _assemble_obs(ns, static, params, j_new, t_new)
    return ns, obs, reward, terminated, truncated


# Uniform env-interface alias (matches jax_env): init_state, reset_obs, step.
step = step_portfolio
# Uniform builder aliases (so jax_trainer can call env.make_device_static / env.params_from_static).
make_device_static = make_portfolio_device_static
params_from_static = portfolio_params
