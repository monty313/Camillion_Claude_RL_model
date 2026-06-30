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
from src.strategies.alpha_pack import CONVICTION_ALIGN_CAP
from src.observation.hug_pressure import (IDX_DOMINANT_SIDE as _HUG_DOM, IDX_CONTINUATION_3PLUS as _HUG_CONT3,
                                          HUG_EXH_THR, HUG_DECAY_THR, HUG_LOC_THR)
from src.observation.momentum_scores import (IDX_EXHAUSTION as _MOM_EXH, IDX_LOCATION as _MOM_LOC,
                                             IDX_DECAY as _MOM_DEC)
from config.asset_specs import asset_class as _asset_class

# v1.10.0 hugging-pressure reward: absolute obs indices of the hug + momentum fields it reads (block base + idx)
_HUG_BASE = BLOCK_RANGES["hug_pressure"][0]
_HUG_W = BLOCK_RANGES["hug_pressure"][1] - _HUG_BASE      # 15
_MOM_BASE = BLOCK_RANGES["momentum"][0]

# v1.12.0 bracket model: action bounds (from constants) + the 1m high/low indices inside the OHLC obs block.
from config.constants import (TP_MIN_PCT as _TP_MIN, TP_MAX_PCT as _TP_MAX, SL_MIN_PCT as _SL_MIN,
                              SL_MAX_PCT as _SL_MAX, LOT_MIN_MULT as _LOT_MIN, LOT_MAX_MULT as _LOT_MAX,
                              MAX_TRADE_RISK_PCT as _MAX_RISK)
from src.data.aux_features import OHLC_COLUMNS as _OHLC_COLS
_OHLC_HI = BLOCK_RANGES["ohlc"][0] + _OHLC_COLS.index("1m__high")   # intrabar high/low for bracket TP/SL checks
_OHLC_LO = BLOCK_RANGES["ohlc"][0] + _OHLC_COLS.index("1m__low")
# v1.12.0 Stage 4 R:R reward scales + the session-active obs flags (london/ny in the time block)
from config.constants import (RR_BONUS_SCALE as _RR_BONUS, RR_PENALTY_SCALE as _RR_PEN,
                              RR_TAX_SCALE as _RR_TAX, RR_SESSION_PENALTY as _RR_SESS)
_TIME_LON = BLOCK_RANGES["time"][0] + 4
_TIME_NY = BLOCK_RANGES["time"][0] + 5

_HOLD, _BUY, _SELL, _CLOSE = 0, 1, 2, 3
_CONV_CAP = float(CONVICTION_ALIGN_CAP)   # conviction reward saturates at this many aligned signals
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
    conviction_bonus: float      # >=2 of the 3 strong-setup alphas confirmed the entry + won (PnL-capped)
    open_gate: float             # 1.0 -> block a NEW open when the 5m is flat (both CCIs in +/-50)
    hug_pressure_bonus: float    # v1.10.0: per-step bonus for riding a >=3-TF shifted-SMA hug (aligned)
    hug_miss_penalty: float      # v1.10.0: per-step penalty for sitting out a CLEAN hug on an INDEX/METAL
    overtrade_soft_cap: float    # v1.12.0: trades/day before the over-trading penalty kicks in
    overtrade_penalty: float     # v1.12.0: discrete penalty per NEW open once at/over the cap
    bracket_enabled: float       # v1.12.0: 1.0 -> TP/SL/lot heads active (bracket orders); 0.0 -> discrete env


class PortfolioDeviceStatic(NamedTuple):
    static_obs: jnp.ndarray     # (N, T, 499)
    close: jnp.ndarray          # (N, T)
    is_new_day: jnp.ndarray     # (T,)
    ref_move: jnp.ndarray       # (N, T)
    week_avg: jnp.ndarray       # (N, T)
    prev_day: jnp.ndarray       # (N, T)
    prev2_day: jnp.ndarray      # (N, T)
    today_sofar: jnp.ndarray    # (N, T)
    open_gate_blocked: jnp.ndarray  # (N, T) — 1.0 where the 5m is flat (block new opens)
    alpha_matrix: jnp.ndarray   # (N, T, 64)
    occupancy: jnp.ndarray      # (N, 64)
    position_size: jnp.ndarray  # (N,)
    value_per_point: jnp.ndarray# (N,)
    typical_range: jnp.ndarray  # (N,)
    cost_frac: jnp.ndarray      # (N,)
    is_index_metal: jnp.ndarray # (N,) — 1.0 if the symbol is an INDEX or METAL (v1.10.0 hug miss-penalty target)
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
    entry_confirms: jnp.ndarray  # (N,) # of the 3 strong-setup alphas confirming the entry direction
    days_elapsed: jnp.ndarray
    daily_pass_streak: jnp.ndarray
    days_won: jnp.ndarray        # cumulative WON days this episode (v1.8.0 consistency obs)
    phase2_active: jnp.ndarray
    phase2_peak: jnp.ndarray
    day_locked: jnp.ndarray
    day_progress_hwm: jnp.ndarray
    day_had_exposure: jnp.ndarray
    daily_target_reached: jnp.ndarray   # v1.10.0: 1.0 once today hit +2.5% -> mute hug penalty/bonus + obs
    tp_price: jnp.ndarray               # v1.12.0: per-symbol LOCKED take-profit price (0.0 = no bracket)
    sl_price: jnp.ndarray               # v1.12.0: per-symbol LOCKED stop-loss price (0.0 = no bracket)
    daily_target_frac: jnp.ndarray
    trailing_dd_frac: jnp.ndarray


def make_portfolio_device_static(psd) -> PortfolioDeviceStatic:
    j = jnp.asarray
    # v1.10.0: per-symbol INDEX/METAL flag (the hug miss-penalty only targets these). Inferred from the symbol
    # (SPECS override, else roots) -> 1:1 with PortfolioEnv._is_index_metal.
    is_index_metal = jnp.asarray(
        [1.0 if _asset_class(s) in ("index", "metal") else 0.0 for s in psd.symbols], dtype=jnp.float32)
    return PortfolioDeviceStatic(
        static_obs=j(psd.static_obs), close=j(psd.close), is_new_day=j(psd.is_new_day),
        ref_move=j(psd.ref_move), week_avg=j(psd.week_avg), prev_day=j(psd.prev_day),
        prev2_day=j(psd.prev2_day), today_sofar=j(psd.today_sofar),
        open_gate_blocked=j(psd.open_gate_blocked),
        alpha_matrix=j(psd.alpha_matrix), occupancy=j(psd.occupancy),
        position_size=j(psd.position_size), value_per_point=j(psd.value_per_point),
        typical_range=j(psd.typical_range), cost_frac=j(psd.cost_frac), is_index_metal=is_index_metal,
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
                     band_stack_bonus=0.0, reentry_bonus=0.0, conviction_bonus=0.0, open_gate=0.0,
                     hug_pressure_bonus=0.0, hug_miss_penalty=0.0,
                     overtrade_soft_cap=15.0, overtrade_penalty=0.0, bracket_enabled=0.0,
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
        band_stack_bonus=float(band_stack_bonus), reentry_bonus=float(reentry_bonus),
        conviction_bonus=float(conviction_bonus), open_gate=float(open_gate),
        hug_pressure_bonus=float(hug_pressure_bonus), hug_miss_penalty=float(hug_miss_penalty),
        overtrade_soft_cap=float(overtrade_soft_cap), overtrade_penalty=float(overtrade_penalty),
        bracket_enabled=float(bracket_enabled))


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
        entry_band_long=zN, entry_band_short=zN, entry_reentry=zN, entry_confirms=zN,
        days_elapsed=jnp.float32(0.0), daily_pass_streak=jnp.float32(0.0), days_won=jnp.float32(0.0),
        phase2_active=jnp.float32(0.0), phase2_peak=z, day_locked=jnp.float32(0.0),
        # day_progress_hwm is derived from equity (money dtype) each step -> init in the MONEY dtype so the
        # lax.scan training carry stays dtype-stable under x64 too (float32 on TPU is unchanged).
        day_progress_hwm=z, day_had_exposure=jnp.float32(0.0), daily_target_reached=jnp.float32(0.0),
        tp_price=jnp.zeros((N,), bal.dtype), sl_price=jnp.zeros((N,), bal.dtype),
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
    cons = jax_obs_blocks.consistency_features(s.daily_pass_streak, s.days_won, s.days_elapsed)  # v1.8.0
    return {"account_daily": daily, "account_episode": epi, "portfolio": port,
            "sizing": size, "recent_context": ctx, "trade_risk": tr, "consistency": cons}


def _assemble_obs(s: PortfolioState, static: PortfolioDeviceStatic, params: PortfolioParams, j, t):
    obs = static.static_obs[j, t].astype(jnp.float32)
    # v1.10.0: ZERO the hugging-pressure block once today's +2.5% goal is reached (operator: stop observing the
    # agent after the goal so it can't tempt a give-back). 1:1 with PortfolioEnv._obs.
    obs = obs.at[_HUG_BASE:_HUG_BASE + _HUG_W].multiply(1.0 - s.daily_target_reached.astype(jnp.float32))
    blocks = _dynamic_obs(s, static, params, j, t)
    for name in DYNAMIC_BLOCKS:
        a, b = _SL[name]
        obs = obs.at[a:b].set(blocks[name].astype(jnp.float32))
    return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def reset_obs(s: PortfolioState, static: PortfolioDeviceStatic, params: PortfolioParams):
    return _assemble_obs(s, static, params, s.j, s.t)


def step_portfolio(s: PortfolioState, action, tp01, sl01, lot01,
                   static: PortfolioDeviceStatic, params: PortfolioParams):
    """One branchless step (decides symbol j). 1:1 with src/env/portfolio_env.py step(). tp01/sl01/lot01 are
    the continuous heads in [0,1], used only when params.bracket_enabled AND a new BUY/SELL opens."""
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
    # 5m CCI open-gate: block a NEW directional open when the 5m is flat (both CCIs in +/-50). 1:1 with CPU.
    gate = params.open_gate * static.open_gate_blocked[j, t] * ((target != 0.0) & (target != pos_j)).astype(pos_j.dtype)
    target = jnp.where(gate > 0.5, 0.0, target)

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
    # conviction SCALES with aligned-signal count (capped), gated on trading WITH the majority (entry agreed)
    conviction = params.conviction_bonus * (jnp.minimum(s.entry_confirms[j], _CONV_CAP) / _CONV_CAP) * s.entry_agreed[j]
    bonus = alpha_part + params.band_stack_bonus * band_ind + params.reentry_bonus * s.entry_reentry[j] + conviction
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
    # v1.12.0 BRACKET sizing: map [0,1] heads -> bounded TP/SL distances + lot, HARD-CLAMP the lot so open risk
    # (lot x SL price-distance) <= MAX_TRADE_RISK_PCT of CURRENT equity (eq_before = pre-step equity, 1:1 CPU).
    tp_pct = _TP_MIN + jnp.clip(tp01, 0.0, 1.0) * (_TP_MAX - _TP_MIN)
    sl_pct = _SL_MIN + jnp.clip(sl01, 0.0, 1.0) * (_SL_MAX - _SL_MIN)
    lot_mult = _LOT_MIN + jnp.clip(lot01, 0.0, 1.0) * (_LOT_MAX - _LOT_MIN)
    lot_raw = lot_mult * base_j
    max_by_risk = (_MAX_RISK / 100.0 * eq_before) / jnp.maximum(sl_pct * close_jt, 1e-12)
    ts_bracket = jnp.minimum(lot_raw, max_by_risk)
    use_bracket = params.bracket_enabled
    ts_open = jnp.where(use_bracket > 0.5, ts_bracket, ts_risk)     # the size for THIS open
    ts_new = jnp.where(open_mask > 0.5, ts_open, ts_old)           # only an OPEN changes the size
    # locked TP/SL prices on the open leg (bracket only); cleared on a close leg (then maybe re-set by the open)
    set_bracket = (open_mask > 0.5) & (use_bracket > 0.5)
    tp_after_close = jnp.where(close_mask > 0.5, 0.0, s.tp_price[j])
    sl_after_close = jnp.where(close_mask > 0.5, 0.0, s.sl_price[j])
    tp_j = jnp.where(set_bracket, close_jt * (1.0 + target * tp_pct), tp_after_close)
    sl_j = jnp.where(set_bracket, close_jt * (1.0 - target * sl_pct), sl_after_close)
    ecost = cost_j * close_jt * ts_new
    balance = balance - ecost * open_mask
    daily_realized = daily_realized - ecost * open_mask
    episode_realized = episode_realized - ecost * open_mask
    agree, disagree, net_dir = _consensus(static.alpha_matrix[j, t], static.occupancy[j], target)
    against = params.alpha_on * open_mask * (disagree >= 0.5).astype(jnp.float32) * params.alpha_against
    alpha_shaping = alpha_shaping - against
    # OVERTRADING: discrete penalty when this NEW open is at/over today's soft cap (daily_trades already
    # includes this step's close leg, matching CPU). 1:1 with PortfolioEnv.
    over_cap = (daily_trades >= params.overtrade_soft_cap).astype(jnp.float32)
    alpha_shaping = alpha_shaping - params.overtrade_penalty * open_mask * over_cap
    # R:R reward: opening a BRACKET trade OUTSIDE an active session -> penalty (1:1 with CPU). session = london|ny.
    sess = ((static.static_obs[j, t, _TIME_LON] > 0.5) | (static.static_obs[j, t, _TIME_NY] > 0.5)).astype(jnp.float32)
    alpha_shaping = alpha_shaping - _RR_SESS * open_mask * use_bracket * (1.0 - sess)
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
    # SELECTIVITY: count ALL firing alphas pointing the SAME way as this entry (empty/inactive = 0 != target).
    am_jt = static.alpha_matrix[j, t]
    confirms_raw = jnp.sum((am_jt == target).astype(jnp.float32))
    entry_confirms_j = jnp.where(op, confirms_raw, s.entry_confirms[j])

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
    entry_confirms = s.entry_confirms.at[j].set(entry_confirms_j)
    tp_price = s.tp_price.at[j].set(tp_j.astype(s.tp_price.dtype))   # v1.12.0 locked bracket prices
    sl_price = s.sl_price.at[j].set(sl_j.astype(s.sl_price.dtype))

    # --- advance cursor: next symbol; wrap -> advance the bar ---
    j2 = j + 1
    bar_adv = (j2 >= N).astype(jnp.float32)
    j_new = jnp.where(j2 >= N, 0, j2).astype(jnp.int32)
    t_new = jnp.where(j2 >= N, jnp.minimum(t + 1, params.T - 1), t).astype(jnp.int32)

    # --- mark the pot at the (possibly advanced) bar (uses each trade's ACTUAL risk-based size) ---
    close_col = static.close[:, t_new]
    unreal = jnp.sum(position * (close_col - entry) * trade_size)
    equity = balance + unreal
    # --- v1.12.0 BRACKET EXITS (branchless, all symbols): close at the LOCKED level on an intrabar 1m high/low
    # touch (SL checked first; skip the entry bar). Uses the 1m high/low from the OHLC obs block. BEFORE the
    # reward (the realized level PnL is part of this step's reward), then re-mark. 1:1 with _apply_brackets. ---
    bgate = params.bracket_enabled
    bracket_shaping = 0.0                                    # accumulate the R:R reward over the bracket closes
    for sidx in range(N):
        p = position[sidx]; tp = tp_price[sidx]; sl = sl_price[sidx]
        hi = static.static_obs[sidx, t_new, _OHLC_HI]; lo = static.static_obs[sidx, t_new, _OHLC_LO]
        long_sl = (p > 0.0) & (lo <= sl); short_sl = (p < 0.0) & (hi >= sl)
        long_tp = (p > 0.0) & (hi >= tp) & (~long_sl); short_tp = (p < 0.0) & (lo <= tp) & (~short_sl)
        hit_sl = long_sl | short_sl; hit_tp = long_tp | short_tp
        level = jnp.where(hit_sl, sl, jnp.where(hit_tp, tp, 0.0))
        do_b = bgate * ((tp != 0.0) & (p != 0.0)).astype(close_col.dtype) \
            * (t_new > entry_bar[sidx]).astype(close_col.dtype) * (hit_sl | hit_tp).astype(close_col.dtype)
        ts_b = trade_size[sidx]
        r_b = p * (level - entry[sidx]) * ts_b - static.cost_frac[sidx] * level * ts_b
        # R:R REWARD (1:1 with CPU; constants only): rr from the LOCKED levels vs entry, won from realized PnL.
        e_b = entry[sidx]
        tp_pct = jnp.abs(tp / e_b - 1.0); sl_pct = jnp.abs(1.0 - sl / e_b)
        rr = tp_pct / jnp.maximum(sl_pct, 1e-12)
        won_b = (r_b > 0.0).astype(close_col.dtype)
        rr_rew = (won_b * jnp.log1p(rr) * _RR_BONUS
                  - (1.0 - won_b) * (rr < 1.0).astype(close_col.dtype) * (1.0 - rr) * _RR_PEN
                  - (rr < 0.5).astype(close_col.dtype) * (0.5 - rr) / 0.5 * _RR_TAX)
        bracket_shaping = bracket_shaping + do_b * rr_rew
        balance = balance + r_b * do_b
        daily_realized = daily_realized + r_b * do_b
        episode_realized = episode_realized + r_b * do_b
        w = ((r_b > 0.0) & (do_b > 0.5)).astype(jnp.float32)
        l = ((r_b <= 0.0) & (do_b > 0.5)).astype(jnp.float32)
        daily_trades = daily_trades + do_b.astype(jnp.float32)
        episode_trades = episode_trades + do_b.astype(jnp.float32)
        daily_wins = daily_wins + w; episode_wins = episode_wins + w
        daily_losses = daily_losses + l; episode_losses = episode_losses + l
        daily_consec = jnp.where(do_b > 0.5, jnp.where(w > 0.5, 0.0, daily_consec + 1.0), daily_consec)
        episode_consec = jnp.where(do_b > 0.5, jnp.where(w > 0.5, 0.0, episode_consec + 1.0), episode_consec)
        last_close_bar = last_close_bar.at[sidx].set(jnp.where(do_b > 0.5, t_new, last_close_bar[sidx]))
        last_close_dir = last_close_dir.at[sidx].set(jnp.where(do_b > 0.5, p, last_close_dir[sidx]))
        last_exit_px = last_exit_px.at[sidx].set(jnp.where(do_b > 0.5, level, last_exit_px[sidx]))
        position = position.at[sidx].set(jnp.where(do_b > 0.5, jnp.zeros_like(p), p))
        tp_price = tp_price.at[sidx].set(jnp.where(do_b > 0.5, jnp.zeros_like(tp), tp))
        sl_price = sl_price.at[sidx].set(jnp.where(do_b > 0.5, jnp.zeros_like(sl), sl))
    unreal = jnp.sum(position * (close_col - entry) * trade_size)   # re-mark after the bracket exits
    equity = balance + unreal
    day_peak = jnp.maximum(day_peak, equity)
    epi_peak = jnp.maximum(epi_peak, equity)
    # v1.7.0: update each open trade's max favorable / adverse excursion (ATR units) at the new bar
    in_pos = (position != 0.0) & (entry_atr > 0.0)
    exc = jnp.where(in_pos, position * (close_col - entry) / jnp.maximum(entry_atr, 1e-9), 0.0)
    mfe_atr = jnp.where(in_pos, jnp.maximum(mfe_atr, exc), mfe_atr)
    mae_atr = jnp.where(in_pos, jnp.maximum(mae_atr, -exc), mae_atr)

    reward = ((equity - eq_before) / start) * params.reward_scale + alpha_shaping + bracket_shaping

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
    # DAILY GOAL REACHED (operator 2026-06-30): once today hits +2.5%, latch a flag -> mutes the hug agent
    # (reward below + obs in _assemble_obs). Reset at midnight. 1:1 with PortfolioEnv._daily_target_reached.
    target_amt_h = s.daily_target_frac * start
    daily_target_reached = jnp.maximum(
        s.daily_target_reached, (equity - s.day_start_balance >= target_amt_h).astype(jnp.float32))
    # HUGGING-PRESSURE (v1.10.0; operator's heavy momentum agent): ride a >=3-TF shifted-SMA hug (current
    # symbol j); penalise sitting out a CLEAN one on an INDEX/METAL. "Clean" = ALL 3 TFs agree AND momentum is
    # NOT exhausted / extended-in-direction / decaying. MUTED once today's goal is reached. 1:1 with CPU.
    o_jt = static.static_obs[j, t]
    dom = o_jt[_HUG_BASE + _HUG_DOM]; cont3 = o_jt[_HUG_BASE + _HUG_CONT3]; loc = o_jt[_MOM_BASE + _MOM_LOC]
    conflict = ((o_jt[_MOM_BASE + _MOM_EXH] > HUG_EXH_THR) | (o_jt[_MOM_BASE + _MOM_DEC] > HUG_DECAY_THR)
                | ((jnp.abs(loc) > HUG_LOC_THR) & ((loc > 0) == (dom > 0)) & (dom != 0.0)))
    clean = ((cont3 > 0.5) & (dom != 0.0) & (~conflict)).astype(reward.dtype)
    hug_active = clean * (1.0 - daily_target_reached.astype(reward.dtype))   # muted after the daily goal
    aligned = (position[j] == dom).astype(reward.dtype)
    is_im = (static.is_index_metal[j] > 0.5).astype(reward.dtype)
    reward = reward + params.hug_pressure_bonus * hug_active * aligned                       # RIDE the hug
    reward = reward - params.hug_miss_penalty * hug_active * (1.0 - aligned) * is_im          # sat out index/metal
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
    days_won = s.days_won + new_day * won                      # v1.8.0: cumulative won days this episode

    def _rst(val, cur):
        return jnp.where(new_day > 0.5, val, cur)
    day_progress_hwm = _rst(0.0, day_progress_hwm)            # new day -> reset the seek high-water mark
    day_had_exposure = _rst(exposed_now, day_had_exposure)    # new day -> re-seed from carried position
    daily_target_reached = _rst(jnp.float32(0.0), daily_target_reached)   # new day -> un-mute the hug agent
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
        entry_reentry=entry_reentry, entry_confirms=entry_confirms,
        days_elapsed=days_elapsed, daily_pass_streak=daily_pass_streak, days_won=days_won,
        phase2_active=phase2_active, phase2_peak=phase2_peak, day_locked=day_locked,
        day_progress_hwm=day_progress_hwm, day_had_exposure=day_had_exposure,
        daily_target_reached=daily_target_reached, tp_price=tp_price, sl_price=sl_price,
        daily_target_frac=s.daily_target_frac, trailing_dd_frac=s.trailing_dd_frac)
    obs = _assemble_obs(ns, static, params, j_new, t_new)
    return ns, obs, reward, terminated, truncated


# Uniform env-interface alias (matches jax_env): init_state, reset_obs, step.
step = step_portfolio
# Uniform builder aliases (so jax_trainer can call env.make_device_static / env.params_from_static).
make_device_static = make_portfolio_device_static
params_from_static = portfolio_params
