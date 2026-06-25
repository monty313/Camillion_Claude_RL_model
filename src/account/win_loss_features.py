# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Build the daily(7) + episode(7) + portfolio(8) feature blocks AS
#      FRACTIONS OF THE ACTIVE CONFIG LIMITS, read at call time -- so changing
#      target / trailing-DD / toggle needs NO RETRAIN (the fractions still mean
#      the same thing to the policy).
# WHERE src/account/win_loss_features.py
# HOW  Pull the active FTMO/FREE config, divide pnl/drawdown by its (editable)
#      limits, clip to safe ranges. Pure functions; CONTEXT features only --
#      the bot is never rewarded from these.
# DEPENDS_ON: config/{variables,ftmo_config}.py, src/account/account_state.py, numpy
# USED_BY: src/observation/builder.py, tests/test_observation_contract.py
# CHANGE_NOTES(IRAC): I: features must survive live risk retunes. R: Monty
#   "change target/trailing without retrain" + spec percentage features. A:
#   fractions of active-config limits + clipping. C: one model reused across
#   many challenge configs.
# =====================================================================
"""Daily / episode / portfolio feature blocks as fractions of active limits."""
from __future__ import annotations
import numpy as np
from config import variables as V
from config import constants as C
from config.ftmo_config import load_active_config
from src.account.account_state import AccountState


def _pct(cfg, name: str, default: float) -> float:
    return float(getattr(cfg, name, default)) / 100.0


def daily_features(acc: AccountState, cfg=None) -> np.ndarray:
    """7 daily features (fractions of editable daily limits)."""
    cfg = cfg or load_active_config()
    bal0 = acc.day_start_balance or acc.starting_balance
    win_rate = acc.daily_win_rate
    pnl_frac = (acc.daily_realized_pnl / bal0) if bal0 else 0.0
    target_frac = _pct(cfg, "daily_target_pct", 2.5)
    target_progress = (pnl_frac / target_frac) if target_frac > 0 else 0.0
    dd_limit = _pct(cfg, "daily_drawdown_pct",
                    getattr(cfg, "max_daily_drawdown_pct", 5.0))
    daily_loss_frac = max(0.0, -pnl_frac)
    dd_used = (daily_loss_frac / dd_limit) if dd_limit > 0 else 0.0
    risk_remaining = 1.0 - dd_used
    trades_pct = acc.daily_trades / max(1, V.MAX_DAILY_TRADES)
    streak_pct = acc.daily_consecutive_losses / max(1, V.MAX_CONSECUTIVE_LOSSES)
    return np.array([
        np.clip(win_rate, 0, 1),
        np.clip(pnl_frac, -1, 1),
        np.clip(dd_used, 0, 2),
        np.clip(target_progress, -2, 2),
        np.clip(risk_remaining, 0, 1),
        np.clip(trades_pct, 0, 2),
        np.clip(streak_pct, 0, 2),
    ], dtype=np.float32)


def episode_features(acc: AccountState, cfg=None) -> np.ndarray:
    """7 episode features (fractions of editable episode limits)."""
    cfg = cfg or load_active_config()
    bal0 = acc.episode_start_balance or acc.starting_balance
    win_rate = acc.episode_win_rate
    pnl_frac = (acc.episode_realized_pnl / bal0) if bal0 else 0.0
    # episode drawdown vs the binding wall (trailing if enabled, else total)
    trailing_on = bool(getattr(cfg, "trailing_enabled", False))
    wall = _pct(cfg, "trailing_drawdown_pct", 4.0) if trailing_on \
        else _pct(cfg, "max_total_drawdown_pct", 10.0)
    peak = acc.episode_peak_equity or bal0
    dd_frac = max(0.0, (peak - acc.equity) / peak) if peak else 0.0
    dd_used = (dd_frac / wall) if wall > 0 else 0.0
    risk_remaining = 1.0 - dd_used
    profit_target = _pct(cfg, "profit_target_total_pct",
                         getattr(cfg, "daily_target_pct", 2.5))
    target_progress = (pnl_frac / profit_target) if profit_target > 0 else 0.0
    # pass progress: profit progress, zeroed if breached (Phase 1 adds two-phase)
    pass_progress = 0.0 if acc.episode_breached else np.clip(target_progress, 0, 1)
    streak_pct = acc.episode_consecutive_losses / max(1, V.MAX_CONSECUTIVE_LOSSES)
    return np.array([
        np.clip(win_rate, 0, 1),
        np.clip(pnl_frac, -1, 1),
        np.clip(dd_used, 0, 2),
        np.clip(target_progress, -2, 2),
        np.clip(pass_progress, 0, 1),
        np.clip(risk_remaining, 0, 1),
        np.clip(streak_pct, 0, 2),
    ], dtype=np.float32)


def portfolio_features(acc: AccountState) -> np.ndarray:
    """8 portfolio/open-trade features so the bot can manage exposure."""
    bal0 = acc.starting_balance or 1.0
    open_pct = acc.open_positions / max(1, V.MAX_OPEN_POSITIONS)
    equity_ratio = acc.equity / bal0 if bal0 else 1.0
    balance_ratio = acc.balance / bal0 if bal0 else 1.0
    unreal_frac = acc.unrealized_pnl / bal0 if bal0 else 0.0
    age_pct = acc.avg_position_age_bars / 1000.0  # vs a nominal 1000-bar hold
    return np.array([
        np.clip(open_pct, 0, 1),
        np.clip(acc.net_exposure, -1, 1),
        np.clip(acc.gross_exposure, 0, 1),
        np.clip(unreal_frac, -1, 1),
        np.clip(age_pct, 0, 1),
        float(np.clip(acc.largest_position_dir, -1, 1)),
        np.clip(equity_ratio, 0, 5),
        np.clip(balance_ratio, 0, 5),
    ], dtype=np.float32)


def sizing_features(acc: AccountState, cfg=None, *, value_per_point: float,
                    ref_move: float, position_size: float) -> np.ndarray:
    """v1.3.0 SIZING block (10 floats), ALL as fractions of the INITIAL balance.

    OBSERVATION ONLY -- sizing is not an action yet. Shows the per-asset $ conversion,
    how much is still needed today, drawdown room, and a 0.01..4-lot what-if ladder so the
    policy learns the size<->risk/reward relationship before it can choose size.
      value_per_point: account $ per 1.0 PRICE move per 1 lot (asset contract size).
      ref_move:        a recent typical PRICE move (leak-free, from the cache).
      position_size:   the env's ACTIVE size (= value_per_point * active_lots).
    """
    cfg = cfg or load_active_config()
    init = acc.starting_balance or 1.0
    one_lot_move = float(value_per_point) * float(ref_move)          # $ a typical move = at 1 lot
    ladder = [np.clip(one_lot_move * L / init, 0.0, 1.0) for L in C.SIZING_LOTS_LADDER]
    # how much of the +target% is still needed today (the day's gain on EQUITY vs INITIAL)
    day0 = acc.day_start_balance if acc.day_start_balance is not None else init
    day_gain_frac = (acc.equity - day0) / init if init else 0.0
    target_frac = _pct(cfg, "daily_target_pct", 2.5)
    target_remaining = np.clip(target_frac - day_gain_frac, 0.0, 1.0)
    # room before the binding wall (trailing if on, else total)
    wall = _pct(cfg, "trailing_drawdown_pct", 4.0) if bool(getattr(cfg, "trailing_enabled", False)) \
        else _pct(cfg, "max_total_drawdown_pct", 10.0)
    peak = acc.episode_peak_equity or init
    dd_frac = max(0.0, (peak - acc.equity) / peak) if peak else 0.0
    dd_room = np.clip(wall - dd_frac, 0.0, 1.0)
    active_lots = (float(position_size) / float(value_per_point)) if value_per_point else 0.0
    active_norm = np.clip(active_lots / max(C.SIZING_LOTS_LADDER), 0.0, 1.0)
    active_move = np.clip(one_lot_move * active_lots / init, 0.0, 1.0)
    return np.array([*ladder, target_remaining, dd_room, active_norm, active_move], dtype=np.float32)
