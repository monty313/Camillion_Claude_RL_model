# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  THE single, authoritative list of all OBS_TOTAL_SIZE observation feature
#      names (517 @ v1.8.0), their block boundaries, and a validator. This IS the contract.
# WHERE src/observation/observation_contract.py
# HOW  Concatenate per-block name lists in OBS_BLOCK_ORDER; assert total==OBS_TOTAL_SIZE.
#      validate() checks shape, dtype and finiteness of a built observation.
# DEPENDS_ON: config/constants.py, src/indicators/base.py, numpy
# USED_BY: src/observation/builder.py, src/barbershop/feature_doctor.py,
#          tests/test_observation_contract.py
# CHANGE_NOTES(IRAC): I: every consumer must agree on feature order/size. R:
#   spec fixed observation shape + Monty hybrid blocks. A: one ordered name
#   list + offsets + validator. C: a frozen contract = model compatibility
#   across phases and safe ablation by block.
# =====================================================================
"""The observation contract: 517 ordered feature names (v1.8.0), offsets, validator."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.indicators.base import ALL_INDICATOR_COLUMNS
from src.data.aux_features import OHLC_COLUMNS  # v1.6.0 raw OHLC obs block names (20)

ACCOUNT_DAILY_NAMES = (
    "daily_win_rate_pct", "daily_realized_pnl_pct", "daily_drawdown_used_pct",
    "daily_target_progress_pct", "daily_risk_remaining_pct",
    "daily_trades_count_pct", "daily_consecutive_losses_pct",
)
ACCOUNT_EPISODE_NAMES = (
    "episode_win_rate_pct", "episode_realized_pnl_pct", "episode_drawdown_used_pct",
    "episode_target_progress_pct", "episode_pass_progress_pct",
    "episode_risk_remaining_pct", "episode_consecutive_losses_pct",
)
TIME_NAMES = ("tod_sin", "tod_cos", "dow_sin", "dow_cos",
              "session_london", "session_newyork")
PORTFOLIO_NAMES = (
    "open_positions_pct", "net_exposure_signed", "gross_exposure_pct",
    "unrealized_pnl_pct", "avg_position_age_pct", "largest_position_dir",
    "equity_ratio", "balance_ratio",
)
# v1.3.0 SIZING block: 6 what-if lot rungs + 4 context, all as fractions of INITIAL balance.
SIZING_NAMES = (
    "size_move_pct_lot0_01", "size_move_pct_lot0_1", "size_move_pct_lot0_5",
    "size_move_pct_lot1", "size_move_pct_lot2", "size_move_pct_lot4",
    "daily_target_remaining_pct", "dd_room_pct", "active_lots_norm", "active_move_value_pct",
)
# v1.4.0 CROSS-ASSET block: asset-class one-hot (contract order) + ATR-normalized movement +
# sessions. ATR-normalized parts are scale-free -> comparable across the whole FTMO universe.
CROSS_ASSET_NAMES = tuple(f"class_{c}" for c in C.ASSET_CLASSES) + (
    "move_in_atr", "atr_pct_price", "atr_regime", "session_asian", "session_london_ny_overlap",
)
# v1.5.0 RECENT-CONTEXT block: recent daily movement RELATIVE to the symbol's own average +
# a time-aware "am I on pace to pass" read.
RECENT_CONTEXT_NAMES = (
    "week_avg_range_vs_typical", "prev_day_range_vs_week", "prev2_day_range_vs_week",
    "today_range_so_far_vs_week", "days_elapsed_norm", "episode_return_so_far",
    "pace_vs_2_5pct_plan", "challenge_target_remaining",
)
# v1.7.0 TRADE-RISK block: live risk state of the CURRENT symbol's open trade (manage + re-enter).
TRADE_RISK_NAMES = (
    "tr_in_trade", "tr_direction", "tr_unrealized_pnl_atr", "tr_unrealized_pnl_pct",
    "tr_dist_to_soft_stop_2atr", "tr_dist_to_hard_stop_bb", "tr_bars_held_norm",
    "tr_max_favorable_atr", "tr_max_adverse_atr", "tr_bars_since_last_close",
    "tr_last_trade_dir", "tr_price_vs_last_exit_atr", "tr_band_stack_long", "tr_band_stack_short",
)
# v1.8.0 CONSISTENCY block: the bot's multi-day FTMO standing (value/protect the won-day streak).
CONSISTENCY_NAMES = (
    "won_day_streak_norm", "days_won_norm", "won_day_rate", "days_into_journey_norm",
)


def _block_names() -> dict[str, list[str]]:
    return {
        "indicators": list(ALL_INDICATOR_COLUMNS),
        "alpha_values": [f"alpha_{i:02d}" for i in range(C.MAX_STRATEGIES)],
        "alpha_mask": [f"alpha_mask_{i:02d}" for i in range(C.MAX_STRATEGIES)],
        "alpha_summary": ["buy_pct", "sell_pct", "active_pct", "net_signal_pct"],
        "signal_memory": [f"signal_balance_lag_{k}" for k in range(C.SIGNAL_MEMORY_LAGS)],
        "signal_accuracy": ["signal_accuracy_1bar_pct", "signal_accuracy_3bar_pct"],
        "account_daily": list(ACCOUNT_DAILY_NAMES),
        "account_episode": list(ACCOUNT_EPISODE_NAMES),
        "time": list(TIME_NAMES),
        "portfolio": list(PORTFOLIO_NAMES),
        "alpha_streak": [f"alpha_streak_{i:02d}" for i in range(C.MAX_STRATEGIES)],
        "sizing": list(SIZING_NAMES),
        "cross_asset": list(CROSS_ASSET_NAMES),
        "recent_context": list(RECENT_CONTEXT_NAMES),
        "ohlc": list(OHLC_COLUMNS),   # v1.6.0: raw O/H/L/C per timeframe (20)
        "trade_risk": list(TRADE_RISK_NAMES),   # v1.7.0: current symbol's open-trade risk state (14)
        "consistency": list(CONSISTENCY_NAMES),  # v1.8.0: multi-day FTMO standing / won-day streak (4)
    }


BLOCK_NAMES = _block_names()

# Full ordered feature-name list (length OBS_TOTAL_SIZE (517)) + block offset slices.
FEATURE_NAMES: list[str] = []
BLOCK_SLICES: dict[str, slice] = {}
_cursor = 0
for _name, _size in C.OBS_BLOCK_ORDER:
    _names = BLOCK_NAMES[_name]
    assert len(_names) == _size, f"block {_name}: {len(_names)} names != size {_size}"
    BLOCK_SLICES[_name] = slice(_cursor, _cursor + _size)
    FEATURE_NAMES.extend(_names)
    _cursor += _size

assert _cursor == C.OBS_TOTAL_SIZE, (_cursor, C.OBS_TOTAL_SIZE)
assert len(FEATURE_NAMES) == C.OBS_TOTAL_SIZE


def validate(obs: np.ndarray, *, allow_nan: bool = False) -> np.ndarray:
    """Assert obs matches the contract (shape, dtype). Returns it unchanged."""
    obs = np.asarray(obs)
    if obs.shape != C.OBS_SHAPE:
        raise ValueError(f"observation shape {obs.shape} != contract {C.OBS_SHAPE}")
    if obs.dtype != np.float32:
        raise ValueError(f"observation dtype {obs.dtype} != float32")
    if not allow_nan and not np.all(np.isfinite(obs)):
        bad = np.where(~np.isfinite(obs))[0][:5]
        raise ValueError(f"observation has non-finite values at indices {bad.tolist()}")
    return obs
