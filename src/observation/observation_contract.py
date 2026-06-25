# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  THE single, authoritative list of all 357 observation feature names,
#      their block boundaries, and a validator. This file IS the contract.
# WHERE src/observation/observation_contract.py
# HOW  Concatenate per-block name lists in OBS_BLOCK_ORDER; assert total==357.
#      validate() checks shape, dtype and finiteness of a built observation.
# DEPENDS_ON: config/constants.py, src/indicators/base.py, numpy
# USED_BY: src/observation/builder.py, src/barbershop/feature_doctor.py,
#          tests/test_observation_contract.py
# CHANGE_NOTES(IRAC): I: every consumer must agree on feature order/size. R:
#   spec fixed observation shape + Monty hybrid blocks. A: one ordered name
#   list + offsets + validator. C: a frozen contract = model compatibility
#   across phases and safe ablation by block.
# =====================================================================
"""The observation contract: 357 ordered feature names, offsets, validator."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.indicators.base import ALL_INDICATOR_COLUMNS

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
    }


BLOCK_NAMES = _block_names()

# Full ordered feature-name list (length 357) + block offset slices.
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
