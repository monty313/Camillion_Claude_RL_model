# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Assemble the final (513,) float32 observation by concatenating every
#      block in the contract order. Single place that builds what the policy sees.
# WHERE src/observation/builder.py
# HOW  build_from_blocks() fills missing blocks with zeros, sanitises any
#      non-finite values to 0.0 (Phase-0 indicator stubs are NaN), validates
#      against the contract. build() is a convenience that derives the alpha
#      summary and the account/portfolio/time blocks for you.
# DEPENDS_ON: config/constants.py, src/observation/observation_contract.py,
#             src/signals/signal_summary.py, src/account/win_loss_features.py, numpy
# USED_BY: src/env/trading_env.py (Phase 1), src/barbershop/feature_doctor.py,
#          tests/test_observation_contract.py
# CHANGE_NOTES(IRAC): I: one safe place to build the obs; never emit NaN. R:
#   spec fixed shape + float32 + Monty hybrid blocks. A: ordered concat +
#   nan_to_num + validate. C: a finite, contract-shaped vector every step.
# =====================================================================
"""ObservationBuilder: assemble + sanitise + validate the 513-float32 vector."""
from __future__ import annotations
import math
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC
from src.signals.signal_summary import summarize
from src.account import win_loss_features as WL


def time_features(timestamp=None) -> np.ndarray:
    """6 time/session features. Accepts a datetime (UTC) or None (-> zeros)."""
    if timestamp is None:
        return np.zeros(C.OBS_BLOCK_TIME, dtype=np.float32)
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    tod = 2.0 * math.pi * minute_of_day / 1440.0
    dow_ang = 2.0 * math.pi * timestamp.weekday() / 7.0
    hour = timestamp.hour + timestamp.minute / 60.0
    london = 1.0 if 7.0 <= hour < 16.0 else 0.0     # ~London session (UTC)
    newyork = 1.0 if 12.0 <= hour < 21.0 else 0.0   # ~New York session (UTC)
    return np.array([math.sin(tod), math.cos(tod), math.sin(dow_ang),
                     math.cos(dow_ang), london, newyork], dtype=np.float32)


def build_from_blocks(blocks: dict) -> np.ndarray:
    """Concatenate blocks in contract order; missing -> zeros; NaN/inf -> 0.0."""
    parts: list[np.ndarray] = []
    for name, size in C.OBS_BLOCK_ORDER:
        b = blocks.get(name)
        b = np.zeros(size, dtype=np.float32) if b is None \
            else np.asarray(b, dtype=np.float32).ravel()
        if b.shape[0] != size:
            raise ValueError(f"block '{name}' size {b.shape[0]} != contract {size}")
        parts.append(b)
    obs = np.concatenate(parts).astype(np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # never emit non-finite
    return OC.validate(obs)


def build(*, indicators=None, alpha_values=None, occupancy_mask=None,
          alpha_summary=None, signal_memory=None, signal_accuracy=None,
          account=None, account_daily=None, account_episode=None,
          portfolio=None, timestamp=None) -> np.ndarray:
    """Convenience builder. Derives alpha summary + account blocks if needed."""
    blocks: dict = {}
    if indicators is not None:
        blocks["indicators"] = indicators
    if alpha_values is not None:
        blocks["alpha_values"] = alpha_values
    if occupancy_mask is not None:
        blocks["alpha_mask"] = occupancy_mask
    if alpha_summary is None and alpha_values is not None and occupancy_mask is not None:
        alpha_summary = summarize(alpha_values, occupancy_mask)
    if alpha_summary is not None:
        blocks["alpha_summary"] = alpha_summary
    if signal_memory is not None:
        blocks["signal_memory"] = signal_memory
    if signal_accuracy is not None:
        blocks["signal_accuracy"] = signal_accuracy
    if account is not None:
        blocks.setdefault("account_daily", WL.daily_features(account))
        blocks.setdefault("account_episode", WL.episode_features(account))
        blocks.setdefault("portfolio", WL.portfolio_features(account))
    if account_daily is not None:
        blocks["account_daily"] = account_daily
    if account_episode is not None:
        blocks["account_episode"] = account_episode
    if portfolio is not None:
        blocks["portfolio"] = portfolio
    blocks["time"] = time_features(timestamp)
    return build_from_blocks(blocks)


def zeros() -> np.ndarray:
    """A valid all-zero observation (shape/dtype correct)."""
    return build_from_blocks({})
