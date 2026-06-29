# =====================================================================
# WHEN 2026-06-29 (Phase 2, contract v1.7.0) | WHO Claude for Monty
# WHY  The 14-float TRADE-RISK observation block (v1.7.0): the live RISK state of the
#      CURRENT symbol's OPEN trade, so the policy can MANAGE the trade (hold vs close)
#      and learn to RE-ENTER a winner. ONE shared builder used by BOTH CPU envs
#      (TradingEnv single-symbol + PortfolioEnv shared-pot); jax_tpu/jax_obs_blocks.py
#      holds a byte-for-byte jnp twin so the TPU env emits the IDENTICAL 14 floats.
# WHERE src/observation/trade_risk.py
# HOW   Pure numpy scalar math (no pandas/TA-Lib -> safe to call from the hot step()).
#       The env gathers the open-trade state + the precomputed band/ATR values for the
#       current bar and calls build(); this file owns the field ORDER + the normalizers.
# DEPENDS_ON: numpy, config.constants (block size), config.observation_contract (names)
# USED_BY: src/env/trading_env.py, src/env/portfolio_env.py, jax_tpu/jax_obs_blocks.py (twin),
#          tests/test_trade_risk_block.py
# CHANGE_NOTES(IRAC): I: the bot traded with NO awareness of its open-trade risk, no hard
#   stop, no re-entry sense. R: operator 2026-06-29 (trade-risk obs + BB hard stop + risk
#   sizing + re-entry + band-stack bonus). A: a dynamic 14-float block (append-only contract
#   bump v1.6.0->v1.7.0) describing the open trade in ATR / account / band-distance units +
#   a re-entry context + the band-stack flags. C: the policy can now see and manage trade
#   risk, time exits, and re-enter trends -> tighter, more consistent FTMO days.
# =====================================================================
"""The v1.7.0 TRADE-RISK observation block (14 floats) — shared by the CPU envs (jnp twin in jax_tpu)."""
from __future__ import annotations
import numpy as np
from config import constants as C

# --- normalizers (these literals MUST match jax_tpu/jax_obs_blocks.trade_risk_features) ---
ATR_PNL_SCALE: float = 5.0        # P&L (in ATR units) is divided by this before the [-1,1] clip (5 ATR -> 1.0)
SOFT_STOP_ATR: float = 2.0        # the SOFT stop sits 2x ATR(14) adverse from entry (distance fraction 0->1)
MFE_SCALE: float = 5.0            # max-favorable-excursion (ATR units) normalizer (5 ATR -> 1.0)
MAE_SCALE: float = 2.0            # max-adverse-excursion normalizer (relative to the 2-ATR soft stop)
BARS_HELD_NORM: float = 480.0     # bars-held normalizer (~a third of an M1 trading day -> 1.0)
BARS_SINCE_NORM: float = 480.0    # bars-since-last-close normalizer (re-entry recency)
_EPS: float = 1e-9

N_FIELDS: int = C.OBS_BLOCK_TRADE_RISK   # 14


def band_stack_long(price, bb200_1m_up, bb10_1m_up, bb200_5m_up, bb10_5m_up) -> bool:
    """BULLISH band stack: price ABOVE BB200(dev1) AND BB10(dev1) on BOTH 1m and 5m (NaN band -> False)."""
    return bool((price > bb200_1m_up) and (price > bb10_1m_up)
                and (price > bb200_5m_up) and (price > bb10_5m_up))


def band_stack_short(price, bb200_1m_lo, bb10_1m_lo, bb200_5m_lo, bb10_5m_lo) -> bool:
    """BEARISH band stack: price BELOW BB200(dev1) AND BB10(dev1) on BOTH 1m and 5m (NaN band -> False)."""
    return bool((price < bb200_1m_lo) and (price < bb10_1m_lo)
                and (price < bb200_5m_lo) and (price < bb10_5m_lo))


def build(*, pos, entry_px, price, trade_size, equity, entry_atr, atr_now,
          entry_stop_band, bars_held, mfe_atr, mae_atr,
          bars_since_close, last_dir, last_exit_px,
          bb200_1m_up, bb200_1m_lo, bb200_5m_up, bb200_5m_lo,
          bb10_1m_up, bb10_1m_lo, bb10_5m_up, bb10_5m_lo) -> np.ndarray:
    """The 14-float trade-risk block for ONE symbol's open trade. All scalars; returns (14,) float32.

    Field ORDER (== observation_contract.TRADE_RISK_NAMES):
      0 in_trade            1 direction               2 unrealized_pnl_atr   3 unrealized_pnl_pct
      4 dist_to_soft_2atr   5 dist_to_hard_bb         6 bars_held_norm       7 max_favorable_atr
      8 max_adverse_atr     9 bars_since_last_close   10 last_trade_dir      11 price_vs_last_exit_atr
      12 band_stack_long    13 band_stack_short
    """
    pos = float(pos)
    in_trade = 1.0 if pos != 0.0 else 0.0
    flat = 1.0 - in_trade
    move = float(price) - float(entry_px)
    signed_move = pos * move                                   # >0 = favorable
    eatr = max(float(entry_atr), _EPS)
    aatr = max(float(atr_now), _EPS)

    pnl_atr = _clip(signed_move / eatr / ATR_PNL_SCALE, -1.0, 1.0) * in_trade
    pnl_pct = _clip(signed_move * float(trade_size) / max(float(equity), _EPS), -1.0, 1.0) * in_trade
    adverse_atr = max(0.0, -signed_move) / eatr
    dist_soft = _clip(adverse_atr / SOFT_STOP_ATR, 0.0, 1.0) * in_trade

    # distance to the 1m BB(10,1) opposite-band HARD stop, as a fraction of the room there was at entry
    # (0 = as far from the stop as at entry, 1 = at / through the band).
    stop_band_now = bb10_1m_lo if pos > 0 else bb10_1m_up
    room_now = pos * (float(price) - float(stop_band_now))
    room_entry = pos * (float(entry_px) - float(entry_stop_band))
    valid_band = 1.0 if (np.isfinite(room_entry) and room_entry > _EPS) else 0.0
    dist_hard = _clip(1.0 - room_now / max(room_entry, _EPS), 0.0, 1.0) * in_trade * valid_band

    bars_held_norm = _clip(float(bars_held) / BARS_HELD_NORM, 0.0, 1.0) * in_trade
    mfe_norm = _clip(float(mfe_atr) / MFE_SCALE, 0.0, 1.0) * in_trade
    mae_norm = _clip(float(mae_atr) / MAE_SCALE, 0.0, 1.0) * in_trade

    bars_since = _clip(float(bars_since_close) / BARS_SINCE_NORM, 0.0, 1.0) * flat
    last_trade_dir = _clip(float(last_dir), -1.0, 1.0) * flat
    price_vs_exit = _clip(float(last_dir) * (float(price) - float(last_exit_px)) / aatr / ATR_PNL_SCALE,
                          -1.0, 1.0) * flat

    bsl = 1.0 if band_stack_long(price, bb200_1m_up, bb10_1m_up, bb200_5m_up, bb10_5m_up) else 0.0
    bss = 1.0 if band_stack_short(price, bb200_1m_lo, bb10_1m_lo, bb200_5m_lo, bb10_5m_lo) else 0.0

    return np.array([
        in_trade, _clip(pos, -1.0, 1.0), pnl_atr, pnl_pct,
        dist_soft, dist_hard, bars_held_norm, mfe_norm,
        mae_norm, bars_since, last_trade_dir, price_vs_exit,
        bsl, bss,
    ], dtype=np.float32)


def _clip(x, lo, hi):
    return float(np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi))


def empty() -> np.ndarray:
    """All-zero block (no open trade, no history) — the reset / no-data value."""
    return np.zeros(N_FIELDS, dtype=np.float32)
