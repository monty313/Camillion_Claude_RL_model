# =====================================================================
# WHEN 2026-07-01 (Phase 2, contract v1.13.0) | WHO Claude for Monty
# WHY  The operator's momentum-aware EXIT discipline. On the 1m chart a Bollinger(20, dev=0.5) band is applied
#      to the HIGH series (the BUY band) and to the LOW series (the SELL band). A good exit banks INSIDE the
#      band (momentum paused, price pulled back to its own recent mean); closing OUTSIDE a rail is punished:
#      below the far rail = cutting a winner while momentum still runs, above the near rail = clinging after it
#      reversed. This file owns (a) the 4 leak-free raw rails, (b) the 4 STATIC obs "rooms" (close-vs-rail,
#      so the policy SEES the trigger), (c) the DYNAMIC 2-float bracket-state block (TP/SL as observations),
#      and (d) the shared exit_outside_band() test the reward penalty uses in BOTH engines.
# WHERE src/observation/exit_band.py
# HOW  PURE numpy. Rails = bollinger(high|low, 20, 0.5) -> causal rolling window (LEAK-FREE); precompute-only.
#      The STATIC rooms are lifted byte-identical into the JAX static obs tensor (auto parity); the DYNAMIC
#      bracket block + the outside-band penalty have jnp twins in jax_tpu/jax_obs_blocks.py + jax_portfolio_env.
# DEPENDS_ON: numpy, src.indicators.bollinger, src.data.aux_features (1m high/low columns)
# USED_BY: src/env/trading_env.py (_precompute, _obs), src/env/portfolio_env.py (obs + exit-band penalty),
#          jax_tpu/jax_static_features.py, jax_tpu/jax_obs_blocks.py, observation_contract, tests
# CHANGE_NOTES(IRAC): I: the bot had no learned EXIT discipline -- it banked into the run or clung past the
#   reversal. R: operator 2026-07-01 -- BB(20,0.5) on 1m High (buys) / Low (sells) as the exit reference +
#   a penalty for closing outside it. A: a STATIC 4-float band block + a DYNAMIC 2-float bracket block
#   (append-only v1.12.0->v1.13.0) + a removable reward penalty. C: the policy learns to exit into the pause
#   (bank inside the band, let winners run, don't cling) -> tighter, more consistent FTMO days.
# =====================================================================
"""The v1.13.0 EXIT-BAND (4 static) + BRACKET-STATE (2 dynamic) blocks + the exit-band penalty test."""
from __future__ import annotations
import numpy as np
from src.indicators.bollinger import bollinger
from src.data.aux_features import OHLC_COLUMNS

N_EXIT_BAND: int = 4       # == config.constants.OBS_BLOCK_EXIT_BAND    (STATIC rooms)
N_BRACKET_STATE: int = 2   # == config.constants.OBS_BLOCK_BRACKET_STATE (DYNAMIC TP/SL)

BB_PERIOD: int = 20        # the operator's band period on the 1m chart
BB_DEV: float = 0.5        # +/- 0.5 sigma rails around SMA20(high) / SMA20(low)
BRACKET_DIST_SCALE: float = 5.0   # entry-ATR units: 5 ATR of room -> 1.0 (matches trade_risk.ATR_PNL_SCALE)

_H1 = OHLC_COLUMNS.index("1m__high")
_L1 = OHLC_COLUMNS.index("1m__low")
_EPS = 1e-9


def compute_exit_band_rails(ohlc) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(buy_up, buy_lo, sell_up, sell_lo), each (T,) float32 (NaN warmup — kept NaN so warmup = no penalty).

    BUY band = BB(20, 0.5) on the 1m HIGH series; SELL band = BB(20, 0.5) on the 1m LOW series. LEAK-FREE
    (rolling window ends at the CLOSED bar t); precompute-only. These raw rails feed BOTH the obs rooms and the
    reward penalty."""
    ohlc = np.asarray(ohlc, dtype=np.float64)
    if ohlc.ndim != 2 or ohlc.shape[0] == 0:
        z = np.zeros(0, np.float32)
        return z, z.copy(), z.copy(), z.copy()
    high = ohlc[:, _H1]
    low = ohlc[:, _L1]
    buy_up, _bm, buy_lo = bollinger(high, BB_PERIOD, BB_DEV)     # band on HIGH -> the BUY band
    sell_up, _sm, sell_lo = bollinger(low, BB_PERIOD, BB_DEV)    # band on LOW  -> the SELL band
    return (buy_up.astype(np.float32), buy_lo.astype(np.float32),
            sell_up.astype(np.float32), sell_lo.astype(np.float32))


def compute_exit_band_matrix(close, rails) -> np.ndarray:
    """(T, 4) float32 STATIC obs block: signed room from close to each rail, normalized by the band half-width
    and clipped [-1,1]. NEGATIVE => close is OUTSIDE that rail (the penalty zone). NaN warmup -> 0 (neutral).

    Order (== observation_contract.EXIT_BAND_NAMES):
      0 xb_buyband_up_room   1 xb_buyband_lo_room   2 xb_sellband_up_room   3 xb_sellband_lo_room
    """
    close = np.asarray(close, dtype=np.float64).ravel()
    T = close.shape[0]
    if T == 0:
        return np.zeros((0, N_EXIT_BAND), dtype=np.float32)
    buy_up, buy_lo, sell_up, sell_lo = (np.asarray(r, dtype=np.float64).ravel() for r in rails)
    hw_buy = (buy_up - buy_lo) / 2.0                            # band half-width (= 0.5 sigma of high)
    hw_sell = (sell_up - sell_lo) / 2.0

    def room(x):
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    buy_up_room = room((buy_up - close) / (hw_buy + _EPS))      # <0 => close ABOVE buy upper rail (outside)
    buy_lo_room = room((close - buy_lo) / (hw_buy + _EPS))      # <0 => close BELOW buy lower rail (outside)
    sell_up_room = room((sell_up - close) / (hw_sell + _EPS))
    sell_lo_room = room((close - sell_lo) / (hw_sell + _EPS))
    out = np.stack([buy_up_room, buy_lo_room, sell_up_room, sell_lo_room], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_bracket_state(*, pos, price, tp_price, sl_price, entry_atr) -> np.ndarray:
    """(2,) float32 DYNAMIC bracket block: signed distance from price to the locked TP / SL in entry-ATR units,
    clip[-1,1], zero when flat or no bracket. jnp twin: jax_obs_blocks.bracket_state_features.

    Order (== observation_contract.BRACKET_STATE_NAMES):
      0 bs_dist_to_tp_atr   1 bs_dist_to_sl_atr
    """
    pos = float(pos)
    in_trade = 1.0 if pos != 0.0 else 0.0
    eatr = max(float(entry_atr), _EPS)
    has_tp = 1.0 if float(tp_price) != 0.0 else 0.0
    has_sl = 1.0 if float(sl_price) != 0.0 else 0.0
    tp_room = _clip(pos * (float(tp_price) - float(price)) / eatr / BRACKET_DIST_SCALE, -1.0, 1.0) * in_trade * has_tp
    sl_room = _clip(pos * (float(price) - float(sl_price)) / eatr / BRACKET_DIST_SCALE, -1.0, 1.0) * in_trade * has_sl
    return np.array([tp_room, sl_room], dtype=np.float32)


def exit_outside_band(direction, price, buy_up, buy_lo, sell_up, sell_lo) -> bool:
    """True if a close of `direction` (>0 long/BUY, <0 short/SELL) at `price` lands OUTSIDE that direction's
    1m BB(20,0.5) band (BUY -> band on HIGH, SELL -> band on LOW). NaN rails (warmup) -> False (no penalty).
    jnp twin: the branchless mask in jax_portfolio_env.step_portfolio."""
    if direction > 0:
        up, lo = float(buy_up), float(buy_lo)
    elif direction < 0:
        up, lo = float(sell_up), float(sell_lo)
    else:
        return False
    if not (np.isfinite(up) and np.isfinite(lo)):
        return False
    p = float(price)
    return bool(p > up or p < lo)


def empty_bracket_state() -> np.ndarray:
    return np.zeros(N_BRACKET_STATE, dtype=np.float32)


def _clip(x, lo, hi):
    return float(np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi))
