# =====================================================================
# WHEN 2026-07-01 | WHO Claude for Monty | WHY lock the v1.13.0 EXIT-BAND discipline.
# WHERE tests/test_exit_band.py
# HOW  Locks src/observation/exit_band: (1) the 4 STATIC rooms + 4 raw rails have the right shape / are
#      normalized [-1,1], (2) the BUY band (BB20/0.5 on HIGH) sits ABOVE the SELL band (on LOW), (3) the
#      exit_outside_band penalty test is correct + NaN-safe, (4) the precompute is LEAK-FREE (a prefix
#      reproduces the past), (5) build_bracket_state signs, (6) TradingEnv wires the rails + the obs block.
#      IRAC: I the bot had no learned EXIT discipline. R operator: BB(20,0.5) on 1m High/Low + penalty for
#      closing outside it. A a static band block + a penalty. C the bot learns to bank into the pause.
# =====================================================================
import numpy as np
import pandas as pd

from src.data.cache_builder import build_aligned_indicators, build_aligned_aux
from src.data import aux_features as AX
from src.observation import observation_contract as OC
from src.observation.exit_band import (
    compute_exit_band_rails, compute_exit_band_matrix, build_bracket_state, exit_outside_band, N_EXIT_BAND)


def _series(n=4000, drift=0.0, seed=1):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(seed)
    cl = 100.0 + np.cumsum(rng.standard_normal(n) * 0.03 + drift)
    hi = cl + np.abs(rng.standard_normal(n)) * 0.02
    lo = cl - np.abs(rng.standard_normal(n)) * 0.02
    df = pd.DataFrame({"open": cl, "high": hi, "low": lo, "close": cl, "volume": 1.0}, index=idx)
    ind = build_aligned_indicators(df)
    aux = build_aligned_aux(df)
    ohlc = np.ascontiguousarray(aux[:, AX.OHLC_SLICE], dtype=np.float32)
    t = idx.values.astype("datetime64[ns]").astype("int64")
    return ind, df["close"].values.astype("float64"), ohlc, t


def test_rails_and_matrix_shape():
    _, cl, ohlc, _ = _series(seed=2)
    rails = compute_exit_band_rails(ohlc)
    assert all(r.shape == (cl.shape[0],) for r in rails)
    mat = compute_exit_band_matrix(cl, rails)
    assert mat.shape == (cl.shape[0], N_EXIT_BAND) and mat.dtype == np.float32
    assert np.isfinite(mat).all()                              # matrix is nan_to_num'd (warmup -> 0)
    assert (mat >= -1.0).all() and (mat <= 1.0).all()          # normalized rooms


def test_buy_band_sits_above_sell_band():
    """BUY band = BB(20,0.5) on HIGH; SELL band = BB(20,0.5) on LOW. SMA(high) >= SMA(low) -> the buy band is
    centered above the sell band, and each band's upper rail >= its lower rail."""
    _, _, ohlc, _ = _series(seed=3)
    buy_up, buy_lo, sell_up, sell_lo = compute_exit_band_rails(ohlc)
    w = 300                                                    # skip BB warmup
    assert np.nanmean(buy_up[w:]) >= np.nanmean(sell_lo[w:])
    assert (buy_up[w:] >= buy_lo[w:]).all() and (sell_up[w:] >= sell_lo[w:]).all()


def test_exit_outside_band_logic():
    """Controlled band: BUY band [10,12], SELL band [8,10]. Penalty fires OUTSIDE the direction's rails only."""
    assert exit_outside_band(1, 13.0, 12.0, 10.0, 10.0, 8.0) is True    # long ABOVE buy upper
    assert exit_outside_band(1, 9.0, 12.0, 10.0, 10.0, 8.0) is True     # long BELOW buy lower
    assert exit_outside_band(1, 11.0, 12.0, 10.0, 10.0, 8.0) is False   # long INSIDE the buy band
    assert exit_outside_band(-1, 11.0, 12.0, 10.0, 10.0, 8.0) is True   # short ABOVE sell upper (10)
    assert exit_outside_band(-1, 9.0, 12.0, 10.0, 10.0, 8.0) is False   # short INSIDE the sell band [8,10]
    assert exit_outside_band(0, 100.0, 12.0, 10.0, 10.0, 8.0) is False  # flat -> never penalized
    assert exit_outside_band(1, 11.0, np.nan, np.nan, 10.0, 8.0) is False  # NaN rails (warmup) -> no penalty


def test_leak_free_prefix_reproduces_past():
    """The exit-band value at bar t must use ONLY bars <= t. Recompute on the prefix [:k]; earlier bars match."""
    _, cl, ohlc, _ = _series(seed=5)
    full = compute_exit_band_matrix(cl, compute_exit_band_rails(ohlc))
    k = 3000
    pref = compute_exit_band_matrix(cl[:k], compute_exit_band_rails(ohlc[:k]))
    lo, hi = 100, k - 50
    assert np.array_equal(full[lo:hi], pref[lo:hi]), "future bars leaked into a past exit-band value"


def test_bracket_state_directional_and_flat():
    long_b = build_bracket_state(pos=1.0, price=100.0, tp_price=101.0, sl_price=99.0, entry_atr=0.5)
    assert long_b[0] > 0 and long_b[1] > 0                     # room to TP (above) and SL (below) both positive
    short_b = build_bracket_state(pos=-1.0, price=100.0, tp_price=99.0, sl_price=101.0, entry_atr=0.5)
    assert short_b[0] > 0 and short_b[1] > 0                   # mirror for a short
    flat = build_bracket_state(pos=0.0, price=100.0, tp_price=0.0, sl_price=0.0, entry_atr=0.0)
    assert np.allclose(flat, 0.0)                              # flat / no bracket -> zeros


def test_env_wires_exit_band_and_obs_block():
    """TradingEnv._precompute must expose the raw rails + the static obs matrix, and the exit_band block must
    land at its contract slice in the built observation."""
    from src.env.trading_env import TradingEnv
    from src.strategies.registry import AlphaRegistry
    ind, cl, ohlc, t = _series(seed=6)
    aux = np.zeros((cl.shape[0], AX.AUX_NCOLS), dtype=np.float32)
    aux[:, AX.OHLC_SLICE] = ohlc
    env = TradingEnv(ind, cl.astype("float32"), t, AlphaRegistry(), warmup=210, aux=aux, position_size=1.0)
    # compare against the env's OWN inputs (it stores a float32-rounded close + the aux OHLC), not the originals
    rails = compute_exit_band_rails(env.ohlc_matrix)
    assert np.array_equal(env.exit_buy_up, rails[0], equal_nan=True)    # rails keep NaN warmup -> equal_nan
    assert np.array_equal(env.exit_sell_lo, rails[3], equal_nan=True)
    assert np.array_equal(env.exit_band_matrix, compute_exit_band_matrix(env.close, rails))
    o, _ = env.reset()                                          # ptr is only set on reset()
    assert o.shape == (563,)
    xb = o[OC.BLOCK_SLICES["exit_band"]]
    assert xb.shape == (4,) and np.isfinite(xb).all()
