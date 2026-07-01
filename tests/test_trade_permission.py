# =====================================================================
# WHEN 2026-07-01 | WHO Claude for Monty | WHY lock the "training wheels" directional gate.
# WHERE tests/test_trade_permission.py
# HOW  Locks src/observation/trade_permission.compute_trade_permission: (1) the exact indicator columns the
#      3 groups read still EXIST (a rename would silently kill a group), (2) a clean UPTREND permits BUY /
#      forbids SELL and a DOWNTREND does the mirror, (3) the output is a 0/1 (T,2) mask, (4) it is LEAK-FREE
#      (recomputing on a prefix reproduces the past bar-for-bar), (5) TradingEnv wires it into
#      trade_wheel_sell/buy. IRAC: I RL can't find a known edge. R operator gave the entry conditions. A gate
#      the action space to them. C the bot only explores the operator's windows -> it can FIND the edge.
# =====================================================================
import numpy as np
import pandas as pd

from src.data.cache_builder import build_aligned_indicators, build_aligned_aux
from src.data import aux_features as AX
from src.indicators.base import ALL_INDICATOR_COLUMNS
from src.observation.trade_permission import compute_trade_permission


def _series(n=6000, drift=0.0, seed=1):
    """A 1m OHLC frame with a controllable drift (uptrend / downtrend / flat)."""
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


def test_needed_columns_exist():
    """A silent rename of any of these -> _col returns all-NaN -> that whole condition group DIES quietly.
    Lock the exact names the gate reads."""
    need = [f"{tf}__cci{p}_raw" for tf in ("5m", "30m") for p in ("30", "100")]
    need += [f"{tf}__bb{p}_dev1.0_middle" for tf in ("5m", "30m", "4h") for p in ("20", "200")]
    missing = [n for n in need if n not in ALL_INDICATOR_COLUMNS]
    assert not missing, f"trade-permission columns missing (gate would silently disable a group): {missing}"


def test_shape_and_binary():
    ind, cl, ohlc, t = _series(drift=0.0, seed=2)
    perm = compute_trade_permission(ind, cl, ohlc, t)
    assert perm.shape == (cl.shape[0], 2) and perm.dtype == np.float32
    assert set(np.unique(perm)).issubset({0.0, 1.0}), "mask must be strictly 0/1"


def test_uptrend_permits_buy_forbids_sell():
    ind, cl, ohlc, t = _series(drift=0.02, seed=3)          # steady climb -> price above every SMA/BB middle
    perm = compute_trade_permission(ind, cl, ohlc, t)
    w = 500                                                  # skip warmup / indicator spin-up
    sell, buy = perm[w:, 0], perm[w:, 1]
    assert buy.mean() > 0.5, f"uptrend should broadly permit BUY, got {buy.mean():.2f}"
    assert sell.mean() < 0.05, f"uptrend should almost never permit SELL, got {sell.mean():.2f}"


def test_downtrend_permits_sell_forbids_buy():
    ind, cl, ohlc, t = _series(drift=-0.02, seed=4)         # steady fall -> the mirror
    perm = compute_trade_permission(ind, cl, ohlc, t)
    w = 500
    sell, buy = perm[w:, 0], perm[w:, 1]
    assert sell.mean() > 0.5, f"downtrend should broadly permit SELL, got {sell.mean():.2f}"
    assert buy.mean() < 0.05, f"downtrend should almost never permit BUY, got {buy.mean():.2f}"


def test_leak_free_prefix_reproduces_past():
    """Permission at bar t must use ONLY bars <= t. Recompute on the prefix [:k]; the earlier bars must be
    IDENTICAL to the full run (with a margin below k to exclude the forming TF bar at the cut)."""
    ind, cl, ohlc, t = _series(drift=0.01, seed=5)
    full = compute_trade_permission(ind, cl, ohlc, t)
    k = 4000
    pref = compute_trade_permission(ind[:k], cl[:k], ohlc[:k], t[:k])
    lo, hi = 500, k - 300                                   # 300-bar margin below the cut (>> the 4h shift-4 reach)
    assert np.array_equal(full[lo:hi], pref[lo:hi]), "FUTURE bars leaked into a past permission decision"


def test_env_wires_the_gate():
    """TradingEnv._precompute must expose trade_wheel_sell/buy == the direct compute (this is what the
    PortfolioEnv step + the JAX static-data reader consume)."""
    from src.env.trading_env import TradingEnv
    from src.strategies.registry import AlphaRegistry
    ind, cl, ohlc, t = _series(drift=0.01, seed=6)
    aux = np.zeros((cl.shape[0], AX.AUX_NCOLS), dtype=np.float32)
    aux[:, AX.OHLC_SLICE] = ohlc
    env = TradingEnv(ind, cl.astype("float32"), t, AlphaRegistry(), warmup=210, aux=aux, position_size=1.0)
    perm = compute_trade_permission(ind, cl, ohlc, t)
    assert np.array_equal(env.trade_wheel_sell, perm[:, 0])
    assert np.array_equal(env.trade_wheel_buy, perm[:, 1])
