# The alpha pack: per-family logic truth-tables + full-pack wiring (18 alphas, 557 obs).
import numpy as np, pandas as pd
from src.strategies.context import MarketContext
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.data.cache_builder import build_aligned_indicators
from src.env.trading_env import TradingEnv
from src.observation import observation_contract as OC
from src.strategies.cci_surge_trend_30m_4h_alpha import CciSurgeTrend30m4hAlpha
from src.strategies.cci_surge_pullback_30m_4h_alpha import CciSurgePullback30m4hAlpha
from src.strategies.sma_stack_trend_30m_4h_alpha import SmaStackTrend30m4hAlpha
from src.strategies.sma_reversion_rally_30m_4h_alpha import SmaReversionRally30m4hAlpha

def _ctx(d): return MarketContext(close=0.0, indicators=d)

def test_cci_surge_logic():
    tr = CciSurgeTrend30m4hAlpha()
    assert tr.compute_signal(_ctx({"4h__cci30_raw": 50, "4h__cci100_raw": 50, "30m__cci30_raw": 50, "30m__cci100_raw": 50})) == 1
    assert tr.compute_signal(_ctx({"4h__cci30_raw": -50, "4h__cci100_raw": -50, "30m__cci30_raw": -50, "30m__cci100_raw": -50})) == -1
    assert tr.compute_signal(_ctx({"4h__cci30_raw": 50, "4h__cci100_raw": 50, "30m__cci30_raw": -5, "30m__cci100_raw": 50})) == 0
    pb = CciSurgePullback30m4hAlpha()
    assert pb.compute_signal(_ctx({"4h__cci30_raw": 50, "4h__cci100_raw": 50, "30m__cci30_raw": -5, "30m__cci100_raw": 20})) == 1
    assert pb.compute_signal(_ctx({"4h__cci30_raw": -50, "4h__cci100_raw": -50, "30m__cci30_raw": 5, "30m__cci100_raw": -20})) == -1

def test_sma_stack_logic():
    a = SmaStackTrend30m4hAlpha()
    up = {"4h__sma_p1_s0": 110, "4h__sma4_sh4_high": 105, "4h__sma4_sh4_low": 100,
          "30m__sma_p1_s0": 110, "30m__sma4_sh4_high": 105, "30m__sma4_sh4_low": 100}
    assert a.compute_signal(_ctx(up)) == 1
    dn = dict(up); dn["4h__sma_p1_s0"] = 90; dn["30m__sma_p1_s0"] = 90
    assert a.compute_signal(_ctx(dn)) == -1

def test_sma_reversion_logic():
    a = SmaReversionRally30m4hAlpha()
    bull = {"4h__sma_p30_s0": 105, "4h__sma_p50_s0": 100,
            "30m__sma_p1_s0": 101, "30m__sma_p30_s0": 100, "30m__sma_p1_s1": 99}   # prev<=sma30, close>sma30
    assert a.compute_signal(_ctx(bull)) == 1
    bear = {"4h__sma_p30_s0": 95, "4h__sma_p50_s0": 100,
            "30m__sma_p1_s0": 99, "30m__sma_p30_s0": 100, "30m__sma_p1_s1": 101}
    assert a.compute_signal(_ctx(bear)) == -1

def test_alpha_pack_wiring_and_479():
    n = 3000; idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    cl = 100 + np.cumsum(np.random.default_rng(3).standard_normal(n) * 0.04 + 0.01)
    df = pd.DataFrame({"open": cl, "high": cl + .05, "low": cl - .05, "close": cl, "volume": 1.}, index=idx)
    ind = build_aligned_indicators(df); assert ind.shape[1] == 220
    reg = AlphaRegistry(); register_all(reg); assert reg.assigned_count == 21   # +3 strong-setup alphas (2026-06-29)
    env = TradingEnv(ind, df["close"].values.astype("float32"),
                     idx.values.astype("datetime64[ns]").astype("int64"), reg, warmup=300)
    o, _ = env.reset()
    assert o.shape == (557,) and np.all(np.isfinite(o))
    assert o[OC.BLOCK_SLICES["alpha_mask"]][:21].sum() == 21           # all 21 slots occupied (incl. the 3 new setups)
    assert np.all(np.isfinite(o[OC.BLOCK_SLICES["alpha_streak"]]))     # streak block present
    fires = (env.alpha_matrix[300:, :21] != 0).sum(axis=0)
    assert fires.sum() > 0                                             # the pack produces signals


def test_conviction_slots_match_canonical_order():
    # the conviction bonus reads these EXACT slots in BOTH the CPU and JAX envs; a reorder must not drift them.
    from src.strategies.alpha_pack import conviction_slots, CONVICTION_SLOTS, CONVICTION_ALPHA_NAMES
    assert conviction_slots() == CONVICTION_SLOTS == (18, 19, 20)
    assert CONVICTION_ALPHA_NAMES == ("cci_x160_align_5m_30m", "bb_double_breakout_anytf", "fwd_sma4_align_5m_30m")
