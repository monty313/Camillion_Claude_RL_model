# Phase 2: gravity alpha logic (+1/-1/0 with 30m & 4h agreement) and that it
# wires into one alpha slot without changing the 471 contract.
import numpy as np, pandas as pd
from src.strategies.context import MarketContext
from src.strategies.gravity_30m_4h_alpha import Gravity30m4hAlpha, register_gravity_alpha
from src.strategies.registry import AlphaRegistry
from src.data.cache_builder import build_aligned_indicators
from src.env.trading_env import TradingEnv
from src.observation import observation_contract as OC


def _ctx(tf_dirs, g=None):
    """Populate exactly the columns the alpha declares it reads (any mapping)."""
    g = g or Gravity30m4hAlpha()
    ind = {}
    for tf, d in tf_dirs.items():
        for c in g.CCI_COLS:
            ind[f"{tf}__{c}"] = 120.0 * d if d else 5.0
        for c in g.RSI_COLS:
            ind[f"{tf}__{c}"] = 62.0 if d > 0 else (38.0 if d < 0 else 50.0)
        mid = 100.0; price = mid + 0.5 * d                 # upper/lower half
        for cfg in g.BB_CONFIGS:
            ind[f"{tf}__{cfg}_middle"] = mid
            ind[f"{tf}__{cfg}_upper"] = mid + 1.0
            ind[f"{tf}__{cfg}_lower"] = mid - 1.0
        ind[f"{tf}__{g.CLOSE_COL}"] = price                # close = sma_p1_s0 = fan[0]
        for c in g.SMA_FAN[1:]:
            ind[f"{tf}__{c}"] = mid
    return MarketContext(close=100.0, indicators=ind)


def test_gravity_logic():
    g = Gravity30m4hAlpha()
    assert len(g.BB_CONFIGS) == 8 and len(g.CCI_COLS) == 2 and len(g.RSI_COLS) == 2
    assert g.signal(_ctx({"30m": 1, "4h": 1})) == 1     # both bullish -> +1
    assert g.signal(_ctx({"30m": -1, "4h": -1})) == -1   # both bearish -> -1
    assert g.signal(_ctx({"30m": 1, "4h": -1})) == 0     # conflict -> 0
    assert g.signal(_ctx({"30m": 0, "4h": 0})) == 0      # dead zones -> 0


def test_gravity_wiring_one_slot():
    rng = np.random.default_rng(0); n = 600
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    df = pd.DataFrame({"open": close, "high": close + 0.03, "low": close - 0.03,
                       "close": close, "volume": 1.0}, index=idx)
    ind = build_aligned_indicators(df); cl = df["close"].values.astype("float32")
    tn = df.index.values.astype("datetime64[ns]").astype("int64")
    reg = AlphaRegistry(); slot = register_gravity_alpha(reg)
    assert reg.assigned_count == 1 and slot == 0
    env = TradingEnv(ind, cl, tn, reg, warmup=210)
    o, _ = env.reset()
    assert o.shape == (471,)                       # contract unchanged
    assert o[OC.BLOCK_SLICES["alpha_mask"]][slot] == 1.0                    # mask: slot occupied
    for _ in range(20):
        o, r, te, tr, _ = env.step(1)
        assert o.shape == (471,) and np.isfinite(r)


def test_gravity_vote_modes_debug():
    """Default is family; print per-TF tallies under BOTH modes (debug aid) + assert the flip."""
    g = Gravity30m4hAlpha()
    assert g.VOTE_MODE == "family"
    ind = {}
    for tf in ("30m", "4h"):
        for c in g.CCI_COLS: ind[f"{tf}__{c}"] = -120.0
        for c in g.RSI_COLS: ind[f"{tf}__{c}"] = 35.0
        mid = 100.0
        for cfg in g.BB_CONFIGS:
            ind[f"{tf}__{cfg}_middle"] = mid; ind[f"{tf}__{cfg}_upper"] = mid + 1; ind[f"{tf}__{cfg}_lower"] = mid - 1
        ind[f"{tf}__{g.CLOSE_COL}"] = 100.6
        for c in g.SMA_FAN[1:]: ind[f"{tf}__{c}"] = 101.0
    ctx = MarketContext(close=100.6, indicators=ind)
    dbg = g.debug_votes(ctx)
    for tf in ("30m", "4h"):
        d = dbg[tf]
        print(f"  [{tf}] cci={d['cci']} rsi={d['rsi']} bb={d['bb']} sma={d['sma']} -> flat={d['flat']:+d} family={d['family']:+d}")
    assert Gravity30m4hAlpha(vote_mode="flat").compute_signal(ctx) == 1
    assert Gravity30m4hAlpha(vote_mode="family").compute_signal(ctx) == -1
