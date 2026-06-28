# First 2 of the 14-alpha pack: Regime Pulse trend + pullback (30m/4h). Logic + wiring.
import numpy as np, pandas as pd
from src.strategies.context import MarketContext
from src.strategies.registry import AlphaRegistry
from src.strategies.gravity_30m_4h_alpha import register_gravity_alpha
from src.strategies.regime_pulse_trend_30m_4h_alpha import RegimePulseTrend30m4hAlpha
from src.strategies.register_regime_pulse_trend_30m_4h_alpha import register as reg_trend
from src.strategies.regime_pulse_pullback_30m_4h_alpha import RegimePulsePullback30m4hAlpha
from src.strategies.register_regime_pulse_pullback_30m_4h_alpha import register as reg_pull
from src.data.cache_builder import build_aligned_indicators
from src.env.trading_env import TradingEnv
from src.observation import observation_contract as OC


def _ctx(c4h, c30, b20_4h, b200_4h, b20_30, b200_30):
    ind = {"4h__sma_p1_s0": c4h, "4h__bb20_dev1.0_middle": b20_4h, "4h__bb200_dev1.0_middle": b200_4h,
           "30m__sma_p1_s0": c30, "30m__bb20_dev1.0_middle": b20_30, "30m__bb200_dev1.0_middle": b200_30}
    return MarketContext(close=c30, indicators=ind)


def test_regime_pulse_trend_logic():
    a = RegimePulseTrend30m4hAlpha()
    assert a.compute_signal(_ctx(110, 110, 100, 100, 100, 100)) == 1    # all above both mids
    assert a.compute_signal(_ctx(90, 90, 100, 100, 100, 100)) == -1     # all below both mids
    assert a.compute_signal(_ctx(110, 90, 100, 100, 100, 100)) == 0     # TFs disagree -> flat


def test_regime_pulse_pullback_logic():
    a = RegimePulsePullback30m4hAlpha()
    # bull pullback: 4h above both; 30m above BB200-mid (98) but below BB20-mid (100)
    assert a.compute_signal(_ctx(110, 99, 100, 100, 100, 98)) == 1
    # bear pullback: 4h below both; 30m below BB200-mid (102) but above BB20-mid (100)
    assert a.compute_signal(_ctx(90, 101, 100, 100, 100, 102)) == -1
    # full trend (30m above BOTH) is NOT a pullback -> 0
    assert a.compute_signal(_ctx(110, 110, 100, 100, 100, 100)) == 0


def test_regime_pulse_wiring_slots_and_479():
    rng = np.random.default_rng(0); n = 600
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    cl = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    df = pd.DataFrame({"open": cl, "high": cl + .03, "low": cl - .03, "close": cl, "volume": 1.}, index=idx)
    ind = build_aligned_indicators(df); close = df["close"].values.astype("float32")
    t = idx.values.astype("datetime64[ns]").astype("int64")
    reg = AlphaRegistry()
    s0 = register_gravity_alpha(reg)          # gravity -> slot 0
    s1 = reg_trend(reg)                        # -> slot 1
    s2 = reg_pull(reg)                         # -> slot 2
    assert (s0, s1, s2) == (0, 1, 2)
    env = TradingEnv(ind, close, t, reg, warmup=210); o, _ = env.reset()
    assert o.shape == (499,)                                   # contract unchanged
    assert o[OC.BLOCK_SLICES["alpha_mask"]][s1] == 1.0 and o[OC.BLOCK_SLICES["alpha_mask"]][s2] == 1.0           # both slots occupied (mask)
