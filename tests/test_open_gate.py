# Phase 3: 5m CCI open-gate. Blocks NEW directional opens when EITHER 5m CCI is in
# [-50,50]; holds/closes still pass; a flip just closes (no opposite open). Off by default.
import numpy as np, pandas as pd
from config import constants as C
from src.data.cache_builder import build_aligned_indicators
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry

def _cache(n=600):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    cl = 100 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.05)
    df = pd.DataFrame({"open": cl, "high": cl + .03, "low": cl - .03, "close": cl, "volume": 1.0}, index=idx)
    return build_aligned_indicators(df), df["close"].values.astype("float32"), idx.values.astype("datetime64[ns]").astype("int64")

def test_open_gate_blocks_and_allows():
    ind, cl, t = _cache()
    env = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, open_gate=True, position_size=1.0)
    env.reset()
    env.open_gate_blocked[:] = True                 # force the 5m-neutral condition everywhere
    env.step(C.ACTION_BUY);  assert env.position == 0   # blocked: no fresh open
    env.step(C.ACTION_SELL); assert env.position == 0   # blocked: no fresh open
    env.open_gate_blocked[:] = False                # 5m not neutral -> opens allowed
    env.step(C.ACTION_BUY);  assert env.position == 1   # opens long
    env.open_gate_blocked[:] = True                 # block again
    env.step(C.ACTION_SELL); assert env.position == 0   # flip blocked -> just closes
    env.open_gate_blocked[:] = False; env.step(C.ACTION_BUY); assert env.position == 1
    env.open_gate_blocked[:] = True
    env.step(C.ACTION_HOLD);  assert env.position == 1  # hold passes
    env.step(C.ACTION_CLOSE); assert env.position == 0  # close passes

def test_open_gate_off_by_default():
    ind, cl, t = _cache()
    env = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, position_size=1.0)   # default: gate OFF
    env.reset(); env.open_gate_blocked[:] = True                # even if neutral...
    env.step(C.ACTION_BUY); assert env.position == 1            # ...opens normally (gate off)


def test_open_gate_or_semantics():
    """The mask must be OR: blocked if EITHER 5m CCI is in [-50,50] (both must be
    outside the band to allow a new open). This also proves it is not the AND rule."""
    from src.indicators.base import ALL_INDICATOR_COLUMNS
    ind, cl, t = _cache(n=1500)                       # long enough to warm 5m cci100 + vary
    env = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, open_gate=True, position_size=1.0)
    c30 = np.abs(ind[:, ALL_INDICATOR_COLUMNS.index("5m__cci30_raw")]) <= 50.0
    c100 = np.abs(ind[:, ALL_INDICATOR_COLUMNS.index("5m__cci100_raw")]) <= 50.0
    assert np.array_equal(env.open_gate_blocked, c30 | c100)       # OR, exactly
    differ = c30 ^ c100                                            # exactly-one-neutral bars
    assert differ.any()                                           # such bars exist in this cache
    assert env.open_gate_blocked[differ].all()                    # OR blocks them; AND would NOT
