# Phase 3 (operator 2026-06-29: OR -> AND): 5m CCI open-gate. Blocks NEW directional opens when BOTH
# 5m CCI30 AND CCI100 are in [-50,50] (a flat/chop 5m); holds/closes still pass; a flip just closes. Off by default.
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


def test_open_gate_and_semantics():
    """The mask must be AND (operator 2026-06-29): blocked only when BOTH 5m CCIs are in [-50,50] (a flat
    5m). A new open is ALLOWED as long as at least ONE CCI shows momentum. Proves it is not the OR rule."""
    from src.indicators.base import ALL_INDICATOR_COLUMNS
    ind, cl, t = _cache(n=1500)                       # long enough to warm 5m cci100 + vary
    env = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, open_gate=True, position_size=1.0)
    c30 = np.abs(ind[:, ALL_INDICATOR_COLUMNS.index("5m__cci30_raw")]) <= 50.0
    c100 = np.abs(ind[:, ALL_INDICATOR_COLUMNS.index("5m__cci100_raw")]) <= 50.0
    assert np.array_equal(env.open_gate_blocked, c30 & c100)       # AND, exactly
    differ = c30 ^ c100                                            # exactly-one-neutral bars
    assert differ.any()                                           # such bars exist in this cache
    assert not env.open_gate_blocked[differ].any()                # AND ALLOWS them; OR would have blocked


def test_open_gate_threshold_is_configurable():
    """The +/-threshold is a SETTING. With threshold=100 a bar where a CCI sits
    between 50 and 100 is BLOCKED, though the default 50-gate would have ALLOWED it.
    Stricter threshold => blocks at least as many bars."""
    from src.indicators.base import ALL_INDICATOR_COLUMNS
    ind, cl, t = _cache(n=1500)
    j30 = ALL_INDICATOR_COLUMNS.index("5m__cci30_raw")
    j100 = ALL_INDICATOR_COLUMNS.index("5m__cci100_raw")
    env50 = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, open_gate=True, position_size=1.0)
    env100 = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, open_gate=True,
                        open_gate_threshold=100.0, position_size=1.0)
    assert env50.open_gate_threshold == 50.0 and env100.open_gate_threshold == 100.0
    assert env100.open_gate_blocked.sum() >= env50.open_gate_blocked.sum()   # wider band -> blocks at least as many
    # exact rule at 100 (AND): blocked where BOTH |cci| <= 100
    expect = (np.abs(ind[:, j30]) <= 100.0) & (np.abs(ind[:, j100]) <= 100.0)
    assert np.array_equal(env100.open_gate_blocked, expect)
