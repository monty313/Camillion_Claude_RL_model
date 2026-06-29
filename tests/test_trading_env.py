# Phase 1 (critical guardrails): obs shape/finite; REWARD depends ONLY on prices
# + actions (never on alphas); daily FTMO state resets across midnight; breach
# terminates the episode.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.strategies.examples import register_examples


def _cache(n=400, start="2026-03-02 09:00", seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.1 + drift)
    ind = rng.standard_normal((n, C.N_INDICATORS_TOTAL)).astype(np.float32)
    return ind, close.astype(np.float32), idx.values.astype("datetime64[ns]").astype(np.int64)


def test_obs_shape_and_finite():
    ind, close, t = _cache()
    reg = AlphaRegistry(); register_examples(reg)
    env = TradingEnv(ind, close, t, reg, warmup=5)
    obs, _ = env.reset()
    assert obs.shape == (513,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    for a in (C.ACTION_BUY, C.ACTION_HOLD, C.ACTION_SELL, C.ACTION_CLOSE):
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (513,) and np.all(np.isfinite(obs))
        assert np.isfinite(r)


def test_reward_independent_of_alphas():
    """Same prices + same actions, DIFFERENT alphas -> identical rewards.
    Proves no alpha/accuracy term leaked into the reward path."""
    ind, close, t = _cache(seed=7)
    reg_full = AlphaRegistry(); register_examples(reg_full)   # alphas vary
    reg_empty = AlphaRegistry()                               # no alphas at all
    env1 = TradingEnv(ind, close, t, reg_full, warmup=5, position_size=1.0)
    env2 = TradingEnv(ind, close, t, reg_empty, warmup=5, position_size=1.0)
    env1.reset(); env2.reset()
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 4, size=200)
    r1, r2 = [], []
    for a in acts:
        _, x1, term1, tr1, _ = env1.step(int(a)); r1.append(x1)
        _, x2, term2, tr2, _ = env2.step(int(a)); r2.append(x2)
        if tr1 or tr2 or term1 or term2:
            break
    assert np.allclose(r1, r2), "reward differs when only alphas differ -> alpha leaked into reward!"


def test_breach_terminates():
    ind, close, t = _cache(n=60)
    close = close.copy(); close[:] = np.linspace(100, 70, len(close))  # 30% crash
    reg = AlphaRegistry()
    env = TradingEnv(ind, close, t, reg, warmup=2, position_size=100000.0,
                     cfg=load_ftmo_config())
    env.reset()
    term = False
    env.step(C.ACTION_BUY)               # go long into the crash
    for _ in range(40):
        _, r, term, trunc, info = env.step(C.ACTION_HOLD)
        if term:
            break
    assert term and info["breach_reasons"], "expected an FTMO breach termination"


def test_daily_state_resets_over_midnight():
    ind, close, t = _cache(n=10, start="2026-03-02 23:57")   # crosses midnight
    reg = AlphaRegistry()
    env = TradingEnv(ind, close, t, reg, warmup=0, position_size=10.0)
    env.reset()
    env.step(C.ACTION_BUY)
    env.step(C.ACTION_CLOSE)             # realize a trade on day 1 (pnl != 0)
    pnl_before = env.acc.daily_realized_pnl
    crossed = False
    for _ in range(8):
        _, _, term, trunc, _ = env.step(C.ACTION_HOLD)
        if env._cur_date == np.datetime64("2026-03-03"):
            crossed = True
            assert env.acc.daily_realized_pnl == 0.0   # reset over midnight
            break
        if trunc or term:
            break
    assert crossed and pnl_before != 0.0
