# Two-phase daily engine (2026-06-25): hit +2.5% of the INITIAL balance (measured on
# EQUITY) -> close ALL & bank it. DEFAULT: stop for the day (no new opens). OPTIONAL
# (phase2_continue=True): keep trading under a tight 1% trailing wall from the banked
# peak; give it back -> bank & stop (NOT a breach). Phase-1 risk wall = 4% trailing breach.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config, FTMOConfig
from src.env.trading_env import TradingEnv
from src.account.account_state import AccountState
from src.risk.ftmo_rules import FTMORules
from src.strategies.registry import AlphaRegistry


def _env(prices, start="2026-03-02 09:00", cfg=None, **kw):
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="1min")
    close = np.asarray(prices, dtype=np.float64)            # float64 prices -> clean 2.5% math
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), dtype=np.float32)
    kw.setdefault("position_size", 100000.0)
    kw.setdefault("warmup", 0)
    kw.setdefault("cost_frac", 0.0)
    kw.setdefault("cfg", cfg or load_ftmo_config())
    return TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64),
                      AlphaRegistry(), **kw)


def test_daily_target_is_pct_of_initial_measured_on_equity():
    rules = FTMORules(load_ftmo_config())
    acc = AccountState(starting_balance=100_000.0)   # day_start = 100,000
    acc.mark_equity(102_400.0)                        # +2.4% (incl. open) -> not yet
    assert not rules.daily_target_hit(acc)
    acc.mark_equity(102_500.0)                        # +2.5% of INITIAL -> hit
    assert rules.daily_target_hit(acc)


def test_banks_at_2pct5_then_stops_by_default():
    """+0.025 on 1 lot = +$2,500 = +2.5% of $100k. Default -> close all + stop for the day."""
    env = _env([100.0, 100.025, 100.025, 100.025, 100.025])
    env.reset()
    _, _, term, trunc, info = env.step(C.ACTION_BUY)   # opens; marks +2.5% -> auto-bank + lock
    assert env.position == 0, "must close ALL at +2.5%"
    assert 2490 < (env.acc.balance - 100_000) < 2510, "banked ~+$2,500"
    assert info["day_locked"] is True
    _, _, term, trunc, info = env.step(C.ACTION_BUY)   # rest of day: no new opens
    assert env.position == 0 and info["day_locked"] is True
    assert not env.acc.episode_breached


def test_optional_continue_under_1pct_trail():
    """phase2_continue=True: after banking, may keep trading, but giving back 1% of the
    banked peak flattens and stops the day -- and that is NOT a challenge breach."""
    cfg = FTMOConfig(phase2_continue=True)
    env = _env([100.0, 100.025, 100.025, 100.014, 100.014, 100.014], cfg=cfg)
    env.reset()
    _, _, _, _, info = env.step(C.ACTION_BUY)          # banks +2.5%; continue -> phase2 active
    assert env.position == 0 and info["phase2_active"] is True and info["day_locked"] is False
    env.step(C.ACTION_BUY)                            # allowed to re-open in phase 2
    _, _, _, _, info = env.step(C.ACTION_HOLD)        # equity gives back >1% of peak -> stop
    assert env.position == 0 and info["day_locked"] is True
    assert not env.acc.episode_breached, "phase-2 protective stop is NOT a breach"


def test_phase1_4pct_trailing_is_a_breach():
    """Before the daily target, the risk wall is a 4% trailing drawdown = challenge fail."""
    env = _env([100.0, 99.99, 99.98, 99.97, 99.96, 99.96, 99.96])   # steady drop, ~4% from peak
    env.reset()
    env.step(C.ACTION_BUY)
    term = False
    info = {}
    for _ in range(6):
        _, r, term, trunc, info = env.step(C.ACTION_HOLD)
        if term:
            break
    assert term and env.acc.episode_breached
    assert "trailing_drawdown" in info["breach_reasons"]


def test_phase2_lock_clears_next_day():
    """Bank + lock on day 1; crossing midnight clears the lock so it can trade again."""
    env = _env([100.0, 100.025, 100.025, 100.025, 100.025, 100.025, 100.025, 100.025],
               start="2026-03-02 23:57")
    env.reset()
    env.step(C.ACTION_BUY)                            # banks + locks day 1
    assert env._day_locked is True
    crossed = False
    for _ in range(6):
        env.step(C.ACTION_HOLD)
        if env._cur_date == np.datetime64("2026-03-03"):
            crossed = True
            assert env._day_locked is False           # new day -> lock cleared
            break
    assert crossed
