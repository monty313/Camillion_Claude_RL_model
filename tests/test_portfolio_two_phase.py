# Two-phase daily engine ON THE SHARED POT (2026-06-26): the PortfolioEnv (the env we actually TRAIN)
# must enforce the same documented FTMO rule as the single-symbol TradingEnv -- hit +2.5% of the INITIAL
# balance (measured on pot EQUITY) -> close the WHOLE book & bank it; DEFAULT: stop for the day (no new
# opens); OPTIONAL (phase2_continue): keep trading under a 1% trailing wall from the banked peak, and
# giving that back banks & stops (NOT a challenge breach). Before this, two-phase lived ONLY in the
# single-symbol env, so the trained portfolio bot silently ignored it.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config, FTMOConfig
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _pf_env(prices, cfg=None, start="2026-03-02 09:00", **kw):
    """One-symbol shared-pot env over hand-built prices. 'TESTPAIR' is not in the asset specs, so its
    position_size defaults to 100,000 -> a +0.025 price move = +$2,500 = +2.5% of $100k (clean math)."""
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    close = np.asarray(prices, dtype=np.float64)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, close, idx)}
    return PortfolioEnv(sd, _reg, warmup=0, cfg=cfg or load_ftmo_config(), **kw)


def test_portfolio_banks_at_2pct5_then_stops_by_default():
    """+0.04 on 1 lot ~ +$4,000 = +4% of $100k (clears the tiny round-trip cost) -> the pot banks the day
    and locks: it closes the whole book and blocks new opens for the rest of the day."""
    env = _pf_env([100.0, 100.04, 100.04, 100.04], cfg=FTMOConfig(phase2_continue=False))
    env.reset()
    _, _, term, trunc, info = env.step(C.ACTION_BUY)     # opens; next bar marks +>2.5% -> auto-bank ALL + lock
    assert env.position["TESTPAIR"] == 0, "must close the WHOLE book at +2.5%"
    assert env.acc.balance > 100_000, "banked a profit into the shared pot"
    assert info["day_locked"] is True
    _, _, term, trunc, info = env.step(C.ACTION_BUY)     # rest of the day: new opens are blocked
    assert env.position["TESTPAIR"] == 0 and info["day_locked"] is True
    assert not env.acc.episode_breached


def test_portfolio_phase2_continue_then_protective_stop():
    """phase2_continue=True: after banking +2.5% the pot MAY keep trading, but giving back 1% of the
    banked peak flattens and stops the day -- and that protective stop is NOT a challenge breach."""
    cfg = FTMOConfig(phase2_continue=True)
    env = _pf_env([100.0, 100.04, 100.02, 100.02], cfg=cfg)
    env.reset()
    _, _, _, _, info = env.step(C.ACTION_BUY)            # banks +2.5%; phase-2 active, day NOT locked
    assert env.position["TESTPAIR"] == 0
    assert info["phase2_active"] is True and info["day_locked"] is False
    env.step(C.ACTION_BUY)                               # allowed to re-open in phase 2; then gives back >1%
    assert env.position["TESTPAIR"] == 0 and env._day_locked is True
    assert not env.acc.episode_breached, "phase-2 protective stop is NOT a breach"


def test_portfolio_day_lock_clears_next_day():
    """Bank + lock on day 1; crossing midnight clears the lock so the pot can trade again the next day."""
    env = _pf_env([100.0, 100.04, 100.04, 100.04, 100.04, 100.04, 100.04, 100.04],
                  start="2026-03-02 23:57", cfg=FTMOConfig(phase2_continue=False))
    env.reset()
    env.step(C.ACTION_BUY)                               # banks + locks day 1
    assert env._day_locked is True
    crossed = False
    for _ in range(6):
        env.step(C.ACTION_HOLD)
        if env._cur_date == np.datetime64("2026-03-03"):
            crossed = True
            assert env._day_locked is False             # new day -> lock cleared
            break
    assert crossed


def test_portfolio_two_phase_off_means_no_auto_bank():
    """With two_phase disabled the pot does NOT auto-flatten at +2.5% (the rule is config-driven)."""
    cfg = FTMOConfig(two_phase_enabled=False)
    env = _pf_env([100.0, 100.04, 100.04, 100.04], cfg=cfg)
    env.reset()
    _, _, _, _, info = env.step(C.ACTION_BUY)
    assert env.position["TESTPAIR"] == 1, "two-phase off -> stays in the position"
    assert info["day_locked"] is False
