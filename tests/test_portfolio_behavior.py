# Phase 3 behavior (2026-06-27): the trained portfolio bot must match Mark's rules --
#   * default = keep trading after banking +2.5% (1% leash) -> phase2_continue ON
#   * bank +2.5% NET OF FEES (the amount you actually keep)
#   * +10% / 4 daily-passes-in-a-row = a BIG bonus, and TRAINING keeps going past +10% (consistency)
#   * position sizes SCALE with the account balance (same logic at any FTMO size)
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config, FTMOConfig
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv, build_portfolio_subs


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _pf_env(prices, cfg=None, start="2026-03-02 09:00", **kw):
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, np.asarray(prices, dtype=np.float64), idx)}
    return PortfolioEnv(sd, _reg, warmup=0, cfg=cfg or load_ftmo_config(), **kw)


def test_default_is_phase2_continue_on():
    # Mark's decision 2026-06-27: after banking +2.5%, keep trading under the 1% leash by default.
    assert load_ftmo_config().phase2_continue is True


def test_bank_is_net_of_fees_at_least_2pct5():
    # A +0.04 move banks; because the trigger uses TRUE post-fee equity, the banked balance is >= +2.5%
    # of the initial (you actually KEEP the target, fees already accounted for).
    env = _pf_env([100.0, 100.04, 100.04, 100.04], cfg=FTMOConfig(phase2_continue=False))
    env.reset()
    env.step(C.ACTION_BUY)
    assert env.position["TESTPAIR"] == 0
    assert (env.acc.balance - 100_000.0) >= 2_500.0, "banked at least +2.5% NET of fees"


def test_continue_after_pass_keeps_training_past_10pct():
    # +11% in one move. Default (eval) ends at +10%; training (continue_after_pass) keeps going.
    prices = [100.0, 100.11, 100.11]
    ev = _pf_env(prices, cfg=FTMOConfig(phase2_continue=False))   # default continue_after_pass=False (eval)
    ev.reset()
    _, _, term, _, _ = ev.step(C.ACTION_BUY)
    assert ev.acc.episode_passed is True and term is True          # a real challenge ENDS at +10%

    tr = _pf_env(prices, cfg=FTMOConfig(phase2_continue=False))
    tr.continue_after_pass = True                                  # training keeps going for consistency
    tr.reset()
    _, _, term, _, _ = tr.step(C.ACTION_BUY)
    assert tr.acc.episode_passed is True and term is False         # passed flag set, but NOT terminated


def test_four_daily_passes_in_a_row_earns_a_big_bonus():
    # 4 days that each bank +2.5% (net) in a row -> a big bonus at the 4th day's rollover; training continues.
    # Each "day" = 2 bars (00:00 open, 12:00 jump +0.04 to bank); price carries up day to day.
    days = 5
    prices, ts = [], []
    base = 100.0
    for d in range(days):
        day0 = base + d * 0.04
        prices += [day0, day0 + 0.04]                              # 00:00 (open), 12:00 (+0.04 -> bank)
        ts += [f"2026-03-{2 + d:02d} 00:00", f"2026-03-{2 + d:02d} 12:00"]
    idx = pd.to_datetime(ts).values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((len(prices), C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, np.asarray(prices, dtype=np.float64), idx)}
    env = PortfolioEnv(sd, _reg, warmup=0, cfg=FTMOConfig(phase2_continue=False), continue_after_pass=True)
    env.reset()
    rewards, max_streak = [], 0
    for i in range(len(prices) - 1):
        action = C.ACTION_BUY if i % 2 == 0 else C.ACTION_HOLD     # BUY at each day open, HOLD over the jump+midnight
        _, r, term, trunc, info = env.step(action)
        rewards.append(r)
        max_streak = max(max_streak, info.get("daily_pass_streak", 0))
        if term or trunc:
            break
    assert max_streak >= 4, f"should reach a 4-day pass streak (got {max_streak})"
    assert max(rewards) >= 0.5, "the 4-in-a-row bonus (~pass_bonus=1.0) should dwarf the tiny step rewards"


def test_position_size_scales_with_account_balance():
    n = 600
    idx = pd.date_range("2026-03-02", periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    close = (1.10 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 3e-4)).astype(np.float32)
    sd = {"EURUSD": (np.zeros((n, C.N_INDICATORS_TOTAL), np.float32), close, idx)}
    subs_100 = build_portfolio_subs(sd, _reg, progress=False, cfg=FTMOConfig(starting_balance=100_000.0))
    subs_50 = build_portfolio_subs(sd, _reg, progress=False, cfg=FTMOConfig(starting_balance=50_000.0))
    ratio = subs_50["EURUSD"].position_size / subs_100["EURUSD"].position_size
    assert abs(ratio - 0.5) < 0.05, f"halving the account should ~halve the position size (got {ratio:.3f})"
