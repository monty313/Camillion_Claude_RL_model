# v1.7.0 RECENT-CONTEXT block: recent DAILY movement (prior days + last-week avg) expressed
# RELATIVE to the symbol's own average (scale-free), plus a TIME-aware "am I on pace to pass"
# read (days elapsed, return so far, pace vs the +2.5%/day plan, target remaining).
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config
from src.account.account_state import AccountState
from src.account import win_loss_features as WL
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.observation import observation_contract as OC


def test_block_in_contract():
    assert C.OBSERVATION_CONTRACT_VERSION == "v1.9.0" and C.OBS_TOTAL_SIZE == 526
    assert C.OBS_BLOCK_RECENT_CONTEXT == 8
    sl = OC.BLOCK_SLICES["recent_context"]
    assert sl.stop - sl.start == 8
    assert OC.FEATURE_NAMES[sl.start] == "week_avg_range_vs_typical"
    assert OC.FEATURE_NAMES[sl.stop - 1] == "challenge_target_remaining"


def test_recent_movement_is_relative_to_average():
    f = WL.recent_context_features(AccountState(starting_balance=100_000.0), load_ftmo_config(),
                                   week_avg=0.005, prev_day=0.010, prev2=0.005, today_sofar=0.0025,
                                   typical_range=0.008, days_elapsed=0)
    assert f.shape == (8,)
    assert abs(f[1] - 1.0) < 1e-6        # yesterday 2x the week avg -> 2/2 -> 1.0 (capped)
    assert abs(f[3] - 0.25) < 1e-6       # today 0.5x the week avg -> 0.5/2 -> 0.25
    assert abs(f[0] - 0.3125) < 1e-4     # week 0.005 vs typical 0.008 = 0.625 -> /2


def test_time_to_pass_pace():
    acc = AccountState(starting_balance=100_000.0)
    acc.mark_equity(105_000.0)           # +5% return
    f = WL.recent_context_features(acc, load_ftmo_config(), week_avg=0.005, prev_day=0.005,
                                   prev2=0.005, today_sofar=0.005, typical_range=0.008, days_elapsed=2)
    assert abs(f[5] - 0.05) < 1e-6       # return so far = +5%
    assert abs(f[6] - 0.5) < 1e-6        # day 2 at +5% = exactly on the +2.5%/day plan -> 0.5
    assert abs(f[7] - 0.05) < 1e-6       # 10% target - 5% = 5% remaining


def test_behind_and_ahead_of_pace():
    behind = AccountState(starting_balance=100_000.0); behind.mark_equity(100_500.0)   # +0.5% on day 3
    fb = WL.recent_context_features(behind, load_ftmo_config(), week_avg=0.005, prev_day=0.005,
                                    prev2=0.005, today_sofar=0.005, typical_range=0.008, days_elapsed=3)
    assert fb[6] < 0.5                   # behind the +2.5%/day plan
    ahead = AccountState(starting_balance=100_000.0); ahead.mark_equity(108_000.0)     # +8% on day 1
    fa = WL.recent_context_features(ahead, load_ftmo_config(), week_avg=0.005, prev_day=0.005,
                                    prev2=0.005, today_sofar=0.005, typical_range=0.008, days_elapsed=1)
    assert fa[6] > 0.5                   # ahead of plan


def test_in_env_obs_shape_and_days_elapsed():
    n = 3 * 1440
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    close = (1.10 + np.cumsum(np.random.default_rng(0).standard_normal(n)) * 0.0003).astype(np.float32)
    env = TradingEnv(np.zeros((n, C.N_INDICATORS_TOTAL), np.float32), close,
                     idx.values.astype("datetime64[ns]").astype(np.int64), AlphaRegistry(),
                     warmup=300, symbol="EURUSD", position_size=1.0)
    obs, _ = env.reset()
    assert obs.shape == (526,) and np.all(np.isfinite(obs))
    b = obs[OC.BLOCK_SLICES["recent_context"]]
    assert b.shape == (8,) and np.all((b >= -1.0) & (b <= 1.0))
    for _ in range(1500):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert env._days_elapsed >= 1        # crossed at least one midnight
