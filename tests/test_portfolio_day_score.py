# Per-day consistency scoring (2026-06-28, operator): a "WON day" = the day ENDS at >= +2.5% of INITIAL
# (scored at midnight, AFTER any give-back -- so banking +2.5% then leaking it back counts as a FAIL).
# A won day adds day_pass_reward; a failed day subtracts day_fail_penalty. These tests force a known
# end-of-day equity and check the midnight-step reward.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import FTMOConfig
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _env(day1_close, *, pass_r=0.5, fail_p=0.5):
    # 2 bars/day over 2 days; price ~1.0, position_size defaults to 100k -> a +0.03 move = +3% of the account.
    ts = ["2026-03-02 00:00", "2026-03-02 12:00", "2026-03-03 00:00", "2026-03-03 12:00"]
    px = [1.00, day1_close, day1_close, day1_close]
    idx = pd.to_datetime(ts).values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((4, C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, np.asarray(px, dtype=np.float64), idx)}
    cfg = FTMOConfig(two_phase_enabled=False, alpha_reward_enabled=False,   # isolate the day score
                     day_pass_reward=pass_r, day_fail_penalty=fail_p)
    return PortfolioEnv(sd, _reg, warmup=0, cfg=cfg)


def _rewards(env, actions):
    out = []
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        out.append(r)
        if term or trunc:
            break
    return out


# BUY at day1 00:00, then HOLD -> the second step advances into day2 00:00 = the MIDNIGHT scoring step (index 1)
SEQ = [C.ACTION_BUY, C.ACTION_HOLD, C.ACTION_HOLD]


def test_won_day_is_rewarded():
    rs = _rewards(_env(1.03), SEQ)          # day1 ends +3% (>= +2.5%)
    assert rs[1] > 0.4, f"a WON day should add ~+0.5 at midnight, got {rs[1]}"


def test_failed_day_is_penalised():
    rs = _rewards(_env(1.005), SEQ)         # day1 ends +0.5% (< +2.5%)
    assert rs[1] < -0.4, f"a FAILED day should subtract ~0.5 at midnight, got {rs[1]}"


def test_flat_day_also_fails():
    rs = _rewards(_env(1.00), SEQ)          # day1 ends flat -> still a fail (HOLD doesn't escape the penalty)
    assert rs[1] < -0.4
