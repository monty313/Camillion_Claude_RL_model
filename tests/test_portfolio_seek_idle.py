# SEEK-THE-TARGET vs HIDE rebalance (2026-06-28, operator). The PortfolioEnv reward now (a) gives a DENSE
# reward for NEW progress toward the +2.5%/day target (high-water-mark) so the bot SEEKS profit, and
# (b) penalises a day that ends with ZERO trades so "hiding" isn't free. These tests isolate each term
# (day-score + alpha-shaping OFF) and check it has the intended EFFECT. Parity to the JAX env is covered
# separately in jax_tpu/tests/test_jax_portfolio_parity.py.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import FTMOConfig
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _env(prices, *, seek=0.10, idle=0.02):
    # 4 bars over 2 days; isolate seek/idle by turning day-score + alpha-shaping + two-phase OFF.
    ts = ["2026-03-02 00:00", "2026-03-02 12:00", "2026-03-03 00:00", "2026-03-03 12:00"]
    idx = pd.to_datetime(ts).values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((4, C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, np.asarray(prices, dtype=np.float64), idx)}
    cfg = FTMOConfig(two_phase_enabled=False, alpha_reward_enabled=False,
                     day_pass_reward=0.0, day_fail_penalty=0.0,
                     target_seek_weight=seek, idle_day_penalty=idle)
    return PortfolioEnv(sd, _reg, warmup=0, cfg=cfg)


def _total(env, actions):
    tot = 0.0
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        tot += r
        if term or trunc:
            break
    return tot


SEQ = [C.ACTION_BUY, C.ACTION_HOLD, C.ACTION_HOLD]
HOLD_SEQ = [C.ACTION_HOLD, C.ACTION_HOLD, C.ACTION_HOLD]


def test_seek_rewards_progress_toward_target_even_on_a_failed_day():
    """Day rises to +1.5% (below the +2.5% target -> a FAILED day). The seek term should STILL reward the
    60%-of-target progress made -> total reward is higher WITH seek than without (it rewards SEEKING).
    BUY then CLOSE before midnight so only ONE day's progress is measured (no carried-position double count)."""
    prices = [1.00, 1.015, 1.015, 1.015]   # +1.5% = 0.6 of the +2.5% target
    seq = [C.ACTION_BUY, C.ACTION_CLOSE, C.ACTION_HOLD]
    on = _total(_env(prices, seek=0.10), seq)
    off = _total(_env(prices, seek=0.0), seq)
    gain = on - off
    # ~ seek_weight * progress = 0.10 * 0.6 = 0.06
    assert 0.03 < gain < 0.10, f"seek should add ~0.06 for 60% progress, got {gain:.4f}"


def test_idle_day_is_penalised_relative_to_no_idle():
    """A full day with ZERO trades (all HOLD) should cost idle_day_penalty at midnight -> total reward is
    LOWER with the idle penalty on than off (hiding is no longer free)."""
    prices = [1.00, 1.00, 1.00, 1.00]
    on = _total(_env(prices, idle=0.02), HOLD_SEQ)
    off = _total(_env(prices, idle=0.0), HOLD_SEQ)
    diff = on - off
    assert abs(diff + 0.02) < 5e-3, f"zero-trade day should cost ~-0.02, got {diff:.4f}"


def test_trading_avoids_the_idle_penalty():
    """A day that DID trade must NOT get the idle penalty (only zero-trade days are penalised)."""
    prices = [1.00, 1.00, 1.00, 1.00]
    traded = _total(_env(prices, idle=0.02), SEQ)        # BUY -> traded
    idled = _total(_env(prices, idle=0.02), HOLD_SEQ)    # all HOLD -> idle
    # the traded day avoids the -0.02 idle hit (minus a tiny transaction cost), so it ends up higher
    assert traded > idled, f"a traded day ({traded:.4f}) should beat an idle day ({idled:.4f})"
