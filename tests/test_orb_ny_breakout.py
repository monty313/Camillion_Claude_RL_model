# ORB NY-open breakout alpha (INDICES only) + the NY-session reward bonus (paid on CLOSED-in-profit
# session P&L, ONLY if the day passes >= +2.5% of initial; erased on a failed/breached day).
import numpy as np
import pandas as pd
from config import constants as C
from config import variables as V
from config.ftmo_config import load_ftmo_config
from src.strategies.orb_ny_breakout_indices_alpha import OrbNyBreakoutIndicesAlpha
from src.strategies.context import MarketContext
from src.strategies.registry import AlphaRegistry
from src.env.trading_env import TradingEnv


def _ctx(close, mod, symbol="US30", sma200=100.0):
    return MarketContext(close=close, indicators={"30m__bb200_dev1.0_middle": sma200},
                         symbol=symbol, minute_of_day=mod)


# ---------------- ORB alpha logic ----------------
def test_orb_is_index_only():
    a = OrbNyBreakoutIndicesAlpha(); a.reset()
    assert a.compute_signal(_ctx(105, 820, symbol="EURUSD")) == 0   # not an index -> never fires
    assert a.compute_signal(_ctx(105, 820, symbol="GBPUSD")) == 0


def test_orb_breakout_long_short_and_trend_filter():
    a = OrbNyBreakoutIndicesAlpha(); a.reset()
    for c, m in [(100, 570), (99, 600), (101, 700), (100, 800)]:    # opening range -> [99, 101]
        assert a.compute_signal(_ctx(c, m)) == 0                    # accumulating
    assert a.compute_signal(_ctx(102, 820, sma200=100.0)) == 1      # break up + above trend
    assert a.compute_signal(_ctx(98, 830, sma200=100.0)) == -1      # break down + below trend
    assert a.compute_signal(_ctx(100, 840, sma200=100.0)) == 0      # back inside range -> 0
    assert a.compute_signal(_ctx(102, 850, sma200=103.0)) == 0      # break up but BELOW trend -> 0
    assert a.compute_signal(_ctx(102, 950, sma200=100.0)) == 0      # outside the 13:30-15:30 window


def test_orb_resets_each_new_day():
    a = OrbNyBreakoutIndicesAlpha(); a.reset()
    a.compute_signal(_ctx(100, 600)); a.compute_signal(_ctx(110, 700))   # range built on day 1
    assert a.compute_signal(_ctx(200, 50)) == 0                    # midnight wrap -> range reset
    assert a.compute_signal(_ctx(200, 820)) == 0                   # no range built this day -> 0


# ---------------- NY reward bonus ----------------
def _index_env(symbol="US30", n=400):
    idx = pd.date_range("2026-03-02 13:00", periods=n, freq="1min")   # bar i -> minute-of-day 780+i
    close = np.full(n, 100.0, np.float32)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    return TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64),
                      AlphaRegistry(), warmup=0, symbol=symbol, position_size=100000.0,
                      cfg=load_ftmo_config())


def test_ny_bonus_qualifies_and_pays_only_if_day_passes():
    env = _index_env(); env.reset()
    assert env._is_index
    target = 0.025 * 100_000.0                       # daily target $ (2.5% of initial)
    env.ptr = 60                                     # minute-of-day 840 -> inside NY 13:30-15:30
    env._ny_start_realized = None; env.acc.daily_realized_pnl = 0.0
    env._ny_qualify()                                # records the session-start baseline
    assert env._ny_start_realized == 0.0 and not env._ny_half_qualified
    env.acc.daily_realized_pnl = target              # closed +2.5% in profit during the session
    env._ny_qualify()
    assert env._ny_half_qualified and env._ny_full_qualified
    # day PASSED (>= +2.5% closed) -> both bonuses paid
    assert abs(env._ny_day_end_bonus()
               - (V.FTMO_NY_HALF_TARGET_BONUS + V.FTMO_NY_FULL_TARGET_BONUS)) < 1e-9
    # day FAILED (< +2.5% at day end) -> bonus erased
    env.acc.daily_realized_pnl = 1_500.0
    assert env._ny_day_end_bonus() == 0.0


def test_ny_bonus_not_for_nonindex_or_session_loss():
    pair = _index_env(symbol="EURUSD"); pair.reset()
    pair.ptr = 60; pair.acc.daily_realized_pnl = 5_000.0; pair._ny_qualify()
    assert not pair._ny_half_qualified               # not an index -> no qualify, no bonus
    assert pair._ny_day_end_bonus() == 0.0
    env = _index_env(); env.reset(); env.ptr = 60
    env._ny_start_realized = None; env.acc.daily_realized_pnl = 0.0; env._ny_qualify()
    env.acc.daily_realized_pnl = -500.0; env._ny_qualify()   # session closed at a LOSS
    assert not env._ny_half_qualified
