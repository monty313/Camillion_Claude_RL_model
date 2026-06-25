# Regression guard (2026-06-25): realized PnL must be banked EXACTLY ONCE.
# Bug: env.step() added `realized` to balance/daily/episode AND then called
# record_close(), which adds the same three again -> every closed trade moved the
# account by 2x its true PnL, corrupting equity, reward, and every FTMO check.
# Fix: record_close() is the SINGLE source of truth; the manual += lines were removed.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config
from src.account.account_state import AccountState
from src.account.trade_history import TradeHistory
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry


def test_record_close_moves_balance_once():
    """record_close(acc, +5000) moves balance/daily/episode by EXACTLY +5000."""
    acc = AccountState(starting_balance=100_000.0)
    th = TradeHistory()
    th.record_close(acc, 5_000.0, bar_index=0)
    assert acc.balance == 105_000.0
    assert acc.daily_realized_pnl == 5_000.0
    assert acc.episode_realized_pnl == 5_000.0
    # a loss too
    th.record_close(acc, -2_000.0, bar_index=1)
    assert acc.balance == 103_000.0
    assert acc.daily_realized_pnl == 3_000.0


def test_env_round_trip_banks_one_times_gross():
    """A long that gains a gross +$10k (1 lot, +0.10 move), with cost OFF, must bank
    ~+$10k ONCE. The old double-count bug banked ~+$20k."""
    prices = [100.0, 100.10, 100.10, 100.10]
    idx = pd.date_range("2026-03-02 09:00", periods=len(prices), freq="1min")
    close = np.asarray(prices, dtype=np.float32)
    ind = np.zeros((len(prices), C.N_INDICATORS_TOTAL), dtype=np.float32)
    env = TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64),
                     AlphaRegistry(), warmup=0, position_size=100000.0,
                     cost_frac=0.0, cfg=load_ftmo_config())
    env.reset()
    env.step(C.ACTION_BUY)      # enter long at 100.00
    env.step(C.ACTION_CLOSE)    # close at 100.10 -> gross +10,000
    pnl = env.acc.daily_realized_pnl
    # ~10,000 within float32 price representation; the bug would give ~20,000.
    assert 9_900 < pnl < 10_010, f"expected ~10k banked once, got {pnl}"
    assert pnl < 15_000, "PnL banked more than 1x -> double-count regression!"
    assert abs(env.acc.balance - (100_000.0 + pnl)) < 1e-3
    # episode realized PnL must equal the balance change (one accounting path)
    assert abs(env.acc.episode_realized_pnl - (env.acc.balance - 100_000.0)) < 1e-3
