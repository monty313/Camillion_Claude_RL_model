# Bug #1 fix (2026-06-27): the daily DD gauge the bot SEES must use LIVE equity (open losses included) so
# it AGREES with the actual daily-loss breach in ftmo_rules. Before, it used closed-trade PnL only, so it
# read "safe" while a losing trade was open -> the bot was blind to the #1 cause of FTMO failure.
import numpy as np
from config.ftmo_config import load_ftmo_config
from src.account.account_state import AccountState
from src.account.win_loss_features import daily_features
from src.risk.ftmo_rules import FTMORules

# daily_features layout: [win_rate, pnl_frac(closed), dd_used, target_progress, risk_remaining, trades, streak]
_DD_USED, _RISK_REMAINING = 2, 4


def test_daily_gauge_reflects_an_OPEN_loss():
    cfg = load_ftmo_config()
    acc = AccountState(starting_balance=100_000.0)   # day_start_balance defaults to 100k
    acc.daily_realized_pnl = 0.0                       # nothing CLOSED yet
    acc.mark_equity(97_000.0)                          # but a -3% open loss is live
    f = daily_features(acc, cfg)
    # 3% live loss against a 5% daily limit -> 0.6 of the budget used (was 0.0 before the fix)
    assert f[_DD_USED] > 0.5, "daily DD gauge must reflect the LIVE open loss"
    assert f[_RISK_REMAINING] < 0.5


def test_daily_gauge_agrees_with_the_breach_at_the_wall():
    cfg = load_ftmo_config()
    rules = FTMORules(cfg)
    acc = AccountState(starting_balance=100_000.0)
    acc.daily_realized_pnl = 0.0
    acc.mark_equity(95_000.0)                          # exactly the 5% daily wall, all from an open loss
    assert rules.daily_drawdown_breached(acc) is True
    f = daily_features(acc, cfg)
    assert f[_DD_USED] >= 1.0, "gauge reads 'at the wall' exactly when the engine breaches"


def test_no_phantom_dd_when_flat_or_up():
    cfg = load_ftmo_config()
    acc = AccountState(starting_balance=100_000.0)
    acc.mark_equity(101_000.0)                         # up +1%, no drawdown
    f = daily_features(acc, cfg)
    assert f[_DD_USED] == 0.0 and f[_RISK_REMAINING] == 1.0
