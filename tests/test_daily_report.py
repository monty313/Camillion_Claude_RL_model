# The day-by-day FTMO report: per-day +2.5%-of-initial target + 4% trailing-DD status.
import numpy as np
import pandas as pd
from config import constants as C
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.training.daily_report import daily_report, format_daily_report


def _env(close):
    n = len(close)
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    reg = AlphaRegistry(); register_all(reg)
    return TradingEnv(ind, np.asarray(close, np.float32),
                      idx.values.astype("datetime64[ns]").astype(np.int64), reg, symbol="EURUSD")


def test_report_columns_and_summary_keys():
    env = _env(1.10 + np.linspace(0.0, 0.02, 3000))     # ~2 days, gentle uptrend
    rows, summary = daily_report(env, policy=None)       # HOLD baseline
    assert rows, "expected at least one day"
    for r in rows:
        for k in ("day", "date", "day_pnl_pct", "passed_target", "trailing_dd_pct",
                  "within_trailing", "daily_loss_pct", "breached", "cum_pnl_pct"):
            assert k in r
        assert isinstance(r["passed_target"], bool) and isinstance(r["within_trailing"], bool)
    for k in ("days", "days_passed_target", "days_within_trailing", "breaches",
              "final_cum_pct", "daily_target_pct", "trailing_pct", "challenge_passed"):
        assert k in summary
    assert summary["daily_target_pct"] == 2.5 and summary["trailing_pct"] == 4.0
    # HOLD never trades -> 0 target days, 0 breaches, all days within the trailing wall
    assert summary["days_passed_target"] == 0 and summary["breaches"] == 0
    assert summary["days_within_trailing"] == summary["days"]
    assert "DAY-BY-DAY FTMO REPORT" in format_daily_report(rows, summary)


def test_report_reflects_a_trading_policy():
    env = _env(1.10 + np.linspace(0.0, 0.02, 3000))      # uptrend
    pol = lambda obs: (np.array([0.0, 9.0, 0.0, 0.0], dtype=np.float32), 0.0)   # always BUY
    rows, summary = daily_report(env, policy=pol)
    assert any(r["day_pnl_pct"] != 0.0 for r in rows)    # the policy actually moved equity
    assert isinstance(summary["challenge_passed"], bool)
    # every day's pass/within flags are real booleans derived from the numbers
    for r in rows:
        assert r["passed_target"] == (r["day_pnl_pct"] >= 2.5)
        assert r["within_trailing"] == (r["trailing_dd_pct"] < 4.0)
