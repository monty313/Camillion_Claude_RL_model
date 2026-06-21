# Phase 2: Day Replay, Trade Autopsy, Signal Doctor produce correct structures.
import numpy as np
from src.barbershop import day_replay, trade_autopsy, signal_doctor
from src.account.trade_history import ClosedTrade


def test_day_replay():
    n = 40
    eq = 100000 + np.cumsum(np.random.randn(n) * 30)
    rep = day_replay.build_replay(np.arange(n), np.random.uniform(-1, 1, n), eq,
                                  np.random.choice([-1, 0, 1], n))
    assert len(rep["bars"]) == n and "max_drawdown_pct" in rep["summary"]
    assert rep["summary"]["bars"] == n


def test_trade_autopsy():
    am = np.random.choice([-1, 0, 1], size=(30, 5)).astype(float)
    net = np.sign(am.sum(1))
    tr = ClosedTrade(pnl=-50.0, is_win=False, bar_index=12, direction=-1)
    au = trade_autopsy.autopsy(tr, alpha_matrix=am, net_signal=net,
                               occupancy=np.array([1, 1, 1, 1, 1]))
    assert au["outcome"] == "loss" and au["direction"] == -1
    assert "aligned_with_consensus" in au and au["n_active"] >= 0


def test_signal_doctor():
    close = 100 + np.cumsum(np.random.randn(120) * 0.1)
    am = np.random.choice([-1, 0, 1], size=(120, 5)).astype(float)
    sd = signal_doctor.report(am, close, np.array([1, 1, 1, 1, 0]), window=40)
    assert len(sd["alphas"]) == 4 and 0.0 <= sd["conflict_rate"] <= 1.0
