# Test 3: signal summary percentages are mathematically correct + zero-safe.
import numpy as np
from src.signals.signal_summary import summarize, net_balance


def test_percentages_correct():
    av = np.array([1, 1, 1, -1, 0, 0] + [0] * 58, np.float32)  # 3 buy, 1 sell
    om = np.array([1, 1, 1, 1, 0, 0] + [0] * 58, np.float32)   # 4 assigned
    s = summarize(av, om)
    assert np.allclose(s, [0.75, 0.25, 1.0, 0.5])


def test_all_sell():
    av = np.array([-1, -1] + [0] * 62, np.float32)
    om = np.array([1, 1] + [0] * 62, np.float32)
    assert np.allclose(summarize(av, om), [0.0, 1.0, 1.0, -1.0])


def test_zero_division_safe():
    assert np.allclose(summarize(np.zeros(64, np.float32), np.zeros(64, np.float32)),
                       [0, 0, 0, 0])


def test_active_pct_is_over_assigned():
    av = np.array([1, -1, 0, 0] + [0] * 60, np.float32)   # 2 active
    om = np.array([1, 1, 1, 1] + [0] * 60, np.float32)    # 4 assigned
    s = summarize(av, om)
    assert abs(s[2] - 0.5) < 1e-6           # active/assigned = 2/4
    assert abs(net_balance(av)) < 1e-6      # (1-1)/2 = 0
