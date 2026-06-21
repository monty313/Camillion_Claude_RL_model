# Test 5: signal accuracy has NO look-ahead leakage (out[t] ignores bars > t).
import numpy as np
from src.signals.signal_accuracy import rolling_accuracy, accuracy_features


def test_no_future_leakage():
    rng = np.random.default_rng(1)
    T = 300
    close = np.cumsum(rng.standard_normal(T)) + 100.0
    net = rng.choice([-1, 0, 1], size=T).astype(float)
    full = accuracy_features(net, close, window=80)   # (T, 2)
    for t in (10, 100, 200, 250):
        c2, n2 = close.copy(), net.copy()
        c2[t + 1:] += rng.standard_normal(T - t - 1) * 10.0   # mutate the FUTURE
        n2[t + 1:] = rng.choice([-1, 0, 1], size=T - t - 1)
        f2 = accuracy_features(n2, c2, window=80)
        assert np.allclose(full[:t + 1], f2[:t + 1]), f"leak at t={t}"


def test_known_accuracy_values():
    up = np.arange(20, dtype=float)               # strictly increasing
    a1 = rolling_accuracy(np.ones(20), up, window=100, horizon=1)
    assert a1[5] == 1.0 and a1[19] == 1.0         # +1 signal always correct
    down = np.arange(20, 0, -1, dtype=float)       # strictly decreasing
    a1b = rolling_accuracy(np.ones(20), down, window=100, horizon=1)
    assert a1b[5] == 0.0                           # +1 signal always wrong


def test_three_bar_horizon_uses_only_past():
    rng = np.random.default_rng(7)
    T = 120
    close = np.cumsum(rng.standard_normal(T)) + 50.0
    net = rng.choice([-1, 0, 1], size=T).astype(float)
    a3 = rolling_accuracy(net, close, window=50, horizon=3)
    # changing bars strictly after t must not change a3[t]
    t = 60
    c2 = close.copy(); c2[t + 1:] += 5.0
    a3b = rolling_accuracy(net, c2, window=50, horizon=3)
    assert np.allclose(a3[:t + 1], a3b[:t + 1])
