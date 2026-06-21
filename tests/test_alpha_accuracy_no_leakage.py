# Phase 1: per-alpha 1/3/10-bar accuracy is leak-free; aggregates behave.
import numpy as np
from src.signals.alpha_accuracy import per_alpha_accuracy, aggregate_reliability


def test_per_alpha_no_future_leakage():
    rng = np.random.default_rng(3); T, nA = 400, 4
    close = 100 + np.cumsum(rng.standard_normal(T))
    alphas = rng.choice([-1, 0, 1], size=(T, nA)).astype(float)
    acc, _ = per_alpha_accuracy(alphas, close, window=60)
    for t in (50, 150, 300):
        c2, a2 = close.copy(), alphas.copy()
        c2[t + 1:] += rng.standard_normal(T - t - 1) * 8
        a2[t + 1:] = rng.choice([-1, 0, 1], size=(T - t - 1, nA))
        acc2, _ = per_alpha_accuracy(a2, c2, window=60)
        for h in (1, 3, 10):
            assert np.allclose(acc[h][:t + 1], acc2[h][:t + 1]), f"leak h={h} t={t}"


def test_deterministic_and_aggregates():
    up = np.arange(60, dtype=float)
    a, _ = per_alpha_accuracy(np.ones((60, 1)), up, window=100)
    assert a[1][20, 0] == 1.0 and a[10][40, 0] == 1.0
    rng = np.random.default_rng(9); T = 300
    close = 100 + np.cumsum(rng.standard_normal(T))
    alphas = rng.choice([-1, 0, 1], size=(T, 3)).astype(float)
    acc, cnt = per_alpha_accuracy(alphas, close, window=60)
    agg = aggregate_reliability(acc, cnt, np.array([1, 1, 1]), min_samples=5)
    assert set(agg) == {1, 3, 10}
    assert agg[3]["best"][-1] >= agg[3]["mean"][-1]      # best >= mean
    assert 0.0 <= agg[3]["dispersion"][-1] < 1.0
