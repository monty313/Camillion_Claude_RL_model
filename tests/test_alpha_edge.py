# Diagnostic sanity: per_alpha_edge runs and returns one row per slot with finite numbers.
import numpy as np
from src.barbershop.alpha_edge import per_alpha_edge, edge_table

def test_per_alpha_edge_runs():
    rng = np.random.default_rng(0); T = 2000
    close = 100 + np.cumsum(rng.standard_normal(T) * 0.05)
    am = np.zeros((T, 15), dtype=np.float32)
    am[:, 0] = np.sign(np.diff(close, prepend=close[0]))   # a 'perfect' lookback signal -> positive hit
    am[:, 1] = rng.choice([-1, 0, 1], size=T)              # noise
    rep = per_alpha_edge(am, close, horizon=60)
    assert len(rep) == 15
    assert all(np.isfinite(r["edge"]) and 0.0 <= r["hit_rate"] <= 1.0 for r in rep)
    assert isinstance(edge_table(rep), str)
