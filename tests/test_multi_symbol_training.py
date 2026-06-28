# Multi-symbol training wiring: ONE policy across MANY symbols. make_multi_symbol_vec_env spreads
# workers round-robin over the symbols, each tagged with its symbol + per-asset calibrated size,
# so one bot learns to trade everything. (Needs gymnasium + SB3 -> skips cleanly otherwise.)
try:
    import gymnasium  # noqa
    import stable_baselines3  # noqa
    _HAVE = True
except Exception:
    _HAVE = False
import numpy as np
import pandas as pd
from config import constants as C
from config import training_speed_config as TS
from config import asset_specs as A
from src.strategies.registry import AlphaRegistry


def _data(n=600, seed=0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = (100 + np.cumsum(np.random.default_rng(seed).standard_normal(n)) * 0.05).astype(np.float32)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    return ind, close, idx.values.astype("datetime64[ns]").astype(np.int64)


def test_multi_symbol_vec_env_spreads_symbols_and_sizes():
    if not _HAVE:
        print("SKIP test_multi_symbol_training: needs gymnasium + stable-baselines3 (run in Colab)")
        return
    from src.training.vector_env_factory import make_multi_symbol_vec_env
    old = TS.VEC_ENV_BACKEND
    TS.VEC_ENV_BACKEND = "dummy"        # in-process so we can inspect the workers
    try:
        sd = {"EURUSD": _data(seed=1), "US30": _data(seed=2), "XAUUSD": _data(seed=3)}
        venv = make_multi_symbol_vec_env(sd, AlphaRegistry, n_envs=6, warmup=210)
        try:
            syms = [e._env.symbol for e in venv.envs]
            assert sorted(set(syms)) == ["EURUSD", "US30", "XAUUSD"]   # all 3 present
            for s in ("EURUSD", "US30", "XAUUSD"):
                assert syms.count(s) == 2                              # 6 workers / 3 = 2 each
            for e in venv.envs:                                       # per-asset size + conversion
                te = e._env
                assert te.value_per_point == A.SPECS[te.symbol].contract_size
                assert abs(te.position_size - A.calibrated_position_size(te.symbol)) < 1e-6
            obs = venv.reset()
            assert obs.shape == (6, C.OBS_TOTAL_SIZE)                  # one stacked 499-obs per worker
        finally:
            venv.close()
    finally:
        TS.VEC_ENV_BACKEND = old
