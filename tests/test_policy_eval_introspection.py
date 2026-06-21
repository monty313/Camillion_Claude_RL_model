# Phase 1: the introspector + Policy Doctor plug into eval, read-only, against a
# mock policy (no torch). Proves the eval harness works end-to-end on the env.
import numpy as np
import pandas as pd
from config import constants as C
from src.observation import observation_contract as OC
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.strategies.examples import register_examples
from src.training.evaluate import evaluate_policy


def _cache(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 09:00", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    ind = rng.standard_normal((n, C.N_INDICATORS_TOTAL)).astype(np.float32)
    return ind, close.astype(np.float32), idx.values.astype("datetime64[ns]").astype(np.int64)


def _mock_policy(obs):
    # depends on the alpha block so introspection should attribute to 'alpha'
    sl = OC.BLOCK_SLICES["alpha_values"]
    logits = np.zeros(C.N_ACTIONS)
    logits[C.ACTION_BUY] = float(np.asarray(obs)[sl].sum())
    logits[C.ACTION_SELL] = -float(np.asarray(obs)[sl].sum())
    return logits, 0.0


def test_eval_harness_runs_and_is_read_only():
    ind, close, t = _cache()
    reg = AlphaRegistry(); register_examples(reg)
    env = TradingEnv(ind, close, t, reg, warmup=5, window=200, random_window=False)
    out = evaluate_policy(env, _mock_policy, max_steps=150, window=60)
    assert out["n_steps"] == 150
    assert len(out["introspection"]) == 150
    r0 = out["introspection"][0]
    assert len(r0.action_probs) == 4 and abs(sum(r0.action_probs) - 1.0) < 1e-6
    assert "alpha" in r0.group_importance
    rep = out["policy_doctor"]
    assert "leader_chasing" in rep and "scoreboard" in rep
