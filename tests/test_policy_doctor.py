# Phase 1: Policy Doctor flags leader-chasing for a copycat policy and clears a
# contextual one; scoreboard + policy-accuracy structure present.
import numpy as np
from config import constants as C
from src.barbershop.policy_doctor import build_report, render


def _setup(seed):
    rng = np.random.default_rng(seed); T = 320
    close = 100 + np.cumsum(rng.standard_normal(T))
    a0 = np.zeros(T)
    a0[:T - 3] = np.sign(close[3:] - close[:T - 3])      # leader: predicts 3-bar move
    a1 = rng.choice([-1, 0, 1], T).astype(float)
    a2 = rng.choice([-1, 0, 1], T).astype(float)
    alphas = np.stack([a0, a1, a2], axis=1)
    return close, alphas, np.array([1, 1, 1]), a0, rng


def test_copycat_is_flagged_as_leader_chasing():
    close, alphas, occ, a0, _ = _setup(1)
    actions = np.where(a0 > 0, C.ACTION_BUY, np.where(a0 < 0, C.ACTION_SELL, C.ACTION_HOLD))
    rep = build_report(alphas, actions, close, occ, window=80, min_samples=5)
    assert rep["leader_chasing"]["leader_follow_rate"] > 0.8
    assert rep["leader_chasing"]["flag"] is True
    assert isinstance(render(rep), str) and "LEADER-CHASING" in render(rep)


def test_contextual_policy_not_flagged():
    close, alphas, occ, a0, rng = _setup(2)
    actions = rng.integers(0, 4, size=len(close))         # independent of the leader
    rep = build_report(alphas, actions, close, occ, window=80, min_samples=5)
    assert rep["leader_chasing"]["flag"] is False


def test_report_structure():
    close, alphas, occ, a0, _ = _setup(3)
    actions = np.zeros(len(close), int)
    rep = build_report(alphas, actions, close, occ, window=80)
    assert set(rep["scoreboard"].keys()) == {1, 3, 10}
    assert "policy_outperforms_best" in rep["best_single_alpha"]
    assert "leader_follow_rate" in rep["leader_chasing"]
