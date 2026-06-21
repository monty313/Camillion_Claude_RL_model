# Phase 1: introspection captures pi(a|s)+value+entropy; block ablation
# attributes importance to the block the (mock) policy actually depends on.
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC
from src.interpret.policy_introspector import introspect


def make_block_policy(block_name, strength=10.0):
    sl = OC.BLOCK_SLICES[block_name]
    W = np.zeros((C.OBS_TOTAL_SIZE, C.N_ACTIONS))
    W[sl, C.ACTION_BUY] = strength
    def policy(obs):
        obs = np.asarray(obs, dtype=np.float64).ravel()
        return obs @ W, float(obs[sl].sum())
    return policy


def test_action_distribution_valid():
    r = introspect(make_block_policy("indicators"), np.ones(C.OBS_TOTAL_SIZE, np.float32))
    assert len(r.action_probs) == 4 and abs(sum(r.action_probs) - 1.0) < 1e-6
    assert 0 <= r.chosen_action < 4 and r.entropy >= 0.0
    assert r.chosen_action_name in C.ACTIONS


def test_block_ablation_attributes_to_right_group():
    obs = np.ones(C.OBS_TOTAL_SIZE, np.float32)
    r_ind = introspect(make_block_policy("indicators"), obs)
    assert r_ind.group_importance["raw_indicators"] > 0.9
    r_alpha = introspect(make_block_policy("alpha_values"), obs)
    assert r_alpha.group_importance["alpha"] > 0.9
    r_acct = introspect(make_block_policy("account_daily"), obs)
    assert r_acct.group_importance["account"] > 0.9
    r_port = introspect(make_block_policy("portfolio"), obs)
    assert r_port.group_importance["portfolio"] > 0.9
