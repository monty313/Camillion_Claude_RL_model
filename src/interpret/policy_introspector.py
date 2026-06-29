# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Capture the policy's decision in the most fundamental way: action
#      distribution pi(a|s), value V(s), entropy, and WHY via block-ablation
#      saliency (zero a block, measure the shift in the action distribution).
#      Read-only telemetry -- NEVER fed back into the observation, never reward.
# WHERE src/interpret/policy_introspector.py
# HOW  policy is any callable: policy(obs(513,)) -> (logits(4,), value).
#      Ablation is model-agnostic (works with the SB3 policy or a mock); a
#      gradient-saliency path can be added when torch is present.
# DEPENDS_ON: config/constants.py, src/observation/observation_contract.py, numpy
# USED_BY: src/barbershop/policy_doctor.py, src/jarvis/*, tests.
"""PolicyIntrospector: action dist + value + entropy + per-block ablation saliency."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC

# Conceptual groups for saliency (indicators vs alpha vs account vs portfolio).
GROUPS = {
    "raw_indicators": ("indicators",),
    "alpha": ("alpha_values", "alpha_mask", "alpha_summary",
              "signal_memory", "signal_accuracy"),
    "account": ("account_daily", "account_episode"),
    "portfolio": ("portfolio",),
    "time": ("time",),
}


def _softmax(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    e = np.exp(x - x.max())
    return e / e.sum()


def _entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


@dataclass
class IntrospectionRecord:
    action_probs: list           # pi(a|s) over {HOLD,BUY,SELL,CLOSE}
    chosen_action: int
    chosen_action_name: str
    value: float                 # V(s)
    entropy: float               # decisiveness (low = sure)
    block_importance: dict       # per contract block: normalized L1 shift if zeroed
    group_importance: dict       # grouped: raw_indicators/alpha/account/portfolio/time
    bar_index: int = -1


def introspect(policy, obs, ablate: bool = True, bar_index: int = -1) -> IntrospectionRecord:
    """Introspect one decision. `policy(obs)` must return (logits(4,), value)."""
    obs = np.asarray(obs, dtype=np.float32).ravel()
    logits, value = policy(obs)
    probs = _softmax(logits)
    chosen = int(np.argmax(probs))
    block_imp = {}
    if ablate:
        for name, sl in OC.BLOCK_SLICES.items():
            o2 = obs.copy(); o2[sl] = 0.0
            l2, _ = policy(o2)
            block_imp[name] = float(np.abs(_softmax(l2) - probs).sum())
        tot = sum(block_imp.values()) or 1.0
        block_imp = {k: v / tot for k, v in block_imp.items()}   # fractions
    group_imp = {g: float(sum(block_imp.get(b, 0.0) for b in blocks))
                 for g, blocks in GROUPS.items()}
    return IntrospectionRecord(
        action_probs=[float(x) for x in probs], chosen_action=chosen,
        chosen_action_name=C.ACTIONS[chosen], value=float(np.asarray(value).ravel()[0]),
        entropy=_entropy(probs), block_importance=block_imp,
        group_importance=group_imp, bar_index=bar_index)
