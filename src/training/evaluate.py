# =====================================================================
# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  READ-ONLY evaluation: run a policy greedily through the env, capture
#      introspection per step, and build the Policy Doctor report. This NEVER
#      affects training (eval-safe) -- it only observes.
# WHERE src/training/evaluate.py
# HOW  policy_fn(obs) -> (logits(4,), value). Greedy action = argmax(logits).
#      Introspection + Policy Doctor are computed from what was visited.
# DEPENDS_ON src/interpret/policy_introspector.py, src/barbershop/policy_doctor.py
# USED_BY src/training/trainer.py (eval callback), src/jarvis/* (Phase 2), tests.
# CHANGE_NOTES(IRAC): I: diagnostics must not change training behaviour. R:
#   operator trainer/eval separation 2026-06-21. A: pure read-only rollout +
#   introspection + Policy Doctor. C: trustworthy 'what is the policy thinking'
#   without contaminating the policy.
# =====================================================================
"""Read-only policy evaluation: rollout + introspection + Policy Doctor."""
from __future__ import annotations
import numpy as np
from src.interpret.policy_introspector import introspect
from src.barbershop.policy_doctor import build_report


def evaluate_policy(env, policy_fn, *, max_steps: int | None = None,
                    do_introspect: bool = True, window: int | None = None) -> dict:
    """Run policy_fn greedily through env (READ-ONLY). policy_fn(obs)->(logits,value)."""
    obs, _ = env.reset()
    start = env.ptr
    actions, rewards, equity, records = [], [], [], []
    steps = 0
    while True:
        logits, _ = policy_fn(obs)
        if do_introspect:
            records.append(introspect(policy_fn, obs, bar_index=env.ptr))
        a = int(np.argmax(np.asarray(logits).ravel()))
        obs, r, term, trunc, info = env.step(a)
        actions.append(a); rewards.append(float(r)); equity.append(info["equity"])
        steps += 1
        if term or trunc or (max_steps and steps >= max_steps):
            break
    n = len(actions)
    report = build_report(env.alpha_matrix[start:start + n], np.asarray(actions),
                          env.close[start:start + n], env.occupancy, window=window,
                          introspection_records=records if do_introspect else None)
    return {"actions": np.asarray(actions), "rewards": np.asarray(rewards),
            "equity": np.asarray(equity), "introspection": records,
            "policy_doctor": report, "n_steps": n}
