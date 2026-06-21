# =====================================================================
# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Barbershop "Policy Doctor": show what the policy is doing vs the alphas,
#      and PROVE it isn't just copying the best-recent-accuracy alpha.
# WHERE src/barbershop/policy_doctor.py
# HOW  Reads leak-free per-alpha + policy accuracy (1/3/10), aggregates, and
#      (optional) block-ablation saliency. Produces an explicit, numeric
#      leader-chasing test + best-single-alpha comparison. DIAGNOSTICS ONLY.
# DEPENDS_ON: src/signals/alpha_accuracy.py, src/diagnostics/policy_accuracy.py
# USED_BY: src/jarvis/* (Phase 2), tests, telemetry reports.
# CHANGE_NOTES(IRAC): I: need to detect shortcut/leader-chasing measurably. R:
#   operator diagnostics spec 2026-06-21. A: scoreboard + leader-follow-rate +
#   best-alpha comparison + block importance. C: catches the policy degenerating
#   into a wrapper around handcrafted alphas instead of a real meta-learner.
# =====================================================================
"""Policy Doctor: alpha-vs-policy scoreboard, leader-chasing test, block saliency."""
from __future__ import annotations
import numpy as np
from src.signals.alpha_accuracy import per_alpha_accuracy, aggregate_reliability, HORIZONS
from src.diagnostics.policy_accuracy import policy_directional_accuracy, action_to_direction


def _mean_valid(acc, cnt) -> float:
    m = np.asarray(cnt) > 0
    return float(np.asarray(acc)[m].mean()) if m.any() else 0.0


def build_report(alpha_matrix, actions, close, occupancy_mask, *, window=None,
                 introspection_records=None, leader_follow_threshold: float = 0.75,
                 min_samples: int = 5, horizons=HORIZONS) -> dict:
    """Full diagnostic report. All metrics leak-free; none used as reward."""
    am = np.asarray(alpha_matrix, dtype=np.float64)
    if am.ndim == 1:
        am = am[:, None]
    T, n = am.shape
    occ = np.asarray(occupancy_mask, dtype=bool).ravel()
    a_acc, a_cnt = per_alpha_accuracy(am, close, window, horizons)
    agg = aggregate_reliability(a_acc, a_cnt, occ, min_samples, horizons)
    p = policy_directional_accuracy(actions, close, window, horizons)
    pdir = action_to_direction(actions)

    scoreboard = {}
    for h in horizons:
        pa, pc = p[h]
        scoreboard[h] = {
            "policy_acc_last": float(pa[-1]), "policy_acc_mean": _mean_valid(pa, pc),
            "alpha_mean_last": float(agg[h]["mean"][-1]),
            "alpha_best_last": float(agg[h]["best"][-1]),
            "alpha_dispersion_last": float(agg[h]["dispersion"][-1]),
        }

    # best single alpha at 3-bar (window-mean) vs the policy
    best_idx, best_acc3 = -1, 0.0
    for a in range(n):
        if not occ[a]:
            continue
        m = a_cnt[3][:, a] > 0
        if m.any():
            v = float(a_acc[3][:, a][m].mean())
            if v > best_acc3:
                best_acc3, best_idx = v, a
    policy_acc3_mean = _mean_valid(*p[3])
    outperforms = policy_acc3_mean > best_acc3

    # explicit leader-chasing test: leader = top-acc3 assigned alpha each bar
    acc3, cnt3 = a_acc[3], a_cnt[3]
    valid = (cnt3 >= min_samples) & occ[None, :]
    masked = np.where(valid, acc3, -np.inf)
    follow = decisions = leader_active = 0
    for t in range(T):
        if not valid[t].any():
            continue
        li = int(np.argmax(masked[t]))
        lsig = am[t, li]
        if lsig == 0:
            continue
        leader_active += 1
        if pdir[t] != 0:
            decisions += 1
            if pdir[t] == np.sign(lsig):
                follow += 1
    follow_rate = (follow / decisions) if decisions else 0.0
    chasing = bool(follow_rate > leader_follow_threshold and not outperforms)

    block_importance = None
    if introspection_records:
        acc = {}
        for r in introspection_records:
            gi = getattr(r, "group_importance", None) or r["group_importance"]
            for k, v in gi.items():
                acc[k] = acc.get(k, 0.0) + v
        block_importance = {k: v / len(introspection_records) for k, v in acc.items()}

    return {
        "n_bars": T, "n_alphas_assigned": int(occ.sum()),
        "scoreboard": scoreboard,
        "best_single_alpha": {"idx": best_idx, "acc3_mean": best_acc3,
                              "policy_acc3_mean": policy_acc3_mean,
                              "margin": policy_acc3_mean - best_acc3,
                              "policy_outperforms_best": bool(outperforms)},
        "leader_chasing": {"leader_follow_rate": follow_rate, "decision_bars": decisions,
                           "leader_active_bars": leader_active,
                           "threshold": leader_follow_threshold, "flag": chasing},
        "block_importance": block_importance,
    }


def render(report: dict) -> str:
    """Human-readable Policy Doctor summary."""
    L = []
    L.append(f"POLICY DOCTOR — {report['n_bars']} bars, "
             f"{report['n_alphas_assigned']} alphas assigned")
    L.append("  horizon | policy_acc | alpha_mean | alpha_best | dispersion")
    for h, s in report["scoreboard"].items():
        L.append(f"   {h:>2}-bar | {s['policy_acc_mean']:.3f}      | "
                 f"{s['alpha_mean_last']:.3f}      | {s['alpha_best_last']:.3f}      | "
                 f"{s['alpha_dispersion_last']:.3f}")
    b = report["best_single_alpha"]
    L.append(f"  best single alpha (3-bar): #{b['idx']} acc={b['acc3_mean']:.3f} | "
             f"policy={b['policy_acc3_mean']:.3f} | margin={b['margin']:+.3f} | "
             f"outperforms_best={b['policy_outperforms_best']}")
    lc = report["leader_chasing"]
    verdict = "LEADER-CHASING ⚠" if lc["flag"] else "contextual ✓"
    L.append(f"  leader-follow-rate: {lc['leader_follow_rate']:.2f} over "
             f"{lc['decision_bars']} decisions (thr {lc['threshold']}) -> {verdict}")
    if report["block_importance"]:
        bi = ", ".join(f"{k}={v:.2f}" for k, v in sorted(
            report["block_importance"].items(), key=lambda kv: -kv[1]))
        L.append(f"  block importance (saliency): {bi}")
    return "\n".join(L)
