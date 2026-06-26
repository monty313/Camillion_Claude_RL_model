# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  The SYSTEM-LOGIC brain the JARVIS council reasons from. Turns the raw
#      /state snapshot into a grounded read of "are we on track to pass the FTMO
#      challenge CONSISTENTLY, and what is the single highest-leverage next move?"
#      Every number the agents cite comes from here, so their advice is always
#      grounded in the real system and always PROGRESSIVE (there is always a next
#      improvement, no matter how well we are doing).
# WHERE src/jarvis/consistency.py
# HOW  Pure functions over the /state dict (stdlib+math only). No env, no LLM, no
#      mutation. Safe to unit-test and to call once per /state.
# DEPENDS_ON: (stdlib only)
# USED_BY: src/jarvis/council.py, jarvis_bridge.py, tests
# CHANGE_NOTES(IRAC): I: the LLM agents need a single, honest, system-grounded
#   analysis to reason over (not vibes). R: operator 2026-06-26 -- advice must be
#   "based in the logic of the system" and "always have a progressive view no
#   matter how well we are doing". A: one analyze_consistency() that scores pace,
#   breach headroom, consensus, decisiveness, concentration -> the binding
#   constraint + a p(pass) estimate + the next progressive step. C: the council
#   always speaks from real numbers and always points at the next gain toward a
#   consistent pass.
# =====================================================================
"""analyze_consistency(state) -> a grounded, progressive read of consistent-pass risk."""
from __future__ import annotations
import math


def _g(d, *path, default=0.0):
    """Safe nested get: _g(state,'account','equity')."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pct_clamp(x):
    return max(0.0, min(100.0, float(x)))


def analyze_consistency(state: dict) -> dict:
    """Score how close we are to passing CONSISTENTLY and name the next move.

    Returns a flat dict of grounded numbers + a single binding constraint, a
    p(pass) estimate, the biggest risk, and a PROGRESSIVE next step (always set).
    Pure: depends only on the /state dict produced by build_state().
    """
    bal0 = float(_g(state, "account", "episode_start_equity", default=0.0)) or \
        float(_g(state, "account", "balance", default=100_000.0)) or 100_000.0
    equity = float(_g(state, "account", "equity", default=bal0))
    day_start = float(_g(state, "account", "day_start_equity", default=bal0))
    peak = float(_g(state, "account", "peak_equity", default=equity))

    daily_limit_pct = float(_g(state, "ftmo", "daily_loss_limit_pct", default=5.0))
    max_dd_pct = float(_g(state, "ftmo", "max_drawdown_limit_pct", default=10.0))
    target_pct = float(_g(state, "ftmo", "profit_target_pct", default=10.0))
    daily_target_pct = float(_g(state, "ftmo", "daily_target_pct", default=2.5))

    # ---- progress to the +target% pass ----
    profit_pct_of_acct = (equity - bal0) / bal0 * 100.0 if bal0 else 0.0
    progress_to_target = _pct_clamp(profit_pct_of_acct / target_pct * 100.0) if target_pct else 0.0

    # ---- breach headroom (how much room before each wall), 100% = full room ----
    daily_limit_usd = bal0 * daily_limit_pct / 100.0
    daily_loss_usd = max(0.0, day_start - equity)
    daily_headroom = _pct_clamp((daily_limit_usd - daily_loss_usd) / daily_limit_usd * 100.0) \
        if daily_limit_usd else 100.0
    floor = bal0 - bal0 * max_dd_pct / 100.0
    maxdd_headroom = _pct_clamp((equity - floor) / (peak - floor) * 100.0) if (peak - floor) > 0 else 100.0
    binding = "daily loss" if daily_headroom <= maxdd_headroom else "max drawdown"
    binding_headroom = min(daily_headroom, maxdd_headroom)

    # ---- pace toward the +2.5%/day -> +10% in ~4 days plan ----
    day_made_pct = (equity - day_start) / bal0 * 100.0 if bal0 else 0.0
    day_target_hit = day_made_pct >= daily_target_pct
    pace_ratio = (day_made_pct / daily_target_pct) if daily_target_pct else 0.0   # 1.0 = on the day's pace

    # ---- consistency of the equity build (no single day carrying the account) ----
    days = [float(x) for x in (_g(state, "perf", "day_history", default=[]) or []) if float(x) > 0]
    total_up = sum(days) or 1.0
    largest_day_share = (max(days) / total_up * 100.0) if days else 0.0

    # ---- decisiveness + consensus (is the brain sure, do the alphas agree) ----
    confidence = float(_g(state, "policy", "confidence", default=0.0))            # 0..1
    entropy = float(_g(state, "policy", "entropy", default=0.0))
    net_signal = float(state.get("net_signal", 0.0))                              # -1..1, directional-only
    consensus = abs(net_signal)                                                   # 0..1 agreement strength
    consec_losses = int(_g(state, "perf", "consecutive_losses", default=0))

    # ---- p(pass) estimate: bounded, grounded blend (NOT a guarantee) ----
    p = (50.0
         + progress_to_target * 0.18
         + (binding_headroom - 55.0) * 0.32
         - max(0.0, largest_day_share - 33.0) * 0.55
         + (confidence * 100.0 - 50.0) * 0.10
         - consec_losses * 4.0)
    p_pass = int(max(2.0, min(98.0, round(p))))

    # ---- the SINGLE biggest risk to passing CONSISTENTLY right now ----
    risks = []
    if binding_headroom < 30.0:
        risks.append((100 - binding_headroom, f"{binding} headroom is thin ({binding_headroom:.0f}%) — a breach ends the challenge"))
    if consec_losses >= 3:
        risks.append((60 + consec_losses, f"{consec_losses} consecutive losses — drawdown is clustering"))
    if largest_day_share >= 45.0:
        risks.append((55, f"one day is {largest_day_share:.0f}% of the gains — too concentrated to be 'consistent'"))
    if daily_target_pct and pace_ratio < 0.25 and not day_target_hit:
        risks.append((40, f"pace is light ({day_made_pct:.2f}% vs {daily_target_pct:.1f}% daily target)"))
    if confidence < 0.45 and consensus < 0.2:
        risks.append((35, f"low conviction (confidence {confidence*100:.0f}%, weak alpha consensus) — edge is unclear"))
    risks.sort(reverse=True)
    top_risk = risks[0][1] if risks else "no acute risk — the lines are comfortable and the build is balanced"

    # ---- the PROGRESSIVE next step (ALWAYS set; never 'nothing to do') ----
    if binding_headroom < 25.0:
        nxt = f"protect the challenge first: cut size now and let the policy re-confirm before adding — {binding} headroom is {binding_headroom:.0f}%"
        posture = "STAND DOWN"
    elif day_target_hit:
        nxt = f"bank today's +{day_made_pct:.2f}% and stop — consistency is built by not giving gains back, then start fresh tomorrow"
        posture = "BANK & STOP"
    elif consec_losses >= 3 or binding_headroom < 40.0:
        nxt = "halve size and trade only the strongest consensus until the streak/headroom recovers"
        posture = "DEFENSIVE"
    elif largest_day_share >= 45.0:
        nxt = "spread the gains: take smaller, steadier days so no single session dominates the pass"
        posture = "STEADY"
    elif confidence < 0.5 or consensus < 0.25:
        nxt = "raise the edge before sizing up: only act on high-consensus setups, and queue more data/alphas to sharpen conviction"
        posture = "SELECTIVE"
    else:
        # even when green, there is ALWAYS a next refinement toward a *consistent* pass
        nxt = (f"keep the cadence: aim for a steady +{daily_target_pct:.1f}%/day and bank it; "
               f"next gain is tightening entries on the highest-consensus signals to lift win-rate")
        posture = "PRESS — STEADY"

    return {
        "profit_pct_of_account": round(profit_pct_of_acct, 3),
        "progress_to_target_pct": round(progress_to_target, 1),
        "target_pct": target_pct,
        "daily_headroom_pct": round(daily_headroom, 1),
        "maxdd_headroom_pct": round(maxdd_headroom, 1),
        "binding_constraint": binding,
        "binding_headroom_pct": round(binding_headroom, 1),
        "day_made_pct": round(day_made_pct, 3),
        "daily_target_pct": daily_target_pct,
        "day_target_hit": bool(day_target_hit),
        "pace_ratio": round(pace_ratio, 2),
        "largest_day_share_pct": round(largest_day_share, 1),
        "confidence": round(confidence, 3),
        "entropy": round(entropy, 3),
        "net_signal": round(net_signal, 3),
        "consensus_strength": round(consensus, 3),
        "consecutive_losses": consec_losses,
        "p_pass_pct": p_pass,
        "top_risk_to_consistency": top_risk,
        "progressive_next_step": nxt,     # ALWAYS present
        "posture": posture,
    }
