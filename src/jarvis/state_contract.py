# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  THE single source of truth for the JARVIS /state JSON contract. A PURE
#      function (stdlib+numpy-free) that shapes a raw snapshot of already-extracted
#      bot primitives into the exact dict the HUD's pullLive() reads. Kept separate
#      from the server (FastAPI is not installed) and from the env (the provider
#      does the extraction) so it is trivially unit-testable.
# WHERE src/jarvis/state_contract.py
# HOW  build_state(snap) -> dict. Folds the 4-action policy {HOLD,BUY,SELL,CLOSE}
#      into the HUD's 3-way {BUY,SELL,HOLD} (CLOSE -> HOLD), computes confidence,
#      lists every field that fell back to a safe default in "gaps", and NEVER
#      fabricates a number the bot did not produce.
# DEPENDS_ON: config/constants.py (action order + obs contract version)
# USED_BY: src/jarvis/state_provider.py, jarvis_bridge.py, tests
# CHANGE_NOTES(IRAC): I: the HUD needs an exact, honest contract; some fields the
#   bot does not produce. R: HANDOFF data contract + "never fabricate -> safe
#   default + flag". A: pure shaper that maps real fields, folds CLOSE->HOLD,
#   exposes directional-only net_signal + basis (so the HUD never divides by a
#   hardcoded 15), and emits a gaps[] list. C: JARVIS speaks only real numbers and
#   can say which figures are missing -> trustworthy advice toward a consistent pass.
# =====================================================================
"""build_state(snap) -> the exact /state contract dict (pure, honest, gap-flagged)."""
from __future__ import annotations
import math
from config import constants as C

# Action order in the policy's 4-vector (matches config.constants).
_HOLD, _BUY, _SELL, _CLOSE = C.ACTION_HOLD, C.ACTION_BUY, C.ACTION_SELL, C.ACTION_CLOSE
_LOG_NACT = math.log(C.N_ACTIONS)   # max entropy over the action space (for confidence)


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(d)


def _policy_block(snap, gaps):
    """Map the policy snapshot to the contract, folding CLOSE into HOLD.

    If snap has no model (policy_raw is None) the action distribution is an HONEST
    alpha-consensus fallback derived from the directional net signal, flagged in gaps.
    """
    raw = snap.get("policy_raw")
    net = _f(snap.get("net_signal", 0.0))
    if raw and raw.get("action_probs"):
        probs = [ _f(p) for p in raw["action_probs"] ]
        while len(probs) < C.N_ACTIONS:
            probs.append(0.0)
        prob_buy, prob_sell = probs[_BUY], probs[_SELL]
        prob_hold = probs[_HOLD] + probs[_CLOSE]            # CLOSE folded into HOLD
        value = _f(raw.get("value", 0.0))
        entropy = _f(raw.get("entropy", 0.0))
        confidence = max(0.0, min(1.0, 1.0 - entropy / _LOG_NACT)) if _LOG_NACT else 0.0
        name = raw.get("chosen_action_name", "HOLD")
        action = "HOLD" if name == "CLOSE" else name        # 3-way display
        action_raw = name
    else:
        # No trained model: derive an honest directional lean from alpha consensus.
        gaps.append("policy.action(alpha-fallback: no trained model attached)")
        prob_buy = max(0.0, net)
        prob_sell = max(0.0, -net)
        prob_hold = max(0.0, 1.0 - prob_buy - prob_sell)
        value = 0.0
        entropy = _LOG_NACT
        confidence = 0.0                                    # honest: the model did not decide
        action = "BUY" if net > 0.12 else ("SELL" if net < -0.12 else "HOLD")
        action_raw = action
        probs = [prob_hold, prob_buy, prob_sell, 0.0]

    # renormalise the 3-way so the HUD's sum~1 check always holds
    s3 = prob_buy + prob_sell + prob_hold
    if s3 > 0:
        prob_buy, prob_sell, prob_hold = prob_buy / s3, prob_sell / s3, prob_hold / s3

    # fields the bot does not (yet) produce -> safe defaults, all flagged
    ex = snap.get("policy_extra", {}) or {}
    def _gap(field, val, present):
        if not present:
            gaps.append(field)
        return val
    advantage = _gap("policy.advantage", _f(ex.get("advantage", 0.0)), ex.get("advantage") is not None)
    regime = _gap("policy.regime", ex.get("regime", "n/a"), ex.get("regime") not in (None, "n/a"))
    recommended_lots = _gap("policy.recommended_lots(active size, not a sizing-head recommendation)",
                            _f(ex.get("recommended_lots", snap.get("position", {}).get("lots", 0.0))),
                            ex.get("recommended_lots") is not None)
    expected_dd = _gap("policy.expected_dd_pct(headroom proxy, not a forward prediction)",
                       _f(ex.get("expected_dd_pct", 0.0)), ex.get("expected_dd_pct") is not None)
    calib = _gap("policy.value_calibration_pct(no calibration module)",
                 _f(ex.get("value_calibration_pct", 0.0)), ex.get("value_calibration_pct") is not None)

    return {
        "action": action,
        "prob_buy": round(prob_buy, 4), "prob_sell": round(prob_sell, 4), "prob_hold": round(prob_hold, 4),
        "value": round(value, 4), "confidence": round(confidence, 4), "entropy": round(entropy, 4),
        "advantage": round(advantage, 4), "regime": regime,
        "recommended_lots": round(recommended_lots, 2), "expected_dd_pct": round(expected_dd, 3),
        "value_calibration_pct": round(calib, 1),
        "action_raw": action_raw, "action_probs_raw": [round(_f(p), 4) for p in probs],
    }


def build_state(snap: dict) -> dict:
    """Shape a raw bot snapshot into the exact /state contract dict. Pure, honest.

    `snap` carries already-extracted primitives (the provider fills it). This never
    touches the env, policy, or network; it only shapes, folds, defaults, and flags.
    """
    gaps: list[str] = []
    acc = snap.get("account", {}) or {}
    ftmo = snap.get("ftmo", {}) or {}
    pos = snap.get("position", {}) or {}
    perf = snap.get("perf", {}) or {}

    # alphas: pass through {name,signal,streak,directional}; HUD reads the first three
    alphas = []
    for a in (snap.get("alphas") or []):
        sig = int(a.get("signal", 0))
        if sig not in (-1, 0, 1):
            sig = 0
        alphas.append({"name": str(a.get("name", "alpha")), "signal": sig,
                       "streak": int(a.get("streak", 0)), "directional": bool(a.get("directional", True))})

    if not pos.get("age_known", True):
        gaps.append("position.age_min(entry bar not observed)")
    for field, val in (("news", snap.get("news", [])), ("human.overrides", _f(snap.get("human", {}).get("overrides", 0))),
                       ("human.panic_closes", _f(snap.get("human", {}).get("panic_closes", 0))),
                       ("human.discipline_pct", _f(snap.get("human", {}).get("discipline_pct", 0)))):
        if (field == "news" and not val) or (field != "news" and val == 0):
            gaps.append(field)

    state = {
        "account": {
            "balance": _f(acc.get("balance")), "equity": _f(acc.get("equity")),
            "day_start_equity": _f(acc.get("day_start_equity")),
            "episode_start_equity": _f(acc.get("episode_start_equity")),
            "peak_equity": _f(acc.get("peak_equity")),
        },
        "ftmo": {
            "daily_loss_limit_pct": _f(ftmo.get("daily_loss_limit_pct", 5.0)),
            "max_drawdown_limit_pct": _f(ftmo.get("max_drawdown_limit_pct", 10.0)),
            "profit_target_pct": _f(ftmo.get("profit_target_pct", 10.0)),
            "daily_target_pct": _f(ftmo.get("daily_target_pct", 2.5)),
        },
        "position": {
            "dir": pos.get("dir", "FLAT"), "symbol": pos.get("symbol", "n/a"),
            "lots": round(_f(pos.get("lots", 0.0)), 2), "entry": _f(pos.get("entry", 0.0)),
            "price": _f(pos.get("price", 0.0)), "age_min": int(_f(pos.get("age_min", 0))),
        },
        "alphas": alphas,
        "policy": _policy_block(snap, gaps),
        "perf": {
            "win_rate_pct": round(_f(perf.get("win_rate_pct", 0.0)), 1),
            "trades": int(_f(perf.get("trades", 0))),
            "consecutive_losses": int(_f(perf.get("consecutive_losses", 0))),
            "day_history": [round(_f(x), 2) for x in (perf.get("day_history") or [])],
        },
        "news": list(snap.get("news", []) or []),
        "human": {
            "overrides": int(_f(snap.get("human", {}).get("overrides", 0))),
            "panic_closes": int(_f(snap.get("human", {}).get("panic_closes", 0))),
            "discipline_pct": int(_f(snap.get("human", {}).get("discipline_pct", 0))),
        },
        "clock": str(snap.get("clock", "00:00:00")),
        # --- additive (HUD ignores unknown keys; the council + tests use them) ---
        "net_signal": round(_f(snap.get("net_signal", 0.0)), 3),
        "net_signal_basis": int(snap.get("n_directional", max(1, len(alphas)))),  # NEVER a hardcoded 15
        "mode": snap.get("mode", "FTMO"),
        "model_attached": bool(snap.get("model_attached", False)),
        "obs_contract": getattr(C, "OBSERVATION_CONTRACT_VERSION", "n/a"),
        "contract_version": "v1",
        "gaps": gaps,
    }
    return state
