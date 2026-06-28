# =====================================================================
# WHEN 2026-06-21 (Phase 2) | WHO Claude for Monty
# WHY  Walk-forward validation: roll train/val/test windows across the data,
#      evaluate the policy on each UNSEEN test window, detect FTMO pass/breach,
#      and aggregate a pass-rate. This is the real scoreboard (pass-rate first).
# WHERE src/training/walk_forward.py
# HOW  Each test window gets a fresh TradingEnv; evaluate_policy runs it READ-ONLY
#      (no training, no leakage — the env only reads cached bars in [ts,te]).
#      Pass = reached the CHALLENGE profit target (+profit_target_total_pct, ~+10%)
#      with NO FTMO breach. (Was the 2.5% DAILY target — wrong threshold — fixed 2026-06-25.)
# DEPENDS_ON: src/env/trading_env.py, src/training/evaluate.py, config/ftmo_config.py
# USED_BY: notebooks (Phase 2), tests.
# CHANGE_NOTES(IRAC): I: need an honest FTMO pass-rate, not just PnL. R: operator
#   'pass rate first' + walk-forward discipline. A: rolling windows + per-window
#   pass/breach + aggregate. C: a defensible estimate of how often the policy
#   actually passes the challenge on unseen data.
# =====================================================================
"""Walk-forward validation: rolling windows -> per-window FTMO pass-rate."""
from __future__ import annotations
import numpy as np
from config.ftmo_config import load_active_config
from src.env.trading_env import TradingEnv
from src.training.evaluate import evaluate_policy


def make_windows(n: int, train: int, val: int, test: int, step: int) -> list[dict]:
    """Rolling (train, val, test) windows by bar count."""
    out, s = [], 0
    while s + train + val + test <= n:
        out.append({"train": (s, s + train), "val": (s + train, s + train + val),
                    "test": (s + train + val, s + train + val + test)})
        s += step
    return out


def run(indicators, close, time_ns, registry_factory, policy_fn, *, cfg=None,
        windows=None, train=None, val=None, test=None, step=None, warmup=50,
        target_pct=None, max_steps=None, aux=None) -> dict:
    """Evaluate policy_fn across walk-forward TEST windows. Returns pass-rate + detail.

    PASS = reached the CHALLENGE profit target (+profit_target_total_pct, default
    +10%) on unseen data with NO FTMO breach. `target_pct` overrides the threshold
    only for experiments/tests; leave it None to use the real challenge target.
    (Previously defaulted to 2.5% -- the DAILY target -- which measured the wrong thing.)
    `aux` (v1.6.0 OHLC obs block + ADX-DI side-channel), if given, is sliced per window so the
    eval env matches training (else the OHLC block is zeros and the two ADX-DI alphas stay inactive)."""
    cfg = cfg or load_active_config()
    target = float(target_pct) if target_pct is not None else float(getattr(cfg, "profit_target_total_pct", 10.0))
    ind, cl, tm = np.asarray(indicators), np.asarray(close), np.asarray(time_ns)
    ax = np.asarray(aux) if aux is not None else None
    n = len(cl)
    if windows is None:
        train = train or int(n * 0.5); val = val or int(n * 0.15)
        test = test or int(n * 0.2); step = step or test
        windows = make_windows(n, train, val, test, step)
    results = []
    for w in windows:
        ts, te = w["test"]
        env = TradingEnv(ind[ts:te], cl[ts:te], tm[ts:te], registry_factory(),
                         cfg=cfg, warmup=min(warmup, max(1, (te - ts) // 4)),
                         aux=(ax[ts:te] if ax is not None else None))
        out = evaluate_policy(env, policy_fn, do_introspect=False, max_steps=max_steps)
        breached = bool(env.acc.episode_breached)
        ret = env.acc.equity / cfg.starting_balance - 1.0
        # WIN = hit the +10% challenge target (env set episode_passed) or finished at/above
        # the target return, and NEVER breached an FTMO limit along the way.
        passed = (not breached) and (bool(env.acc.episode_passed) or ret >= target / 100.0)
        results.append({"test": [int(ts), int(te)], "passed": bool(passed),
                        "breached": breached, "hit_target": bool(env.acc.episode_passed),
                        "final_return_pct": round(ret * 100, 3),
                        "n_steps": out["n_steps"]})
    pr = float(np.mean([r["passed"] for r in results])) if results else 0.0
    return {"pass_rate": pr, "n_windows": len(results),
            "breaches": int(sum(r["breached"] for r in results)), "results": results}
