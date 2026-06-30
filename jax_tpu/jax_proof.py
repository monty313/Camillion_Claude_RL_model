# =====================================================================
# WHEN 2026-06-30 (Phase 2, Stage 6) | WHO Claude for Monty
# WHY  PROVE the policy learned a PRINCIPLE, not memorized our hand-designed features (BUILD_PLAN.md Stage 6,
#      JORDAN_PRINCIPLES.md). Parity + smoke tests prove the PLUMBING works -- they say NOTHING about whether
#      the bot learned momentum as a transferable idea. This is the measuring stick. It is built BEFORE long
#      training so it can DRIVE the decision on whether the heavy machinery (auxiliary heads, preference model)
#      is actually needed -- evidence first, not speculation. The honest risk we are testing: we moved from
#      hard-coded RULES to hard-coded CONCEPTS (my 9 momentum scores); the bot could still overfit MY formulas.
# WHERE jax_tpu/jax_proof.py
# HOW  Four probes, all REUSING the existing held-out eval (jax_eval.evaluate / evaluate_won_day_streak):
#      1. ABLATION       -- mean-impute a named obs block (= remove its information) and re-eval. Measures how
#                           much the policy LEANS on that block. Compare vs a control block.
#      2. PERTURBATION   -- rebuild the momentum block with a DIFFERENT recipe (shifted CCI level / windows) and
#                           re-eval. Small change => the policy learned the PRINCIPLE; collapse => it memorized
#                           THIS exact recipe. (The sharpest principle test.)
#      3. HOLDOUT        -- evaluate on held-out time / symbols / regimes (uses evaluate's holdout_start/end +
#                           a symbol-subset helper). Does the style transfer?
#      4. COUNTERFACTUAL -- take one state, sweep ONE momentum dimension, watch the action probabilities shift.
#                           Tests whether the policy responds to the feature CAUSALLY and in the right direction.
#      Ablation/perturbation work by editing the STATIC obs tensor (momentum is a static block) and rebuilding
#      the device-static -- the rollout code is untouched.
# DEPENDS_ON: numpy, jax, jax_tpu.{jax_config,jax_ppo,jax_eval,jax_portfolio_env,jax_static_features},
#             src.observation.momentum_scores
# USED_BY: jax_tpu/tests/test_proof_harness.py, the training notebook (after a baseline run)
# CHANGE_NOTES(IRAC): I: no proof the bot learned a principle vs a heuristic. R: BUILD_PLAN Stage 6 (operator
#   chose the evidence-driven path 2026-06-30). A: ablation + perturbation + holdout + counterfactual probes
#   over the existing held-out eval, no rollout changes. C: turns "the plumbing works" into a measured answer
#   -- and gates any heavier (auxiliary-head / preference-model) spend on real evidence of memorization.
# =====================================================================
"""Stage 6 PROOF HARNESS — does the policy generalize the momentum PRINCIPLE, or memorize the recipe?"""
from __future__ import annotations
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp

from jax_tpu import jax_config as JC
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_eval as EVAL
from jax_tpu import jax_portfolio_env as JPE
from jax_tpu.jax_static_features import BLOCK_RANGES
from src.observation.momentum_scores import compute_momentum_scores, MOMENTUM_NAMES

# A "different recipe" for the momentum block — same PRINCIPLE, different exact numbers. If the policy survives
# these, it learned the idea; if it collapses, it memorized v1.9.0's constants. (Each is a kwargs override for
# compute_momentum_scores.)
PERTURB_RECIPES: dict[str, dict] = {
    "cci_level_120":   {"strength_level": 120.0, "exhaustion_span": 100.0},  # looser "strong/extreme" ladder
    "cci_level_200":   {"strength_level": 200.0, "exhaustion_span": 150.0},  # stricter ladder
    "windows_fast":    {"structure_win": 60, "persistence_win": 15, "decay_win": 10},   # shorter lookbacks
    "windows_slow":    {"structure_win": 240, "persistence_win": 60, "decay_win": 40},  # longer lookbacks
    "bias_scale_tight": {"bias_atr_scale": 3.0},                              # bias saturates sooner
}


def _l1(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum())


# --------------------------------------------------------------------- ablation
def ablate_psd(psd, block: str = "momentum"):
    """Return a copy of the portfolio static with `block` MEAN-IMPUTED (constant over time => carries no
    information). This is the textbook feature ablation; the rollout never knows the difference."""
    lo, hi = BLOCK_RANGES[block]
    so = np.array(psd.static_obs)                       # (N, T, OBS)
    so[..., lo:hi] = so[..., lo:hi].mean(axis=-2, keepdims=True)   # per-symbol time-mean -> constant in time
    return dataclasses.replace(psd, static_obs=so)


def evaluate_ablation(net_params, norm, psd, env_params, *, block="momentum", env=JPE,
                      eval_kw=None, won_day=True):
    """Held-out eval WITH vs WITHOUT a block's information. Large degradation => the policy DEPENDS on `block`.
    Compare the momentum delta to a control block (e.g. 'ohlc') to judge whether it's outsized."""
    eval_kw = eval_kw or {}
    s_intact = JPE.make_portfolio_device_static(psd)
    s_abl = JPE.make_portfolio_device_static(ablate_psd(psd, block))
    ei = EVAL.evaluate(net_params, norm, s_intact, env_params, env=env, **eval_kw)
    ea = EVAL.evaluate(net_params, norm, s_abl, env_params, env=env, **eval_kw)
    res = {"block": block,
           "pass_rate": (ei["pass_rate"], ea["pass_rate"]),
           "delta_pass_rate": ea["pass_rate"] - ei["pass_rate"],
           "mean_return": (ei.get("mean_return"), ea.get("mean_return"))}
    if won_day:
        wi = EVAL.evaluate_won_day_streak(net_params, norm, s_intact, env_params, env=env)
        wa = EVAL.evaluate_won_day_streak(net_params, norm, s_abl, env_params, env=env)
        res["won_streak"] = (wi["best_won_day_streak"], wa["best_won_day_streak"])
        res["delta_won_streak"] = wa["best_won_day_streak"] - wi["best_won_day_streak"]
        res["action_mix_l1_shift"] = _l1(wi["action_mix"], wa["action_mix"])   # 0 = identical behavior
    return res


# ----------------------------------------------------------------- perturbation
def perturb_momentum_psd(psd, subs, **recipe):
    """Return a copy of the portfolio static with the momentum block REBUILT under a different recipe
    (different CCI level / windows). `subs` = {symbol: TradingEnv} (carries .ind + .close), aligned to
    psd.symbols. The SAME trained policy then meets a differently-defined momentum sense."""
    lo, hi = BLOCK_RANGES["momentum"]
    so = np.array(psd.static_obs)
    for j, s in enumerate(psd.symbols):
        envj = subs[s]
        so[j, :, lo:hi] = compute_momentum_scores(envj.ind, envj.close, **recipe)
    return dataclasses.replace(psd, static_obs=so)


def evaluate_perturbation(net_params, norm, psd, subs, env_params, recipe, *, env=JPE,
                          eval_kw=None, won_day=True):
    """Held-out eval under the ORIGINAL vs a PERTURBED momentum recipe. Small delta => principle-like
    (robust to the exact recipe); large delta => recipe memorization."""
    eval_kw = eval_kw or {}
    s_base = JPE.make_portfolio_device_static(psd)
    s_pert = JPE.make_portfolio_device_static(perturb_momentum_psd(psd, subs, **recipe))
    eb = EVAL.evaluate(net_params, norm, s_base, env_params, env=env, **eval_kw)
    ep = EVAL.evaluate(net_params, norm, s_pert, env_params, env=env, **eval_kw)
    res = {"recipe": recipe,
           "pass_rate": (eb["pass_rate"], ep["pass_rate"]),
           "delta_pass_rate": ep["pass_rate"] - eb["pass_rate"]}
    if won_day:
        wb = EVAL.evaluate_won_day_streak(net_params, norm, s_base, env_params, env=env)
        wp = EVAL.evaluate_won_day_streak(net_params, norm, s_pert, env_params, env=env)
        res["won_streak"] = (wb["best_won_day_streak"], wp["best_won_day_streak"])
        res["delta_won_streak"] = wp["best_won_day_streak"] - wb["best_won_day_streak"]
        res["action_mix_l1_shift"] = _l1(wb["action_mix"], wp["action_mix"])
    return res


# ------------------------------------------------------------------- holdout
def evaluate_holdout(net_params, norm, psd, env_params, *, env=JPE, train_frac=0.6, eval_kw=None):
    """Eval on the TRAIN region vs the held-out TAIL (time holdout). A style that only works on the training
    slice is memorization; one that transfers is closer to a principle. (Symbol/regime holdout = build a psd
    from a held-out symbol subset and call evaluate_ablation/holdout on it.)"""
    eval_kw = eval_kw or {}
    static = JPE.make_portfolio_device_static(psd)
    T = int(env_params.T)
    cut = int(T * train_frac)
    train = EVAL.evaluate(net_params, norm, static, env_params, env=env,
                          holdout_start=int(T * 0.05), holdout_end=cut, **eval_kw)
    test = EVAL.evaluate(net_params, norm, static, env_params, env=env,
                         holdout_start=cut, holdout_end=T - 1, **eval_kw)
    return {"train_pass_rate": train["pass_rate"], "test_pass_rate": test["pass_rate"],
            "generalization_gap": train["pass_rate"] - test["pass_rate"],   # big positive gap = overfit
            "train_mean_return": train.get("mean_return"), "test_mean_return": test.get("mean_return")}


# ------------------------------------------------------------- counterfactual
def counterfactual_probe(net_params, norm, base_obs, *, block="momentum", dim, sweep):
    """Hold a state fixed, sweep ONE feature of `block`, and report how the action probabilities move. A
    principle-aware policy shifts its preference in the RIGHT direction (e.g. rising `bias`/`alignment` ->
    more BUY); a memorizer is flat or erratic. `dim` indexes within the block (0..block_width-1)."""
    lo, hi = BLOCK_RANGES[block]
    assert 0 <= dim < (hi - lo), f"dim {dim} out of block '{block}' width {hi - lo}"
    model = PPO.CamillionPolicy()
    sweep = np.asarray(sweep, np.float32)
    batch = np.tile(np.asarray(base_obs, np.float32).reshape(1, -1), (len(sweep), 1))
    batch[:, lo + dim] = sweep
    nob = PPO.norm_apply(norm, jnp.asarray(batch))
    logits, _ = model.apply(net_params, nob)
    probs = np.asarray(jax.nn.softmax(logits, axis=-1))
    feat = MOMENTUM_NAMES[dim] if block == "momentum" else f"{block}[{dim}]"
    return {"block": block, "dim": dim, "feature": feat, "sweep": sweep.tolist(),
            "p_hold": probs[:, 0].tolist(), "p_buy": probs[:, 1].tolist(),
            "p_sell": probs[:, 2].tolist(), "p_close": probs[:, 3].tolist(),
            # net directional lean per sweep point (+ => buy-biased) — the easy-to-read summary
            "buy_minus_sell": (probs[:, 1] - probs[:, 2]).tolist()}


# ------------------------------------------------------------------- report
def run_proof_report(net_params, norm, psd, subs, env_params, *, env=JPE, eval_kw=None,
                     control_block="ohlc", recipes=None, verbose=True):
    """Run the full Stage-6 battery and return a dict (+ optionally print a verdict table). Interpretation:
      - ablation(momentum) delta vs ablation(control): is the momentum dependence OUTSIZED?
      - perturbation deltas SMALL  => principle-like (robust to recipe).  LARGE => recipe memorization.
      - holdout generalization_gap SMALL => transfers.
    These numbers DECIDE whether we invest in auxiliary heads / a preference model (BUILD_PLAN evidence gate)."""
    recipes = recipes or PERTURB_RECIPES
    report = {"ablation": {}, "perturbation": {}, "holdout": None}
    report["ablation"]["momentum"] = evaluate_ablation(net_params, norm, psd, env_params,
                                                       block="momentum", env=env, eval_kw=eval_kw)
    report["ablation"][control_block] = evaluate_ablation(net_params, norm, psd, env_params,
                                                          block=control_block, env=env, eval_kw=eval_kw)
    for name, recipe in recipes.items():
        report["perturbation"][name] = evaluate_perturbation(net_params, norm, psd, subs, env_params,
                                                             recipe, env=env, eval_kw=eval_kw)
    report["holdout"] = evaluate_holdout(net_params, norm, psd, env_params, env=env, eval_kw=eval_kw)

    if verbose:
        am = report["ablation"]["momentum"]; ac = report["ablation"][control_block]
        print("=" * 70)
        print("STAGE 6 — PROOF REPORT (did it learn the PRINCIPLE or the recipe?)")
        print("=" * 70)
        print(f"ABLATION  momentum: Δpass={am['delta_pass_rate']:+.3f}  Δwon_streak={am.get('delta_won_streak')}"
              f"  behavior_shift(L1)={am.get('action_mix_l1_shift'):.3f}")
        print(f"ABLATION  {control_block:<8}: Δpass={ac['delta_pass_rate']:+.3f}  Δwon_streak={ac.get('delta_won_streak')}"
              f"  behavior_shift(L1)={ac.get('action_mix_l1_shift'):.3f}   (control)")
        print("-" * 70)
        for name, r in report["perturbation"].items():
            print(f"PERTURB   {name:<16}: Δpass={r['delta_pass_rate']:+.3f}  "
                  f"Δwon_streak={r.get('delta_won_streak')}  behavior_shift(L1)={r.get('action_mix_l1_shift'):.3f}")
        h = report["holdout"]
        print("-" * 70)
        print(f"HOLDOUT   train_pass={h['train_pass_rate']:.3f}  test_pass={h['test_pass_rate']:.3f}  "
              f"gen_gap={h['generalization_gap']:+.3f}")
        print("=" * 70)
        print("READ: small perturbation Δ + small gen_gap + momentum-ablation Δ not wildly > control = "
              "principle-like.\nBig perturbation Δ or big gen_gap = memorized the recipe -> earn the heavy "
              "machinery (auxiliary heads / preference model).")
    return report
