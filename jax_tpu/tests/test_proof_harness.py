# =====================================================================
# WHEN 2026-06-30 (Stage 6) | WHO Claude for Monty
# WHY  Lock the PROOF HARNESS mechanics (jax_proof.py): ablation mean-imputes a block, perturbation rebuilds
#      the momentum recipe, holdout splits time, and the counterfactual probe sweeps one feature -> finite
#      action probabilities. (Interpretation needs a TRAINED policy; this proves the machinery runs + is
#      correct with a fresh random policy on tiny synthetic data.)
# WHERE jax_tpu/tests/test_proof_harness.py
# =====================================================================
"""Stage 6 proof-harness mechanics (ablation / perturbation / holdout / counterfactual)."""
from __future__ import annotations
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import pandas as pd

from config import constants as C
from config.ftmo_config import load_ftmo_config
from src.env.portfolio_env import build_portfolio_subs
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from jax_tpu import jax_static_features as JSF
from jax_tpu import jax_portfolio_env as JPE
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_proof as PROOF
from jax_tpu.jax_static_features import BLOCK_RANGES


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _setup(symbols=("EURUSD", "GBPUSD"), n_bars=2500, seed=11):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-03-04 00:00:00").value
    time_ns = (t0 + np.arange(n_bars, dtype=np.int64) * 60_000_000_000).astype(np.int64)
    sd = {}
    for k, s in enumerate(symbols):
        close = (1.10 + 0.2 * k) + np.cumsum(rng.normal(2e-5, 1e-4, n_bars)).astype(np.float64)
        ind = rng.normal(0, 1.0, (n_bars, 220)).astype(np.float32)
        sd[s] = (ind, close, time_ns)
    cfg = load_ftmo_config()
    subs = build_portfolio_subs(sd, _reg, cfg=cfg, warmup=50, progress=False)
    psd = JSF.build_portfolio_static(subs)
    params = JPE.portfolio_params(psd)
    # a FRESH (untrained) policy + identity norm — enough to exercise the harness mechanics
    model = PPO.CamillionPolicy()
    net_params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, C.OBS_TOTAL_SIZE), jnp.float32))
    norm = PPO.norm_init(C.OBS_TOTAL_SIZE)
    return subs, psd, params, net_params, norm


def test_ablation_mean_imputes_the_block():
    _, psd, _, _, _ = _setup()
    lo, hi = BLOCK_RANGES["momentum"]
    abl = PROOF.ablate_psd(psd, "momentum")
    # other blocks untouched
    assert np.array_equal(np.asarray(psd.static_obs)[..., :lo], np.asarray(abl.static_obs)[..., :lo])
    # the momentum block is now CONSTANT over time (axis=-2) per symbol => no information left
    blk = np.asarray(abl.static_obs)[..., lo:hi]
    assert np.allclose(blk, blk[:, :1, :]), "ablated block must be constant over time"
    # and it equals the original per-symbol time-mean
    assert np.allclose(blk[:, 0, :], np.asarray(psd.static_obs)[..., lo:hi].mean(axis=1))


def test_perturbation_changes_the_momentum_recipe():
    subs, psd, _, _, _ = _setup()
    lo, hi = BLOCK_RANGES["momentum"]
    pert = PROOF.perturb_momentum_psd(psd, subs, **PROOF.PERTURB_RECIPES["windows_fast"])
    base_blk = np.asarray(psd.static_obs)[..., lo:hi]
    pert_blk = np.asarray(pert.static_obs)[..., lo:hi]
    assert base_blk.shape == pert_blk.shape
    assert not np.allclose(base_blk, pert_blk), "a different recipe must change the momentum block"
    # everything OUTSIDE the momentum block is identical
    assert np.array_equal(np.asarray(psd.static_obs)[..., :lo], np.asarray(pert.static_obs)[..., :lo])
    assert np.all(np.isfinite(pert_blk))


def test_counterfactual_probe_returns_finite_action_probs():
    _, _, _, net_params, norm = _setup()
    base = np.zeros(C.OBS_TOTAL_SIZE, np.float32)
    sweep = np.linspace(-1.0, 1.0, 7)
    out = PROOF.counterfactual_probe(net_params, norm, base, block="momentum", dim=2, sweep=sweep)  # dim 2 = alignment
    assert out["feature"] == "mom_alignment"
    assert len(out["p_buy"]) == len(sweep) == len(out["buy_minus_sell"])
    p = np.array([out["p_hold"], out["p_buy"], out["p_sell"], out["p_close"]])
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=0), 1.0, atol=1e-4)   # valid distribution


def test_ablation_and_holdout_eval_run_end_to_end():
    subs, psd, params, net_params, norm = _setup()
    ek = {"n_windows": 2, "window_len": 150}   # tiny + fast; won_day walk skipped for speed
    abl = PROOF.evaluate_ablation(net_params, norm, psd, params, block="momentum", eval_kw=ek, won_day=False)
    assert "delta_pass_rate" in abl and np.isfinite(abl["delta_pass_rate"])
    assert 0.0 <= abl["pass_rate"][0] <= 1.0 and 0.0 <= abl["pass_rate"][1] <= 1.0
    hold = PROOF.evaluate_holdout(net_params, norm, psd, params, eval_kw=ek)
    assert np.isfinite(hold["generalization_gap"])
    pert = PROOF.evaluate_perturbation(net_params, norm, psd, subs, params,
                                       PROOF.PERTURB_RECIPES["cci_level_120"], eval_kw=ek, won_day=False)
    assert np.isfinite(pert["delta_pass_rate"])
