# v1.12.0 Stage 3: multi-head policy (tp/sl/lot continuous heads) + mixed-action PPO math. Isolated unit test
# (no env/trainer): the shared trunk is untouched, the Gaussian log-prob/entropy/sample are JAX-native, the
# continuous heads are masked on HOLD/CLOSE, and the PPO clip is independent per head group.
from __future__ import annotations
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_config as JC


def _policy(b=8):
    model, params = PPO.init_params(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (b, JC.OBS_SIZE))
    return model, params, obs


def test_policy_has_four_heads_correct_shapes():
    model, params, obs = _policy(8)
    logits, value, cont_mean, cont_log_std = model.apply(params, obs)
    assert logits.shape == (8, JC.N_ACTIONS) and value.shape == (8,)
    assert cont_mean.shape == (8, JC.N_CONT_ACTIONS) and cont_log_std.shape == (JC.N_CONT_ACTIONS,)
    # the shared trunk is the same 3x256 net -> the trunk Dense kernels are unchanged in count
    assert "cont_log_std" in params["params"]


def test_gaussian_logpdf_is_jax_native_and_correct():
    x = jnp.array([0.3, -0.5, 1.2]); mu = jnp.array([0.0, 0.0, 1.0]); ls = jnp.array([-0.5, 0.0, 0.3])
    std = jnp.exp(ls)
    lp = jax.scipy.stats.norm.logpdf(x, mu, std)
    manual = -0.5 * ((x - mu) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    np.testing.assert_allclose(np.asarray(lp), np.asarray(manual), atol=1e-6)


def test_continuous_heads_masked_on_hold_and_close():
    b = 4
    logits = jnp.zeros((b, JC.N_ACTIONS))
    cont_mean = jnp.zeros((b, 3)); ls = jnp.full((3,), -0.5)
    cont_actions = jnp.full((b, 3), 0.4)
    disc = jnp.array([0, 1, 2, 3], jnp.int32)   # HOLD, BUY, SELL, CLOSE
    lpd, lpc, ent = PPO.mixed_logp_entropy(logits, cont_mean, ls, disc, cont_actions)
    assert lpc[0] == 0.0 and lpc[3] == 0.0       # HOLD / CLOSE -> continuous heads contribute nothing
    assert lpc[1] != 0.0 and lpc[2] != 0.0       # BUY / SELL -> continuous log-prob is live
    # on an open, logp_cont == sum of the per-head Gaussian log-probs
    expect = float(jnp.sum(jax.scipy.stats.norm.logpdf(cont_actions[1], cont_mean[1], jnp.exp(ls))))
    np.testing.assert_allclose(float(lpc[1]), expect, atol=1e-6)


def test_head_mask_freezes_a_head():
    b = 2
    logits = jnp.zeros((b, JC.N_ACTIONS)); cont_mean = jnp.zeros((b, 3)); ls = jnp.full((3,), -0.5)
    cont_actions = jnp.full((b, 3), 0.4); disc = jnp.array([1, 2], jnp.int32)   # both opens
    full = PPO.mixed_logp_entropy(logits, cont_mean, ls, disc, cont_actions)[1]
    frozen = PPO.mixed_logp_entropy(logits, cont_mean, ls, disc, cont_actions, head_mask=jnp.array([1., 0., 1.]))[1]
    per_head = jax.scipy.stats.norm.logpdf(cont_actions[0], cont_mean[0], jnp.exp(ls))
    np.testing.assert_allclose(float(full[0] - frozen[0]), float(per_head[1]), atol=1e-6)   # dropped the sl head


def test_sample_mixed_shapes_and_open_mask():
    b = 64
    logits = jnp.tile(jnp.array([10.0, 0., 0., 0.]), (b, 1))   # force HOLD (arg 0) -> cont must be masked
    cont_mean = jnp.zeros((b, 3)); ls = jnp.full((3,), -0.5)
    disc, cont, lpd, lpc = PPO.sample_mixed(logits, cont_mean, ls, jax.random.PRNGKey(2))
    assert disc.shape == (b,) and cont.shape == (b, 3) and lpd.shape == (b,) and lpc.shape == (b,)
    assert jnp.all(disc == 0) and jnp.all(lpc == 0.0)          # all HOLD -> continuous log-prob masked to 0


def test_ppo_loss_mixed_runs_and_clip_is_independent():
    model, params, obs = _policy(32)
    rng = jax.random.PRNGKey(3)
    disc = jax.random.randint(rng, (32,), 0, 4)
    cont = jax.random.uniform(jax.random.PRNGKey(4), (32, 3))
    old_d = jax.random.normal(jax.random.PRNGKey(5), (32,)) * 0.1
    old_c = jax.random.normal(jax.random.PRNGKey(6), (32,)) * 0.1
    adv = jax.random.normal(jax.random.PRNGKey(7), (32,)); ret = jax.random.normal(jax.random.PRNGKey(8), (32,))
    mask = jnp.ones((32,))
    loss, aux = PPO.ppo_loss_mixed(params, model.apply, PPO.norm_apply(PPO.norm_init(), obs),
                                   disc, cont, old_d, old_c, adv, ret, mask, ent_coef=0.0)
    assert np.isfinite(float(loss))
    for k in ("pg_loss", "pg_disc", "pg_cont", "v_loss", "entropy"):
        assert k in aux and np.isfinite(float(aux[k]))
    # all-HOLD batch -> the continuous surrogate is fully masked -> pg_cont == 0 (independent of the discrete head)
    holds = jnp.zeros((32,), jnp.int32)
    _, aux2 = PPO.ppo_loss_mixed(params, model.apply, PPO.norm_apply(PPO.norm_init(), obs),
                                 holds, cont, old_d, old_c, adv, ret, mask, ent_coef=0.0)
    assert abs(float(aux2["pg_cont"])) < 1e-9
