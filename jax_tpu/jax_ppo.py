# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  A custom PPO in Flax/Optax that MIRRORS the CPU SB3 trainer
#      (src/training/trainer.py) so a JAX-trained policy is directly comparable to
#      a CPU one: same 3x256 tanh MLP, same gamma/lambda/clip/vf/ent-anneal/lr/
#      grad-clip, and the same VecNormalize-style running observation normalization.
# WHERE jax_tpu/jax_ppo.py
# HOW   CamillionPolicy (Flax linen): shared tanh trunk -> 4 action logits + scalar
#       value. GAE via reversed lax.scan. Clipped surrogate loss with masking for
#       dead bars. RunningNorm = a Welford mean/var over observations (frozen at eval,
#       clip +/-10) == SB3 VecNormalize(norm_obs, clip_obs=10, norm_reward=False).
# DEPENDS_ON: jax, flax.linen, optax, jax_tpu.jax_config
# USED_BY: jax_tpu/jax_trainer.py, jax_tpu/jax_eval.py, export_to_pytorch.py
# CHANGE_NOTES(IRAC): I: SB3 is CPU/torch-only; a TPU needs PPO in jnp. R: blueprint
#   Rule 4 + "match CPU hyperparameters initially so results are comparable". A: port
#   the exact PPO math + obs-norm to Flax/Optax with the trainer.py constants. C: a
#   TPU policy that trains the SAME objective and exports to the SAME 3x256 net format.
# =====================================================================
"""Custom PPO in Flax/Optax — mirrors src/training/trainer.py (3x256 tanh, same hyperparams)."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax_tpu import jax_config as JC


# --------------------------------------------------------------------------
# Policy / value network — identical architecture to the SB3 MlpPolicy net_arch.
# SB3 default MlpPolicy: separate-ish heads on a shared body; here a shared tanh
# trunk feeding an actor head (4 logits) + a critic head (scalar). 3x256 tanh.
# --------------------------------------------------------------------------
class CamillionPolicy(nn.Module):
    hidden: tuple = JC.NET_ARCH          # (256, 256, 256)
    n_actions: int = JC.N_ACTIONS        # 4 (HOLD/BUY/SELL/CLOSE)
    n_cont: int = JC.N_CONT_ACTIONS      # 3 continuous heads (tp/sl/lot), v1.12.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in self.hidden:                                    # SHARED tanh trunk -- UNTOUCHED (3x256)
            x = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x))
        logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[..., 0]
        # v1.12.0: 3 continuous heads (tp/sl/lot MEANS) on the SAME trunk x + a learnable state-independent
        # log-std (like SB3 continuous actions). The env clips the sampled values to [0,1] -> bounds.
        cont_mean = nn.Dense(self.n_cont, kernel_init=nn.initializers.orthogonal(0.01))(x)
        cont_log_std = self.param("cont_log_std", nn.initializers.constant(JC.CONT_LOG_STD_INIT), (self.n_cont,))
        return logits, value, cont_mean, cont_log_std


# --------------------------------------------------------------------------
# v1.12.0 MIXED action space (discrete direction + 3 continuous tp/sl/lot). JAX-native Gaussian
# (jax.scipy.stats.norm) for log-prob / entropy / sampling. The continuous heads only count when the discrete
# action is BUY or SELL (an OPEN) -- masked out on HOLD/CLOSE.
# --------------------------------------------------------------------------
def _is_open(disc_action):
    return ((disc_action == 1) | (disc_action == 2)).astype(jnp.float32)   # BUY or SELL -> bracket heads active


def sample_mixed(logits, cont_mean, cont_log_std, key):
    """Sample the discrete action (categorical) + the 3 continuous actions (Gaussian). Returns
    (disc_action, cont_actions, logp_disc, logp_cont) where logp_cont is the OPEN-masked sum over heads."""
    kd, kc = jax.random.split(key)
    disc = jax.random.categorical(kd, logits)
    logp_disc = jnp.take_along_axis(jax.nn.log_softmax(logits), disc[..., None], axis=-1)[..., 0]
    std = jnp.exp(cont_log_std)
    cont = cont_mean + std * jax.random.normal(kc, cont_mean.shape)        # raw Gaussian (env clips to [0,1])
    lp = jax.scipy.stats.norm.logpdf(cont, cont_mean, std)                 # (..., 3) per-head log-prob
    logp_cont = _is_open(disc) * jnp.sum(lp, axis=-1)                      # masked: only counts on an OPEN
    return disc, cont, logp_disc, logp_cont


def mixed_logp_entropy(logits, cont_mean, cont_log_std, disc_action, cont_actions, head_mask=None):
    """Re-evaluate log-probs + entropy for stored (disc, cont) actions under the CURRENT policy (PPO update).
    Returns (logp_disc, logp_cont, entropy) -- cont log-prob + cont entropy are OPEN-masked. `head_mask` (3,)
    zeroes specific continuous heads (the curriculum FREEZE: a frozen head contributes 0 logp -> no gradient)."""
    lpa = jax.nn.log_softmax(logits)
    logp_disc = jnp.take_along_axis(lpa, disc_action[..., None], axis=-1)[..., 0]
    ent_disc = -jnp.sum(jnp.exp(lpa) * lpa, axis=-1)
    std = jnp.exp(cont_log_std)
    lp = jax.scipy.stats.norm.logpdf(cont_actions, cont_mean, std)         # (..., 3) per-head log-prob
    ent_cont_per = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + cont_log_std      # Gaussian entropy per head (state-indep)
    if head_mask is not None:                                             # freeze heads -> drop from logp + entropy
        lp = lp * head_mask
        ent_cont_per = ent_cont_per * head_mask
    om = _is_open(disc_action)
    logp_cont = om * jnp.sum(lp, axis=-1)                                 # cont log-prob only on an OPEN
    entropy = ent_disc + om * jnp.sum(ent_cont_per)
    return logp_disc, logp_cont, entropy


def init_params(key, obs_size: int = JC.OBS_SIZE):
    model = CamillionPolicy()
    return model, model.init(key, jnp.zeros((1, obs_size), jnp.float32))


# --------------------------------------------------------------------------
# Running observation normalizer == SB3 VecNormalize(norm_obs=True, clip_obs=10).
# Welford-style accumulation of mean/var; normalize = clip((x-mean)/sqrt(var+eps), +-clip).
# Frozen (no update) during evaluation, exactly like VecNormalize(training=False).
# --------------------------------------------------------------------------
class RunningNorm(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def norm_init(obs_size: int = JC.OBS_SIZE) -> RunningNorm:
    return RunningNorm(jnp.zeros(obs_size), jnp.ones(obs_size), jnp.asarray(1e-4))


def norm_update(rn: RunningNorm, batch: jnp.ndarray, axis_name=None) -> RunningNorm:
    """Parallel (Chan) variance update over a batch of observations (shape [N, obs]).

    When `axis_name` is given (inside a pmap), the batch moments are pmean'd ACROSS DEVICES first, so
    every device's replicated normalizer is updated with the GLOBAL batch statistics and the replicas
    stay identical (the obs-normalizer is synced exactly like the grads are — otherwise each TPU core
    would normalize observations by its own stats and the shared policy would see inconsistent inputs)."""
    batch = batch.reshape(-1, batch.shape[-1])
    b_mean = batch.mean(axis=0)
    b_var = batch.var(axis=0)
    b_count = jnp.asarray(batch.shape[0], rn.count.dtype)
    if axis_name is not None:
        # global batch mean/var across devices (equal per-device counts -> law of total variance).
        # clamp var >= 0: for CONSTANT obs features, g_second - g_mean^2 can be a tiny NEGATIVE from
        # float cancellation -> would make sqrt(var) NaN. The Chan formula below stays non-negative.
        g_mean = jax.lax.pmean(b_mean, axis_name)
        g_second = jax.lax.pmean(b_var + jnp.square(b_mean), axis_name)
        b_var = jnp.maximum(g_second - jnp.square(g_mean), 0.0)
        b_mean = g_mean
        b_count = b_count * jax.lax.psum(jnp.asarray(1.0, rn.count.dtype), axis_name)
    delta = b_mean - rn.mean
    tot = rn.count + b_count
    new_mean = rn.mean + delta * b_count / tot
    m_a = rn.var * rn.count
    m_b = b_var * b_count
    M2 = m_a + m_b + jnp.square(delta) * rn.count * b_count / tot
    return RunningNorm(new_mean, M2 / tot, tot)


def norm_apply(rn: RunningNorm, obs: jnp.ndarray, clip: float = JC.CLIP_OBS) -> jnp.ndarray:
    return jnp.clip((obs - rn.mean) / jnp.sqrt(jnp.maximum(rn.var, 0.0) + 1e-8), -clip, clip)


# --------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation) — gamma/lambda from the CPU trainer.
# rewards/values/dones shape (T, N). Bootstrap from `last_value` (the value at the
# step AFTER the rollout). Returns (advantages, returns), each (T, N).
# --------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value,
                gamma: float = JC.GAMMA, lam: float = JC.GAE_LAMBDA):
    def _step(carry, x):
        gae, next_value = carry
        reward, value, done = x
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae
        return (gae, value), gae

    (_, _), adv = jax.lax.scan(
        _step, (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones), reverse=True)
    returns = adv + values
    return adv, returns


# --------------------------------------------------------------------------
# Clipped PPO loss (masked for dead bars). ent_coef is passed in (annealed by the
# trainer from ENT_COEF_START -> ENT_COEF_END over training, like _make_entropy_anneal).
# --------------------------------------------------------------------------
def ppo_loss(params, apply_fn, obs, actions, old_log_probs, advantages, returns, mask,
             ent_coef: float, clip: float = JC.CLIP_RANGE, vf_coef: float = JC.VF_COEF):
    logits, values, _cm, _ls = apply_fn(params, obs)   # discrete-only loss (cont heads dormant; v1.12.0 Stage 3)
    log_probs_all = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=-1)[:, 0]
    ratio = jnp.exp(log_probs - old_log_probs)
    # normalize advantages over the (masked) minibatch, like SB3
    m = mask
    msum = jnp.maximum(m.sum(), 1.0)
    adv_mean = (advantages * m).sum() / msum
    adv_var = ((advantages - adv_mean) ** 2 * m).sum() / msum
    adv = (advantages - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
    pg_loss = jnp.maximum(pg1, pg2)
    v_loss = 0.5 * (values - returns) ** 2
    probs = jnp.exp(log_probs_all)
    entropy = -jnp.sum(probs * log_probs_all, axis=-1)
    total = (pg_loss + vf_coef * v_loss - ent_coef * entropy) * m
    loss = total.sum() / msum
    aux = {"pg_loss": (pg_loss * m).sum() / msum,
           "v_loss": (v_loss * m).sum() / msum,
           "entropy": (entropy * m).sum() / msum}
    return loss, aux


def ppo_loss_mixed(params, apply_fn, obs, disc_actions, cont_actions, old_logp_disc, old_logp_cont,
                   advantages, returns, mask, ent_coef: float, head_mask=None,
                   clip: float = JC.CLIP_RANGE, vf_coef: float = JC.VF_COEF):
    """v1.12.0 MIXED-action PPO loss. The DISCRETE head and the CONTINUOUS heads (tp/sl/lot) each get their OWN
    clipped surrogate (independent ratios), summed. The continuous surrogate + entropy are OPEN-masked (count
    only on BUY/SELL). `head_mask` (3,) zeroes specific continuous heads for the curriculum freeze (Stage 4)."""
    logits, values, cont_mean, cont_log_std = apply_fn(params, obs)
    logp_disc, logp_cont, entropy = mixed_logp_entropy(
        logits, cont_mean, cont_log_std, disc_actions, cont_actions, head_mask=head_mask)
    m = mask
    msum = jnp.maximum(m.sum(), 1.0)
    adv_mean = (advantages * m).sum() / msum
    adv_var = ((advantages - adv_mean) ** 2 * m).sum() / msum
    adv = (advantages - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
    # DISCRETE surrogate (own ratio + clip)
    ratio_d = jnp.exp(logp_disc - old_logp_disc)
    pg_d = jnp.maximum(-adv * ratio_d, -adv * jnp.clip(ratio_d, 1.0 - clip, 1.0 + clip))
    # CONTINUOUS surrogate (INDEPENDENT ratio + clip); only counts on an OPEN (logp_cont already open-masked)
    om = _is_open(disc_actions)
    ratio_c = jnp.exp(logp_cont - old_logp_cont)
    pg_c = om * jnp.maximum(-adv * ratio_c, -adv * jnp.clip(ratio_c, 1.0 - clip, 1.0 + clip))
    pg_loss = pg_d + pg_c
    v_loss = 0.5 * (values - returns) ** 2
    total = (pg_loss + vf_coef * v_loss - ent_coef * entropy) * m
    loss = total.sum() / msum
    aux = {"pg_loss": (pg_loss * m).sum() / msum, "pg_disc": (pg_d * m).sum() / msum,
           "pg_cont": (pg_c * m).sum() / msum, "v_loss": (v_loss * m).sum() / msum,
           "entropy": (entropy * m).sum() / msum}
    return loss, aux


def make_optimizer(lr: float = JC.LEARNING_RATE, max_grad_norm: float = JC.MAX_GRAD_NORM):
    """optax.chain(clip_by_global_norm, adam) — matches SB3 PPO's grad clip + Adam."""
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))


def entropy_coef(progress_frac: float,
                 start: float = JC.ENT_COEF_START, end: float = JC.ENT_COEF_END) -> float:
    """Linear anneal start->end over training (progress_frac in [0,1]); == _make_entropy_anneal."""
    p = jnp.clip(progress_frac, 0.0, 1.0)
    return start + (end - start) * p
