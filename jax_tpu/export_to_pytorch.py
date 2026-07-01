# =====================================================================
# WHEN 2026-06-28 (v1.12.0 multi-head: 2026-06-30) | WHO Claude for Monty
# WHY  A JAX-trained policy must deploy the SAME way as a CPU one (MT5 via ONNX). The 3x256 trunk is identical,
#      so we copy the Flax weights into a PyTorch MLP (obs-normalizer baked in) and export ONNX. v1.12.0: the
#      actor is MULTI-HEAD -- the ONNX now emits the discrete DIRECTION logits (4) AND the 3 continuous bracket
#      heads (tp/sl/lot) already CLIPPED to [0,1] and MAPPED to their bounded units (tp_pct, sl_pct, lot_mult),
#      so the deploy side (MT5/go-live) reads final values with no extra math. Verified bit-close vs JAX.
# WHERE jax_tpu/export_to_pytorch.py
# HOW   Flax Dense kernel (in,out) -> torch Linear weight (out,in) = kernel.T; bias copied. Dense order:
#       h0,h1,h2 (trunk), Dense_actor(4), Dense_critic(1), Dense_cont_mean(3). Normalization buffer =
#       clip((obs-mean)/sqrt(var+1e-8), +-CLIP_OBS) == RunningNorm. tp/sl/lot = MIN + clip(mean,0,1)*(MAX-MIN).
# DEPENDS_ON: torch, numpy, jax (verify only), config.constants (the head bounds), jax_tpu.{jax_ppo,jax_checkpoint,jax_config}
# USED_BY: deployment (MT5/ONNX), the notebook's export cell, deep_test onnx_rollout
# CHANGE_NOTES(IRAC): I: the Stage-3 multi-head policy has 6 Dense layers + returns 4 heads; the old 2-output
#   export/verify were BROKEN + would ship a wrong-arch ONNX. R: operator "update ONNX to the 3 continuous
#   heads; output direction_logits[4] + tp[1] + sl[1] + lot[1]". A: export all 4 heads (bracket heads mapped to
#   final units), guard the new Dense count, verify JAX<->torch on ALL heads. C: a TPU multi-head policy drops
#   into the MT5/ONNX path with final tp/sl/lot values -- no silent 2-vs-5 output mismatch at inference.
# =====================================================================
"""Convert a JAX/Flax MULTI-HEAD policy checkpoint to a PyTorch MLP + export ONNX (direction + tp/sl/lot)."""
from __future__ import annotations
import numpy as np
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_checkpoint as CKPT
from jax_tpu import jax_config as JC
from config import constants as C

_OUTPUT_NAMES = ["direction_logits", "tp_pct", "sl_pct", "lot_mult"]   # ONNX outputs (v1.12.0 multi-head)


def _flax_layers(params):
    """Pull ordered (kernel, bias) for Dense_0.. from a Flax param tree (skips non-Dense params like the
    continuous log-std)."""
    p = params["params"]
    names = sorted((k for k in p if k.startswith("Dense_")), key=lambda k: int(k.split("_")[1]))
    return [(np.asarray(p[n]["kernel"]), np.asarray(p[n]["bias"])) for n in names]


def _map01(x01, lo, hi):
    return lo + x01 * (hi - lo)


def build_torch_policy(params, norm, clip: float = JC.CLIP_OBS):
    """Return a torch.nn.Module: raw obs -> normalize -> 3x256 tanh -> (direction_logits, tp_pct, sl_pct,
    lot_mult). The 3 bracket heads are clipped to [0,1] then mapped to their bounded units (constants)."""
    import torch
    import torch.nn as nn

    layers = _flax_layers(params)             # [h0,h1,h2, actor(4), critic(1), cont_mean(3)]
    nh = len(JC.NET_ARCH)
    assert len(layers) == nh + 3, f"expected {nh+3} Dense layers (trunk+actor+critic+cont), got {len(layers)}"
    ka = layers[nh][0]; kc = layers[nh + 2][0]
    assert ka.shape[1] == JC.N_ACTIONS, f"actor head out={ka.shape[1]} != {JC.N_ACTIONS}"
    assert kc.shape[1] == JC.N_CONT_ACTIONS, f"cont head out={kc.shape[1]} != {JC.N_CONT_ACTIONS}"
    mean = np.asarray(norm.mean, np.float32); std = np.sqrt(np.asarray(norm.var, np.float32) + 1e-8)
    bounds = ((C.TP_MIN_PCT, C.TP_MAX_PCT), (C.SL_MIN_PCT, C.SL_MAX_PCT), (C.LOT_MIN_MULT, C.LOT_MAX_MULT))

    def _lin(k, b):
        lin = nn.Linear(k.shape[0], k.shape[1]); lin.weight.data = torch.tensor(k.T.copy())
        lin.bias.data = torch.tensor(b.copy()); return lin

    class CamillionTorchPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean)); self.register_buffer("std", torch.tensor(std))
            self.clip = float(clip)
            self.h = nn.ModuleList([_lin(k, b) for (k, b) in layers[:nh]])
            self.actor = _lin(*layers[nh])            # direction logits (4)
            self.cont = _lin(*layers[nh + 2])         # continuous means (3): tp, sl, lot
            self.bounds = bounds

        def forward(self, obs):
            x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
            for lin in self.h:
                x = torch.tanh(lin(x))
            logits = self.actor(x)                                    # (B,4) -> argmax = action
            cm = torch.clamp(self.cont(x), 0.0, 1.0)                  # (B,3) heads in [0,1] (env clips the same)
            tp = _map01(cm[:, 0:1], *self.bounds[0])                  # -> tp_pct
            sl = _map01(cm[:, 1:2], *self.bounds[1])                  # -> sl_pct
            lot = _map01(cm[:, 2:3], *self.bounds[2])                 # -> lot_mult (deployer applies the 1% clamp)
            return logits, tp, sl, lot

    return CamillionTorchPolicy().eval()


def _jax_heads(params, norm, obs):
    """JAX reference: (logits, tp_pct, sl_pct, lot_mult) for the same obs (bracket heads clipped+mapped)."""
    import jax.numpy as jnp
    model = PPO.CamillionPolicy()
    logits, _v, cm, _ls = model.apply(params, PPO.norm_apply(norm, jnp.asarray(obs)))
    cm = np.clip(np.asarray(cm), 0.0, 1.0)
    tp = _map01(cm[:, 0], C.TP_MIN_PCT, C.TP_MAX_PCT)
    sl = _map01(cm[:, 1], C.SL_MIN_PCT, C.SL_MAX_PCT)
    lot = _map01(cm[:, 2], C.LOT_MIN_MULT, C.LOT_MAX_MULT)
    return np.asarray(logits), tp, sl, lot


def verify(params, norm, torch_model, n=64, obs_size=JC.OBS_SIZE, atol=1e-4) -> float:
    """Max |diff| between the JAX policy and the torch copy across ALL heads (logits + tp/sl/lot) on random obs."""
    import torch
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 3, (n, obs_size)).astype(np.float32)
    jl, jtp, jsl, jlot = _jax_heads(params, norm, obs)
    with torch.no_grad():
        tl, ttp, tsl, tlot = torch_model(torch.tensor(obs))
    d = max(float(np.max(np.abs(jl - tl.numpy()))),
            float(np.max(np.abs(jtp - ttp.numpy()[:, 0]))),
            float(np.max(np.abs(jsl - tsl.numpy()[:, 0]))),
            float(np.max(np.abs(jlot - tlot.numpy()[:, 0]))))
    assert d < atol, f"JAX vs torch heads differ by {d} (> {atol})"
    return d


def convert(checkpoint_dir: str, tag: str, out_onnx: str, obs_size: int = JC.OBS_SIZE) -> str:
    """Load checkpoint -> torch policy -> verify ALL heads -> export ONNX. Returns the ONNX path."""
    import jax, torch
    _, template = PPO.init_params(jax.random.PRNGKey(0), obs_size)
    params, norm, _details = CKPT.load_policy(checkpoint_dir, tag, template, PPO.RunningNorm)
    tm = build_torch_policy(params, norm)
    d = verify(params, norm, tm, obs_size=obs_size)
    dummy = torch.zeros((1, obs_size), dtype=torch.float32)
    dyn = {"obs": {0: "batch"}}
    dyn.update({o: {0: "batch"} for o in _OUTPUT_NAMES})
    torch.onnx.export(tm, dummy, out_onnx, input_names=["obs"], output_names=_OUTPUT_NAMES,
                      dynamic_axes=dyn, opset_version=17)
    print(f"exported {out_onnx} — outputs {_OUTPUT_NAMES} (JAX<->torch max|diff| = {d:.2e})")
    return out_onnx
