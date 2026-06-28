# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  A JAX-trained policy must deploy the SAME way as a CPU one (MT5 via ONNX). The
#      3x256 architecture is identical, so we copy the Flax weights into a PyTorch MLP
#      (with the obs-normalizer baked in) and export ONNX. The exported net takes a RAW
#      499 observation and returns the 4 action logits -> argmax = action, exactly like
#      the trained policy. Verified bit-close against the JAX policy before export.
# WHERE jax_tpu/export_to_pytorch.py
# HOW   Flax Dense kernel (in,out) -> torch Linear weight (out,in) = kernel.T; bias copied.
#       Normalization buffer = ((obs-mean)/sqrt(var+1e-8)) clipped to +-CLIP_OBS == RunningNorm.
# DEPENDS_ON: torch, numpy, jax (verify only), jax_tpu.{jax_ppo, jax_checkpoint, jax_config}
# USED_BY: deployment (MT5/ONNX), the notebook's export cell
# CHANGE_NOTES(IRAC): I: ranked head-to-head only if the JAX policy ships like the CPU one.
#   R: blueprint invariant #6 (same policy file format) + C5 (JAX->torch->ONNX). A: copy
#   weights into a matching torch MLP with baked normalization, verify, export ONNX. C: a
#   TPU-trained bot drops into the existing MT5 path with no surprises.
# =====================================================================
"""Convert a JAX/Flax policy checkpoint to a PyTorch 3x256 MLP and export ONNX (MT5 path)."""
from __future__ import annotations
import numpy as np
from jax_tpu import jax_ppo as PPO
from jax_tpu import jax_checkpoint as CKPT
from jax_tpu import jax_config as JC


def _flax_layers(params):
    """Pull ordered (kernel, bias) for Dense_0.. from a Flax param tree."""
    p = params["params"]
    names = sorted(p.keys(), key=lambda k: int(k.split("_")[1]))   # Dense_0, Dense_1, ...
    return [(np.asarray(p[n]["kernel"]), np.asarray(p[n]["bias"])) for n in names]


def build_torch_policy(params, norm, clip: float = JC.CLIP_OBS):
    """Return a torch.nn.Module: raw obs -> normalize -> 3x256 tanh -> 4 logits."""
    import torch
    import torch.nn as nn

    layers = _flax_layers(params)             # [h0,h1,h2, actor(4), critic(1)]
    # guard the Dense ordering assumption (3 hidden + actor + critic) and the actor head shape, so a
    # future net change can't silently ship a wrong-architecture ONNX (verify() is the second net).
    assert len(layers) == len(JC.NET_ARCH) + 2, f"expected {len(JC.NET_ARCH)+2} Dense layers, got {len(layers)}"
    ka_check = layers[len(JC.NET_ARCH)][0]
    assert ka_check.shape[1] == JC.N_ACTIONS, f"actor head out={ka_check.shape[1]} != {JC.N_ACTIONS} actions"
    mean = np.asarray(norm.mean, np.float32)
    std = np.sqrt(np.asarray(norm.var, np.float32) + 1e-8)

    class CamillionTorchPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean))
            self.register_buffer("std", torch.tensor(std))
            self.clip = float(clip)
            self.h = nn.ModuleList()
            for (k, b) in layers[:len(JC.NET_ARCH)]:
                lin = nn.Linear(k.shape[0], k.shape[1])
                lin.weight.data = torch.tensor(k.T.copy()); lin.bias.data = torch.tensor(b.copy())
                self.h.append(lin)
            ka, ba = layers[len(JC.NET_ARCH)]                  # actor head
            self.actor = nn.Linear(ka.shape[0], ka.shape[1])
            self.actor.weight.data = torch.tensor(ka.T.copy()); self.actor.bias.data = torch.tensor(ba.copy())

        def forward(self, obs):
            x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
            for lin in self.h:
                x = torch.tanh(lin(x))
            return self.actor(x)            # logits (argmax = action)

    return CamillionTorchPolicy().eval()


def verify(params, norm, torch_model, n=64, obs_size=JC.OBS_SIZE, atol=1e-4) -> float:
    """Max |logit| difference between the JAX policy and the torch copy on random obs."""
    import jax, jax.numpy as jnp, torch
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 3, (n, obs_size)).astype(np.float32)
    model = PPO.CamillionPolicy()
    jlog, _ = model.apply(params, PPO.norm_apply(norm, jnp.asarray(obs)))
    with torch.no_grad():
        tlog = torch_model(torch.tensor(obs)).numpy()
    d = float(np.max(np.abs(np.asarray(jlog) - tlog)))
    assert d < atol, f"JAX vs torch logits differ by {d} (> {atol})"
    return d


def convert(checkpoint_dir: str, tag: str, out_onnx: str, obs_size: int = JC.OBS_SIZE) -> str:
    """Load checkpoint -> torch policy -> verify -> export ONNX. Returns the ONNX path."""
    import jax, torch
    _, template = PPO.init_params(jax.random.PRNGKey(0), obs_size)
    params, norm, _details = CKPT.load_policy(checkpoint_dir, tag, template, PPO.RunningNorm)
    tm = build_torch_policy(params, norm)
    d = verify(params, norm, tm, obs_size=obs_size)
    dummy = torch.zeros((1, obs_size), dtype=torch.float32)
    torch.onnx.export(tm, dummy, out_onnx, input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(f"exported {out_onnx} (JAX<->torch max|logit| diff = {d:.2e})")
    return out_onnx
