# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Persist EVERYTHING to Colab Drive so a disconnect never loses progress and
#      every saved policy is documented: params + obs-normalizer + a details JSON,
#      plus a per-eval progress ledger (jax_progress.jsonl), a rolling best policy,
#      and the final "40 passes in a row" policy. Operator 2026-06-28: "save progress,
#      document the details, save the policies and their details in Colab."
# WHERE jax_tpu/jax_checkpoint.py
# HOW   flax.serialization (msgpack) for params/norm (version-stable across jax/orbax),
#       JSON for the human-readable details + the append-only ledger. No orbax API churn.
# DEPENDS_ON: flax.serialization, numpy, json (stdlib)
# USED_BY: jax_tpu/jax_trainer.py, jax_tpu/jax_eval.py, the notebook, export_to_pytorch.py
# CHANGE_NOTES(IRAC): I: Colab sessions drop; progress + policies must survive on Drive.
#   R: operator save/document requirement + resumable training to 40-in-a-row. A: write
#   params+norm+details per checkpoint, append a ledger line per eval, keep best/passed
#   dirs, support resume from latest. C: nothing is lost and every policy is explained.
# =====================================================================
"""Drive-persistent checkpoints: params + obs-norm + details JSON + progress ledger."""
from __future__ import annotations
import json
import os
import shutil
import numpy as np
from flax import serialization


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def save_policy(save_dir: str, tag: str, params, norm, details: dict) -> str:
    """Write one policy bundle: params.msgpack + norm.npz + details.json -> save_dir/tag/."""
    d = _ensure(os.path.join(save_dir, tag))
    with open(os.path.join(d, "params.msgpack"), "wb") as f:
        f.write(serialization.to_bytes(params))
    np.savez(os.path.join(d, "norm.npz"),
             mean=np.asarray(norm.mean), var=np.asarray(norm.var), count=np.asarray(norm.count))
    with open(os.path.join(d, "details.json"), "w") as f:
        json.dump(details, f, indent=2, default=str)
    return d


def load_policy(save_dir: str, tag: str, template_params, norm_cls):
    """Restore (params, norm, details) from save_dir/tag/ using a params template + RunningNorm class."""
    d = os.path.join(save_dir, tag)
    with open(os.path.join(d, "params.msgpack"), "rb") as f:
        params = serialization.from_bytes(template_params, f.read())
    z = np.load(os.path.join(d, "norm.npz"))
    import jax.numpy as jnp
    norm = norm_cls(jnp.asarray(z["mean"]), jnp.asarray(z["var"]), jnp.asarray(z["count"]))
    details = {}
    dj = os.path.join(d, "details.json")
    if os.path.exists(dj):
        with open(dj) as f:
            details = json.load(f)
    return params, norm, details


def checkpoint(save_dir: str, update: int, params, norm, details: dict) -> str:
    """Standard per-eval checkpoint: save under jax_ckpt_<update> + update latest_step.txt."""
    _ensure(save_dir)
    tag = f"jax_ckpt_{update:07d}"
    path = save_policy(save_dir, tag, params, norm, details)
    with open(os.path.join(save_dir, "latest_step.txt"), "w") as f:
        f.write(str(int(update)))
    return path


def find_latest(save_dir: str):
    """Return (tag, update) of the newest checkpoint, or None if none/no Drive yet."""
    f = os.path.join(save_dir, "latest_step.txt")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        update = int(fh.read().strip())
    tag = f"jax_ckpt_{update:07d}"
    return (tag, update) if os.path.isdir(os.path.join(save_dir, tag)) else None


def append_progress(save_dir: str, row: dict, fname: str = "jax_progress.jsonl") -> None:
    """Append one JSON line to the run ledger so progress is documented + resumable."""
    _ensure(save_dir)
    with open(os.path.join(save_dir, fname), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def save_named(save_dir: str, name: str, params, norm, details: dict) -> str:
    """Save/overwrite a NAMED bundle (e.g. best_policy / passed_40_in_a_row)."""
    d = os.path.join(save_dir, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    return save_policy(save_dir, name, params, norm, details)
