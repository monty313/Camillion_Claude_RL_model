# =====================================================================
# WHEN 2026-06-25 | WHO Claude for Monty
# WHY  Append-only TRAINING LEDGER so we NEVER lose track of which policy to trust. One JSONL line
#      per training run: its environment fingerprint + FTMO walk-forward results + where the model
#      lives. `best_run(fingerprint=...)` answers "which policy do we follow?" -- the highest
#      pass-rate among runs of the SAME environment (so CPU and GPU runs are compared like-for-like).
# WHERE src/training/run_log.py
# DEPENDS_ON: env_fingerprint (caller passes the fingerprint) | USED_BY: trainer, notebooks, tests
# CHANGE_NOTES(IRAC): I: many runs (CPU/GPU, seeds) -> confusion over the "real" policy. R: operator
#   "keep very good records of each training vs passing FTMO". A: JSONL ledger + best-by-pass-rate
#   per fingerprint. C: one source of truth for the current best, never lost.
# =====================================================================
"""Append-only training-run ledger (JSONL): config fingerprint + FTMO results per run."""
from __future__ import annotations
import json
import os

LEDGER_PATH = os.environ.get("CAMILLION_RUN_LEDGER", "records/run_ledger.jsonl")
VALID_STATUS = ("candidate", "best", "rejected")


def log_run(*, run_id, fingerprint, contract_version="", trainer="cpu", symbols=(), seed=None,
            total_timesteps=0, pass_rate=None, breaches=None, n_windows=None,
            status="candidate", model_path="", notes="", timestamp="", path=None) -> dict:
    """Append ONE run record to the ledger and return it.
      run_id          : your unique name for the run.
      fingerprint     : env_fingerprint() -- ties the run to an exact environment.
      trainer         : 'cpu' or 'gpu' (SAME fingerprint => comparable regardless).
      pass_rate       : walk-forward FTMO pass-rate (the headline 'did it pass' number).
      status          : candidate | best | rejected.
    """
    if status not in VALID_STATUS:
        raise ValueError(f"status {status!r} not in {VALID_STATUS}")
    rec = {"run_id": str(run_id), "timestamp": str(timestamp), "fingerprint": str(fingerprint),
           "contract_version": str(contract_version), "trainer": str(trainer),
           "symbols": list(symbols), "seed": seed, "total_timesteps": int(total_timesteps),
           "pass_rate": pass_rate, "breaches": breaches, "n_windows": n_windows,
           "status": str(status), "model_path": str(model_path), "notes": str(notes)}
    p = path or LEDGER_PATH
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_runs(path=None) -> list[dict]:
    """All run records, in order logged."""
    p = path or LEDGER_PATH
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def best_run(path=None, *, fingerprint=None) -> dict | None:
    """The non-rejected run with the HIGHEST walk-forward pass-rate -- i.e. WHICH POLICY TO FOLLOW.
    Pass `fingerprint` to compare only runs of the SAME environment (the right way to rank CPU vs
    GPU vs seeds). Returns None if there are no scored, non-rejected runs."""
    runs = [r for r in load_runs(path)
            if r.get("status") != "rejected" and r.get("pass_rate") is not None
            and (fingerprint is None or r.get("fingerprint") == fingerprint)]
    return max(runs, default=None, key=lambda r: r["pass_rate"])
