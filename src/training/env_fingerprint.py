# =====================================================================
# WHEN 2026-06-25 | WHO Claude for Monty
# WHY  A short HASH of EVERYTHING that defines the environment's behaviour, so any two training
#      runs can be checked for "same environment" -> their policies are directly comparable.
#      This is the CPU<->GPU PARITY ANCHOR and the run-to-run "are we still training the same
#      thing?" check. Same fingerprint = identical obs contract + alphas + FTMO rules + reward.
# WHERE src/training/env_fingerprint.py
# DEPENDS_ON: config/constants.py, config/variables.py, config/ftmo_config.py, strategies pack
# USED_BY: src/training/run_log.py (stamps every run), notebooks, tests, docs/ENVIRONMENT_STATE.md
#
# ============ RULES FOR THE FUTURE GPU TRAINER (read BEFORE building it) ============
#  1. PARITY HASH: the GPU env MUST produce the SAME `env_fingerprint()` as the CPU env for the
#     same config. The fingerprint covers the obs contract (version + size), the alpha roster,
#     the FTMO rules, and the reward shaping -- so a matching hash means "same environment".
#  2. STEP-PARITY: beyond the hash, the GPU env MUST return the SAME observation + reward as the
#     CPU env on identical inputs. Add a parity test that steps BOTH on the same bars and asserts
#     they match (within float tolerance). A matching fingerprint with mismatched steps = a bug.
#  3. NEVER DIVERGE SILENTLY: if you deliberately change behaviour, BUMP
#     OBSERVATION_CONTRACT_VERSION (which changes the fingerprint) AND update
#     docs/ENVIRONMENT_STATE.md. CPU and GPU policies are only comparable at the SAME fingerprint.
#  4. SAME POLICY FORMAT: keep the obs size + action space identical, so a GPU-trained policy and
#     a CPU-trained policy are the same file format and can be ranked together by pass-rate.
# ===================================================================================
# =====================================================================
"""Environment fingerprint -- the CPU/GPU + run-to-run 'same environment' check."""
from __future__ import annotations
import hashlib
import json
from config import constants as C
from config import variables as V
from config.ftmo_config import load_active_config

# FTMO/risk knobs that change the environment's behaviour (and thus policy comparability).
_FTMO_KEYS = ("daily_target_pct", "daily_drawdown_pct", "max_total_drawdown_pct",
              "trailing_drawdown_pct", "trailing_enabled", "two_phase_enabled",
              "phase2_trailing_pct", "phase2_continue", "profit_target_total_pct")


def _default_alpha_names() -> list[str]:
    """The current default alpha roster (gravity + the pack). Reads the live registry."""
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    r = AlphaRegistry()
    register_all(r)
    return [s.name for s in r._slots if s is not None]


def env_spec(alpha_names=None, cfg=None) -> dict:
    """The full behaviour-defining spec of the environment -- what makes two runs comparable."""
    cfg = cfg or load_active_config()
    names = list(alpha_names) if alpha_names is not None else _default_alpha_names()
    return {
        "contract_version": C.OBSERVATION_CONTRACT_VERSION,
        "obs_total": int(C.OBS_TOTAL_SIZE),
        "asset_classes": list(C.ASSET_CLASSES),
        "alphas": sorted(names),
        "ftmo": {k: getattr(cfg, k, None) for k in _FTMO_KEYS},
        "reward": {
            "cost_per_side": getattr(V, "TRANSACTION_COST_FRAC_PER_SIDE", None),
            "ny_half_bonus": getattr(V, "FTMO_NY_HALF_TARGET_BONUS", 0.0),
            "ny_full_bonus": getattr(V, "FTMO_NY_FULL_TARGET_BONUS", 0.0),
            "open_gate_cci_threshold": getattr(V, "OPEN_GATE_CCI_THRESHOLD", None),
        },
    }


def env_fingerprint(alpha_names=None, cfg=None) -> str:
    """12-char hash of env_spec(). SAME fingerprint => identical environment => policies are
    comparable (CPU or GPU). A DIFFERENT fingerprint => the runs are NOT comparable -- treat them
    as different experiments. Stamp this on every training run (see run_log.log_run)."""
    blob = json.dumps(env_spec(alpha_names, cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
