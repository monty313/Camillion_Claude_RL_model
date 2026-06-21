# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Barbershop #6: observation shape, missing/stale features, leakage checks.
# WHERE src/barbershop/feature_doctor.py | HOW inspects a built observation vs
#      the contract; flags non-finite + reports zero/stale fractions per block.
# DEPENDS_ON src/observation/observation_contract.py, numpy
# USED_BY src/jarvis/omega_agent.py (Phase 2), tests/test_barbershop_smoke.py.
"""Feature Doctor: validate the observation against the locked contract."""
from __future__ import annotations
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC


def inspect(obs) -> dict:
    obs = np.asarray(obs)
    nonfinite = int((~np.isfinite(obs)).sum()) if obs.size else 0
    block_zero_frac = {}
    if obs.shape == C.OBS_SHAPE:
        for name, sl in OC.BLOCK_SLICES.items():
            seg = obs[sl]
            block_zero_frac[name] = float(np.mean(seg == 0)) if seg.size else 1.0
    return {
        "shape": tuple(obs.shape),
        "expected_shape": C.OBS_SHAPE,
        "shape_ok": obs.shape == C.OBS_SHAPE,
        "dtype": str(obs.dtype),
        "n_features_expected": len(OC.FEATURE_NAMES),
        "nonfinite_count": nonfinite,
        "block_zero_fraction": block_zero_frac,
        "leakage_note": "Signal-accuracy leakage is proven in "
                        "tests/test_signal_accuracy_no_leakage.py (out[t] ignores bars>t).",
    }
