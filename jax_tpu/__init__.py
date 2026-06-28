# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Package marker for the JAX/TPU trainer. EVERYTHING JAX lives in this ONE
#      folder so it never gets confused with the CPU reference code in src/.
# WHERE jax_tpu/__init__.py
# HOW   Empty package marker. See PLAN.md for the full build plan + the exact
#       numbers the JAX code must reproduce to stay bar-for-bar with the CPU env.
# DEPENDS_ON: (nothing)
# USED_BY: jax_tpu/* modules, jax_tpu/tests/*, the Colab notebook
# CHANGE_NOTES(IRAC): I: keep JAX isolated. R: operator "one folder". A: package
#   marker. C: a clean, self-contained second implementation of the same env.
# =====================================================================
"""Camillion JAX/TPU trainer — a second, on-device implementation of the CPU env.

The CPU env (`src/env/trading_env.py`) is the reference. This package must match it
bar-for-bar (479 obs, same reward, same FTMO rules, same `env_fingerprint`). See PLAN.md.
"""
__all__ = []
