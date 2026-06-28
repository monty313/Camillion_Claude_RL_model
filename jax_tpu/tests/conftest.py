# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Make the parity tests runnable from anywhere: put the repo root on sys.path
#      and skip the whole module cleanly when JAX isn't installed (Colab-CPU sessions
#      and the default `pytest -q` on a box without jax should NOT error).
# WHERE jax_tpu/tests/conftest.py
# =====================================================================
"""Pytest fixtures/skips for the JAX parity tests."""
from __future__ import annotations
import os
import sys
import pytest

# repo root = two levels up from this file (jax_tpu/tests/conftest.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Skip every test in this folder if JAX is absent (keeps `pytest -q` green CPU-only).
collect_ignore_glob = []
try:  # pragma: no cover
    import jax  # noqa: F401
except Exception:  # pragma: no cover
    collect_ignore_glob = ["test_*.py"]


def pytest_configure(config):  # pragma: no cover
    try:
        import jax  # noqa: F401
    except Exception:
        pytest.skip("jax not installed — JAX/TPU parity tests skipped", allow_module_level=True)
