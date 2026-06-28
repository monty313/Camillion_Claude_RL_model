# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Run the JAX parity tests WITHOUT pytest (stdlib only), mirroring the repo's
#      tools/run_tests.py convention so Monty can verify on Colab with one command.
# WHERE jax_tpu/tests/run_parity.py
# HOW   python jax_tpu/tests/run_parity.py   (skips cleanly if jax is not installed)
# =====================================================================
"""Stdlib runner for the JAX parity tests:  python jax_tpu/tests/run_parity.py"""
from __future__ import annotations
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    try:
        import jax  # noqa: F401
    except Exception:
        print("jax not installed -> skipping JAX parity tests (install jax[cpu] flax optax).")
        return 0
    from jax_tpu.tests import test_jax_parity as T1
    from jax_tpu.tests import test_jax_indicators as T2
    from jax_tpu.tests import test_jax_portfolio_parity as T3
    tests = [(m.__name__, name, getattr(m, name))
             for m in (T1, T2, T3) for name in dir(m) if name.startswith("test_")]
    failed = 0
    for mod, name, fn in tests:
        try:
            fn()
            print(f"  PASS  {mod}.{name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {mod}.{name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
