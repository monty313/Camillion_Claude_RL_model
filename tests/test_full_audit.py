# =====================================================================
# WHEN 2026-06-26 | WHO Claude for Mark | WHERE tests/test_full_audit.py
# WHY  pytest mirror of the brutal full-system audit (tools/run_full_audit.py).
#      Same checks, as standard pytest functions, tagged by severity so Mark can run
#      just the critical ones:  pytest tests/test_full_audit.py -m critical
# HOW  pytest tests/test_full_audit.py -v        (needs stable-baselines3 + torch)
#      The fast stdlib runner (tools/run_tests.py) SKIPS this heavy diagnostic — run it
#      directly with `python tools/run_full_audit.py` instead.
# DEPENDS_ON tools/run_full_audit.py (single source of truth for the test logic)
# =====================================================================
"""pytest wrapper around tools/run_full_audit.py — one test per audit check, severity-marked.

Dual-mode on purpose:
  * fast stdlib runner  -> test_full_system_audit() SKIPS (heavy: spins up real PPO) unless RUN_FULL_AUDIT=1
  * pytest              -> one parametrized, severity-marked test per check (run -m critical for the subset)
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pytest
    _HAS_PYTEST = True
except Exception:                       # the stdlib runner has no pytest — stay importable
    pytest = None
    _HAS_PYTEST = False

# Load the harness as the single source of truth (cheap: it only defines functions at import).
_spec = importlib.util.spec_from_file_location("run_full_audit", os.path.join(ROOT, "tools", "run_full_audit.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _sb3_present():
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------- stdlib-runner entry point
def test_full_system_audit():
    """Heavy diagnostic — SKIPPED in the fast suite. Run it with `python tools/run_full_audit.py`
    (or `RUN_FULL_AUDIT=1 python tools/run_tests.py`, or `pytest tests/test_full_audit.py`)."""
    if os.environ.get("RUN_FULL_AUDIT") != "1":
        print("SKIP test_full_system_audit: heavy audit — run `python tools/run_full_audit.py`")
        return
    if not _sb3_present():
        print("SKIP test_full_system_audit: stable-baselines3 / torch not installed")
        return
    failures = []
    for tid, name, cat, sev, fn, scored in A.TESTS:
        try:
            status, msg = fn()
        except Exception as e:                       # a crash is a hard failure
            status, msg = A.FAIL, str(e)
        if status == A.FAIL:
            failures.append(f"[{sev}] {tid} {name}: {msg}")
    assert not failures, "audit FAILs:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------- pytest parametrized version
if _HAS_PYTEST:
    _MARK = {"CRITICAL": pytest.mark.critical, "HIGH": pytest.mark.high, "MEDIUM": pytest.mark.medium}

    def _cases():
        if not _sb3_present():
            return [pytest.param(None, None, None, None,
                                 marks=pytest.mark.skip(reason="stable-baselines3 / torch not installed"),
                                 id="audit-skipped-no-sb3")]
        return [pytest.param(tid, name, fn, sev, marks=_MARK[sev], id=f"{tid}-{name.replace(' ', '_')}")
                for tid, name, cat, sev, fn, scored in A.TESTS]

    @pytest.mark.parametrize("tid,name,fn,sev", _cases())
    def test_audit(tid, name, fn, sev):
        """A FAIL is a hard pytest failure; PASS/WARNING/SKIP all pass (warnings are gaps, not bugs)."""
        status, msg = fn()
        assert status != A.FAIL, f"[{sev}] {tid} {name} FAILED: {msg}"
        if status == A.WARN:
            print(f"\nWARNING {tid} {name}: {msg}")

    def pytest_configure(config):  # pragma: no cover - pytest hook
        for m in ("critical", "high", "medium"):
            config.addinivalue_line("markers", f"{m}: audit severity {m}")
