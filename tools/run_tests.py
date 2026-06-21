#!/usr/bin/env python3
# WHEN 2026-06-21 | WHO Claude for Monty | WHY run the pytest-style tests with
# ONLY the stdlib (pytest can't be installed in some sandboxes/Colab-free).
# HOW import each tests/test_*.py and call every test_* function.
# Colab/users with pytest can instead just run:  pytest -q
import glob, importlib.util, os, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    total = passed = failed = 0
    failures = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # import-time failure
            print(f"[IMPORT FAIL] {name}: {exc}")
            traceback.print_exc()
            failed += 1
            failures.append(name)
            continue
        for fn_name in sorted(dir(mod)):
            if not fn_name.startswith("test_"):
                continue
            fn = getattr(mod, fn_name)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                passed += 1
                print(f"  PASS  {name}.{fn_name}")
            except Exception as exc:
                failed += 1
                failures.append(f"{name}.{fn_name}")
                print(f"  FAIL  {name}.{fn_name}: {exc}")
                traceback.print_exc()
    print(f"\n==== {passed}/{total} passed, {failed} failed ====")
    if failures:
        print("Failures:", ", ".join(failures))
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
