# Auto-calibrate (2026-06-27): pick memory-safe, sane training settings from the machine so it never
# freezes (over-subscribes RAM) and doesn't waste a paid tier. These tests lock the basic guarantees.
from src.training.autotune import autotune, detect


def test_detect_returns_basics():
    d = detect()
    assert d["cores"] >= 1
    assert "gpu" in d and "ram_total_gb" in d


def test_autotune_is_sane():
    s = autotune(apply=False, verbose=False)
    assert s["n_envs"] >= 1
    assert 1 <= s["threads"] <= 8                 # tiny model: a few threads, never oversubscribe
    assert s["device"] in ("cpu", "cuda")


def test_autotune_is_memory_safe():
    # An absurd per-copy RAM estimate must collapse to a single copy (never freeze by over-subscribing).
    s = autotune(apply=False, verbose=False, per_env_gb=1e6)
    if s.get("ram_avail_gb"):
        assert s["n_envs"] == 1


def test_autotune_prefers_cpu_for_tiny_model():
    s = autotune(apply=False, verbose=False, prefer_cpu=True)
    assert s["device"] == "cpu"
