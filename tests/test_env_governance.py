# Governance: the environment FINGERPRINT (CPU/GPU + run-to-run "same env" check) and the
# append-only TRAINING LEDGER (which policy to follow). No torch needed.
import os
import tempfile
from src.training.env_fingerprint import env_fingerprint, env_spec
from src.training import run_log


def test_fingerprint_is_deterministic_and_sensitive():
    fp1, fp2 = env_fingerprint(), env_fingerprint()
    assert fp1 == fp2 and len(fp1) == 12                 # same env -> same 12-char hash
    # a different alpha roster -> a different fingerprint (so the runs aren't mistaken as comparable)
    assert env_fingerprint(alpha_names=["a", "b", "c"]) != env_fingerprint(alpha_names=["a", "b"])
    # order doesn't matter (roster is a set), so CPU/GPU registering in any order still matches
    assert env_fingerprint(alpha_names=["a", "b"]) == env_fingerprint(alpha_names=["b", "a"])


def test_env_spec_captures_behaviour_defining_parts():
    spec = env_spec()
    assert spec["contract_version"] and spec["obs_total"] > 0
    for k in ("asset_classes", "alphas", "ftmo", "reward"):
        assert k in spec
    assert "daily_target_pct" in spec["ftmo"]            # FTMO rules are part of the identity


def test_ledger_log_load_and_best_by_passrate():
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd); os.remove(path)
    try:
        run_log.log_run(run_id="cpu-1", fingerprint="abc", trainer="cpu", pass_rate=0.40, path=path)
        run_log.log_run(run_id="gpu-1", fingerprint="abc", trainer="gpu", pass_rate=0.70, path=path)
        run_log.log_run(run_id="cpu-2", fingerprint="abc", trainer="cpu", pass_rate=0.95,
                        status="rejected", path=path)                      # rejected -> ignored
        run_log.log_run(run_id="other", fingerprint="zzz", trainer="cpu", pass_rate=0.99, path=path)
        assert len(run_log.load_runs(path)) == 4
        # overall best (non-rejected): the other-env run at 0.99
        assert run_log.best_run(path)["run_id"] == "other"
        # WHICH POLICY TO FOLLOW for env 'abc': gpu-1 (0.70); cpu-2 excluded (rejected)
        b = run_log.best_run(path, fingerprint="abc")
        assert b["run_id"] == "gpu-1" and b["trainer"] == "gpu"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_ledger_rejects_bad_status():
    raised = False
    try:
        run_log.log_run(run_id="x", fingerprint="a", status="bogus", path="/tmp/_none.jsonl")
    except ValueError:
        raised = True
    assert raised
