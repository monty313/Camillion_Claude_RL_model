# =====================================================================
# WHEN 2026-06-27 (Phase 2) | WHO Claude for Monty
# WHY  Auto-calibrate training to the machine so we DON'T freeze (over-subscribe RAM) and DON'T
#      waste a paid Colab tier: detect CPU cores / RAM / GPU and pick a memory-safe number of
#      parallel copies + compute threads + device, then print a clear report.
# WHERE src/training/autotune.py
# HOW  Pure detection + simple, honest heuristics. The tiny 3x256 MLP usually runs fastest on CPU,
#      so we prefer CPU unless told otherwise. NOTE: with the single-process DummyVecEnv path the
#      market is stepped one copy at a time, so total CPU use stays modest on a big box -- the
#      report says so honestly. True multi-core saturation = the (separate) multi-worker upgrade.
# DEPENDS_ON: os; (optional) psutil, torch
# USED_BY: src/training/trainer.train_portfolio
# CHANGE_NOTES(IRAC): I: owner wants ~70-80% utilisation, no freeze, no wasted paid time. R: detect
#   and calibrate. A: cores/RAM/GPU detection + memory-safe n_envs + thread/device pick + honest
#   report. C: sensible resource use out of the box; an explicit, truthful utilisation picture.
# =====================================================================
"""Detect the machine and choose sane, memory-safe, honest training settings."""
from __future__ import annotations
import os


def _ram_gb():
    """(total_gb, available_gb) or (None, None). psutil if present, else /proc/meminfo, else unknown."""
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.total / 1e9, m.available / 1e9
    except Exception:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, rest = line.partition(":")
                    info[k.strip()] = rest
            tot = int(info["MemTotal"].split()[0]) / 1e6
            av = int(info.get("MemAvailable", info["MemFree"]).split()[0]) / 1e6
            return tot, av
        except Exception:
            return None, None


def detect() -> dict:
    """What hardware do we have? cores, RAM (GB), and GPU name (or None)."""
    cores = os.cpu_count() or 2
    tot, av = _ram_gb()
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"cores": int(cores), "ram_total_gb": tot, "ram_avail_gb": av, "gpu": gpu}


def autotune(*, target_util: float = 0.80, prefer_cpu: bool = True, max_envs: int = 8,
             per_env_gb: float = 11.0, apply: bool = True, verbose: bool = True) -> dict:
    """Return {n_envs, threads, device, ...} tuned to this machine.

    - threads: a few intra-op compute threads (the tiny 3x256 MLP does NOT benefit from many; too many
      hurts), capped so we never oversubscribe.
    - n_envs: memory-SAFE number of parallel copies. With the single-process path these add batch
      diversity, not extra cores; we still cap by RAM so we never freeze.
    - device: CPU by default (fastest for this tiny net). GPU only if prefer_cpu=False and one exists.
    `per_env_gb` is a conservative RAM estimate per parallel copy of the full-history portfolio env.
    """
    d = detect()
    cores = d["cores"]
    device = "cuda" if (d["gpu"] and not prefer_cpu) else "cpu"
    # WORKER PROCESSES (true multi-core): aim for ~target_util of the cores, but never over-subscribe RAM
    # (reserve one env's worth for the parent process). Each worker is a separate process stepping the
    # market in parallel, so workers ~= cores used. RAM is the hard cap that prevents the freeze.
    by_cores = max(1, round(cores * target_util))
    if d["ram_avail_gb"]:
        by_ram = max(1, int((d["ram_avail_gb"] * target_util - per_env_gb) // per_env_gb))
    else:
        by_ram = by_cores
    n_workers = max(1, min(max_envs, by_cores, by_ram))
    use_subproc = (n_workers >= 2) and (os.name == "posix")   # multi-process only when it actually helps
    n_envs = n_workers
    # compute threads: with many WORKERS we want FEW threads each (avoid oversubscription); single-process
    # benefits from a handful. Keep small -- the 3x256 MLP never needs many.
    threads = 1 if use_subproc else max(1, min(4, round(cores * target_util)))
    if apply:
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    if verbose:
        ram = f"{d['ram_total_gb']:.0f} GB" if d["ram_total_gb"] else "unknown RAM"
        avail = f"{d['ram_avail_gb']:.0f} GB free" if d["ram_avail_gb"] else "free RAM unknown"
        mode = f"MULTI-CORE: {n_workers} worker processes" if use_subproc else f"single process ({n_envs} copies)"
        print(f"      [autotune] machine: {cores} CPU cores · {ram} ({avail}) · GPU: {d['gpu'] or 'none'}",
              flush=True)
        print(f"      [autotune] using: {mode} · {threads} thread(s) each · device={device}", flush=True)
        if device == "cpu" and d["gpu"]:
            print("      [autotune] (a GPU is present but the model is tiny — CPU is faster; "
                  "pass prefer_cpu=False to force GPU)", flush=True)
        if not use_subproc:
            print("      [autotune] note: single process here (few cores / tight RAM) — on a bigger paid tier "
                  "this auto-scales to multi-core workers to use ~70–80%.", flush=True)
    return {"n_envs": int(n_envs), "n_workers": int(n_workers), "use_subproc": bool(use_subproc),
            "threads": int(threads), "device": device, **d}
