# =====================================================================
# WHEN 2026-06-27 (Phase 2) | WHO Claude for Monty
# WHY  Save the EXPENSIVE per-symbol precompute (alphas/streaks/cross-asset/etc.) to disk
#      (Google Drive on Colab) so re-runs SKIP the slow build -- WITHOUT ever loading STALE
#      features. The owner's rule: "specify exactly what it's used for ... no mismatches."
# WHERE src/data/feature_cache.py
# HOW  A COMPLETE fingerprint keys each cache: a content hash of the input arrays (close +
#      indicators + time -> this alone captures the data slice AND all indicator math), the
#      resolved obs-contract values (contract version, MAX_STRATEGIES, asset classes, block
#      sizes), source-hashes of the code that DEFINES the features (strategies + signals +
#      the env precompute + asset_specs -- catches threshold/logic edits that names miss), the
#      slot-ORDERED alpha roster, the per-symbol asset spec, and the resolved open-gate /
#      signal-accuracy values. Load ONLY if the fingerprint matches exactly, else rebuild.
# DEPENDS_ON: numpy; config/constants; config/variables; config/asset_specs; src/indicators/base
# USED_BY: src/env/portfolio_env.build_portfolio_subs (load-or-build-and-save)
# CHANGE_NOTES(IRAC): I: rebuilding 1.8M-bar features every run is slow; a naive (symbol+dates)
#   key would silently load WRONG features after a code/data change. R: owner "no mismatches".
#   A: a fingerprint that folds in data content + contract + feature CODE hashes + ordered roster;
#   load only on exact match. C: fast re-runs with ZERO risk of training on stale features.
# =====================================================================
"""Drive-friendly feature cache with a no-mismatch fingerprint (load only on exact match)."""
from __future__ import annotations
import glob
import hashlib
import json
import os
import numpy as np

from config import constants as C
from config import variables as V
from config import asset_specs as A
from src.indicators.base import ALL_INDICATOR_COLUMNS

# Bump this if the cache FORMAT or the set of saved arrays changes (a manual backstop on top of
# the automatic code/data hashes below).
FEATURE_CACHE_VERSION = "fc-v1"

# The arrays produced by TradingEnv._precompute (+ its sub-methods) -- the expensive part we cache.
# Kept here as the single source of truth for save/load; TradingEnv.export_precomputed mirrors it.
PRECOMPUTED_ARRAY_KEYS = [
    "alpha_matrix", "occupancy", "net_signal", "sig_acc", "time_feats", "open_gate_blocked",
    "streak_matrix", "ref_move", "cross_asset_matrix",
    "_today_sofar", "_prev_day", "_prev2_day", "_week_avg",
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _sha(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _arrays_hash(close, ind, time_ns) -> str:
    """Content hash of the INPUT arrays in canonical dtypes. This single hash captures the exact
    data slice AND all upstream indicator math (the indicators array bakes it in) -- so a data
    revision, a date-range change, or an indicator-formula edit all change this hash."""
    return _sha(
        np.ascontiguousarray(close, dtype=np.float64).tobytes(),
        np.ascontiguousarray(ind, dtype=np.float32).tobytes(),
        np.ascontiguousarray(time_ns, dtype=np.int64).tobytes(),
    )


def _source_hash(rel_globs) -> str:
    """Hash the SOURCE BYTES of the code that defines the features. Catches logic/threshold edits
    that leave names unchanged (the gap in env_fingerprint). Path-sorted for determinism."""
    paths: list[str] = []
    for g in rel_globs:
        paths.extend(glob.glob(os.path.join(_REPO_ROOT, g), recursive=True))
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        rel = os.path.relpath(p, _REPO_ROOT)
        h.update(rel.encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _code_hashes() -> dict:
    """Source hashes grouped by what they cover (also human-useful in the manifest)."""
    return {
        # alpha LOGIC + slot wiring + the signal math built on top of it
        "strategy_code": _source_hash(["src/strategies/**/*.py", "src/signals/**/*.py"]),
        # the precompute itself (rolling-window literals, cross-asset/recent-context, time features)
        "precompute_code": _source_hash(["src/env/trading_env.py", "src/observation/builder.py"]),
        # symbol -> asset-class / one-hot / typical-ATR resolution
        "asset_specs_code": _source_hash(["config/asset_specs.py"]),
        # v1.6.0 aux feed: OHLC obs block + ADX-DI side-channel that the alphas depend on. Hashing this
        # code busts the cache if the DI/OHLC math changes (the alpha_matrix bakes the DI alpha outputs in).
        "aux_features_code": _source_hash(["src/data/aux_features.py", "src/indicators/adx.py"]),
    }


def _alpha_roster(registry) -> list:
    """Slot-ORDERED roster [[slot, name], ...] for occupied slots. Order matters (env_fingerprint's
    SORTED names miss a slot reassignment), so we keep slot index + name in order."""
    out = []
    slots = getattr(registry, "_slots", [])
    for i, s in enumerate(slots):
        if s is not None:
            out.append([i, getattr(s, "name", str(s))])
    return out


def _aux_hash(aux) -> str:
    """Content hash of the v1.6.0 aux array (OHLC+DI), or 'none'. The alpha_matrix bakes the ADX-DI
    alpha outputs in, and aux needs raw high/low that the indicator data_hash does NOT capture -- so a
    cache built WITH aux must never be loaded as one built WITHOUT it (and vice versa). 'no mismatches'."""
    if aux is None:
        return "none"
    return _sha(np.ascontiguousarray(aux, dtype=np.float32).tobytes())


def fingerprint(symbol, ind, close, time_ns, registry, *, open_gate_threshold=None, aux=None) -> tuple[str, dict]:
    """Return (key, manifest). `key` is the exact-match cache key; `manifest` is the human-readable
    record of EXACTLY what the cache is for + every fingerprint component."""
    thr = float(V.OPEN_GATE_CCI_THRESHOLD if open_gate_threshold is None else open_gate_threshold)
    t = np.ascontiguousarray(time_ns, dtype=np.int64).ravel()
    manifest = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "symbol": symbol,
        "contract_version": C.OBSERVATION_CONTRACT_VERSION,
        "max_strategies": int(C.MAX_STRATEGIES),
        "asset_classes": list(C.ASSET_CLASSES),
        "obs_block_cross_asset": int(C.OBS_BLOCK_CROSS_ASSET),
        "obs_block_ohlc": int(C.OBS_BLOCK_OHLC),   # v1.6.0 raw OHLC block
        "obs_block_time": int(C.OBS_BLOCK_TIME),
        "n_indicators": int(C.N_INDICATORS_TOTAL),
        "indicator_columns_hash": _sha(json.dumps(list(ALL_INDICATOR_COLUMNS)).encode()),
        "alpha_roster": _alpha_roster(registry),
        "open_gate_cci_threshold": thr,
        "signal_accuracy_window": int(V.SIGNAL_ACCURACY_WINDOW),
        "per_symbol_spec": {
            "asset_class": A.asset_class(symbol),
            "class_one_hot": list(A.class_one_hot(symbol)),
            "typical_atr": A.typical_atr(symbol),
        },
        "code_hashes": _code_hashes(),
        "data": {
            "data_hash": _arrays_hash(close, ind, time_ns),
            "aux_hash": _aux_hash(aux),   # v1.6.0: OHLC+DI content (or 'none') -> no aux/no-aux cross-loading
            "n_bars": int(t.shape[0]),
            "first_ts": int(t[0]) if t.size else 0,
            "last_ts": int(t[-1]) if t.size else 0,
        },
    }
    key = _sha(json.dumps(manifest, sort_keys=True, default=str).encode())
    manifest["key"] = key
    return key, manifest


def _date_tag(ts_ns: int) -> str:
    try:
        return str(np.datetime64(int(ts_ns), "ns").astype("datetime64[D]"))
    except Exception:
        return "na"


def cache_subdir(base: str, symbol: str, time_ns, key: str) -> str:
    """Human-readable, fingerprint-safe folder name: SYMBOL__from_to__contract__key8."""
    t = np.ascontiguousarray(time_ns, dtype=np.int64).ravel()
    frm = _date_tag(t[0]) if t.size else "na"
    to = _date_tag(t[-1]) if t.size else "na"
    name = f"{symbol}__{frm}_{to}__{C.OBSERVATION_CONTRACT_VERSION}__{key[:8]}"
    return os.path.join(base, name)


def load(base, symbol, ind, close, time_ns, registry, *, open_gate_threshold=None, aux=None):
    """Return the cached precomputed arrays dict ONLY if the fingerprint matches EXACTLY, else None
    (so the caller rebuilds). Never returns stale/mismatched features. `aux` (OHLC+DI) is part of the
    fingerprint so a no-aux cache is never loaded as an aux one (v1.6.0)."""
    if not base:
        return None
    key, _ = fingerprint(symbol, ind, close, time_ns, registry, open_gate_threshold=open_gate_threshold, aux=aux)
    d = cache_subdir(base, symbol, time_ns, key)
    npz, man = os.path.join(d, "features.npz"), os.path.join(d, "manifest.json")
    if not (os.path.isfile(npz) and os.path.isfile(man)):
        return None
    try:
        with open(man) as f:
            if json.load(f).get("key") != key:      # exact-match guard (defends against key8 collision)
                return None
        with np.load(npz, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None                                  # any corruption -> treat as miss, rebuild


def save(base, symbol, ind, close, time_ns, registry, env, *, open_gate_threshold=None, built: str | None = None,
         aux=None):
    """Write features.npz + a plain-English manifest.json describing EXACTLY what this cache is.
    Returns the folder path. `env` must expose export_precomputed() -> {array_name: ndarray}. `aux`
    (OHLC+DI) defaults to the env's own aux so the saved key reflects what actually built alpha_matrix."""
    if not base:
        return None
    if aux is None:
        aux = getattr(env, "aux", None)
    key, manifest = fingerprint(symbol, ind, close, time_ns, registry, open_gate_threshold=open_gate_threshold, aux=aux)
    d = cache_subdir(base, symbol, time_ns, key)
    os.makedirs(d, exist_ok=True)
    arrays = env.export_precomputed()
    np.savez_compressed(os.path.join(d, "features.npz"), **arrays)
    manifest["built"] = built or ""
    manifest["what_this_is"] = (f"Prepared market features for {symbol} "
                                f"({manifest['data']['n_bars']:,} bars, {_date_tag(manifest['data']['first_ts'])} "
                                f"-> {_date_tag(manifest['data']['last_ts'])}, contract {C.OBSERVATION_CONTRACT_VERSION}). "
                                f"Loaded only if the fingerprint matches exactly; otherwise it is rebuilt.")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return d


def default_cache_dir() -> str:
    """Where to keep the feature cache. Prefer the env override, then Google Drive (Colab, persists
    across sessions), else a local folder. Organized under .../Camillion/feature_cache."""
    env = os.environ.get("CAMILLION_FEATURE_CACHE_DIR")
    if env:
        return env
    drive = "/content/drive/MyDrive"
    if os.path.isdir(drive):
        return os.path.join(drive, "Camillion", "feature_cache")
    return os.path.join(_REPO_ROOT, "feature_cache")
