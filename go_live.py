# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  One command to take the JARVIS cockpit LIVE on real data (and an optional
#      trained model) instead of the synthetic demo feed. Wraps StateProvider.from_cache
#      + the read-only bridge so going live is a single line.
# WHERE go_live.py (repo root)
# HOW  python go_live.py --data <cache_dir> --symbol EURUSD [--model models/camillion_ppo] [--port 8000]
#      (or set CAMILLION_CACHE_DIR / CAMILLION_SYMBOL). No --data -> the honest synthetic demo.
# DEPENDS_ON: src/jarvis/state_provider.py, jarvis_bridge.py (fastapi/uvicorn to serve);
#             optional: stable-baselines3 (only if --model is given)
# USED_BY: the operator (Monty) to run the cockpit on real data
# CHANGE_NOTES(IRAC): I: from_cache + a model need manual wiring to go live. R: operator
#   asked for a quick go_live.py. A: a tiny argparse launcher: build the provider (real
#   cache + optional model with its frozen VecNormalize, else honest no-model), then serve
#   the read-only bridge. C: live cockpit on the real bot in one command; still cannot trade.
# =====================================================================
"""go_live.py -- launch the read-only JARVIS cockpit on REAL data (one command)."""
from __future__ import annotations
import argparse
import os


def _load_policy(model_path: str, data_dir: str, symbol: str):
    """Build a policy callable obs->(logits,value) from a trained model WITH its frozen
    VecNormalize stats (so the observation is scaled exactly as at train time). Best-effort:
    returns None (honest no-model feed) if SB3/the files aren't available."""
    try:
        from src.data.cache_builder import load_cache
        from src.training.trainer import load_for_eval, sb3_policy_fn
        from src.strategies.registry import AlphaRegistry
        from src.strategies.alpha_pack import register_all
        ind, close, time_ns = load_cache(data_dir, symbol)
        model, venv = load_for_eval(model_path, ind, close, time_ns,
                                    lambda: _reg(AlphaRegistry, register_all), symbol=symbol)
        return sb3_policy_fn(model, venv)
    except Exception as e:   # pragma: no cover - depends on SB3 + a saved model
        print(f"[go_live] could not load model ({model_path}): {e}\n"
              f"[go_live] continuing with the honest NO-MODEL feed (alpha-consensus, confidence 0).")
        return None


def _reg(AlphaRegistry, register_all):
    r = AlphaRegistry(); register_all(r); return r


def build_provider(data_dir: str | None, symbol: str, model_path: str | None):
    """Build the StateProvider: real cache (+ optional model) if --data, else the synthetic demo."""
    from src.jarvis.state_provider import StateProvider
    if data_dir:
        policy = _load_policy(model_path, data_dir, symbol) if model_path else None
        print(f"[go_live] LIVE on real cache: {data_dir} ({symbol}) | model={'yes' if policy else 'no (honest fallback)'}")
        return StateProvider.from_cache(data_dir, symbol=symbol, policy=policy)
    print("[go_live] no --data given -> running the synthetic DEMO feed (real env + real alphas, placeholder prices).")
    return StateProvider.from_synthetic(symbol=symbol)


def main():
    ap = argparse.ArgumentParser(description="Launch the read-only JARVIS cockpit (live or demo).")
    ap.add_argument("--data", default=os.environ.get("CAMILLION_CACHE_DIR"),
                    help="cache dir built by cache_builder.build_cache (else CAMILLION_CACHE_DIR; else demo)")
    ap.add_argument("--symbol", default=os.environ.get("CAMILLION_SYMBOL", "EURUSD"))
    ap.add_argument("--model", default=None, help="optional trained model path (e.g. models/camillion_ppo)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    provider = build_provider(args.data, args.symbol, args.model)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("[go_live] FastAPI/uvicorn not installed. Run: pip install -r requirements-jarvis.txt")
    from jarvis_bridge import create_app
    print(f"[go_live] serving READ-ONLY cockpit on http://{args.host}:{args.port}  "
          f"(/state /council /ask /knowledge /health). Open /JARVIS%20Cockpit.dc.html")
    uvicorn.run(create_app(provider), host=args.host, port=args.port)


if __name__ == "__main__":   # pragma: no cover
    main()
