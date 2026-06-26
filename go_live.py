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


def build_provider(data_dir: str | None, symbols, model_path: str | None):
    """Build the PORTFOLIO view (MarketView over the whole FTMO universe): real cache (+ optional
    one policy across the book) if --data, else the honest synthetic demo. The bot trades all symbols."""
    from src.jarvis.market_view import MarketView
    syms = list(symbols) if isinstance(symbols, (list, tuple)) else [symbols]
    if data_dir:
        policy = _load_policy(model_path, data_dir, syms[0]) if model_path else None
        print(f"[go_live] LIVE on real cache: {data_dir} | universe={syms} | "
              f"model={'yes' if policy else 'no (honest fallback)'}")
        return MarketView.from_cache(data_dir, syms, policy=policy)
    print(f"[go_live] no --data given -> synthetic DEMO portfolio (real env + real alphas) over {syms}.")
    return MarketView.from_synthetic(syms)


def main():
    from config import variables as V
    ap = argparse.ArgumentParser(description="Launch the read-only JARVIS cockpit (portfolio; live or demo).")
    ap.add_argument("--data", default=os.environ.get("CAMILLION_CACHE_DIR"),
                    help="cache dir built by cache_builder.build_cache (else CAMILLION_CACHE_DIR; else demo)")
    ap.add_argument("--symbols", default=os.environ.get("CAMILLION_SYMBOLS", ",".join(V.SYMBOLS)),
                    help="comma-separated FTMO universe to trade as a portfolio (default: config SYMBOLS)")
    ap.add_argument("--model", default=None, help="optional trained model path (e.g. models/camillion_ppo)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    provider = build_provider(args.data, symbols, args.model)
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
