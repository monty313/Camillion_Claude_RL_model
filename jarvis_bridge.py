# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  The read-only HTTP bridge between the Camillion bot and the JARVIS cockpit
#      HUD. Serves GET /state (the data contract) + GET /council (OMEGA->JUSTICE->
#      JARVIS grounded, progressive advice) + the static HUD. FastAPI is imported
#      LAZILY so the repo's stdlib test runner (no FastAPI) can still import the
#      contract logic; the bridge is structurally READ-ONLY (GET routes only --
#      it can never place or modify a trade).
# WHERE jarvis_bridge.py (repo root)
# HOW  create_app(provider) wires the routes; main() runs uvicorn. The contract is
#      src/jarvis/state_contract.build_state; the reasoning is src/jarvis/council.
# DEPENDS_ON: src/jarvis/{state_contract,state_provider,council}.py
#             (+ fastapi/uvicorn ONLY when actually serving)
# USED_BY: the HUD's pullLive() (fetches /state) + ASK JARVIS / IRAC loop (/council)
# CHANGE_NOTES(IRAC): I: the HUD needs a live, honest, read-only feed + the council's
#   reasoning. R: HANDOFF contract + read-only guarantee + lazy deps. A: GET-only
#   FastAPI app (state/council/health) over a headless StateProvider, static HUD
#   mount, deps imported lazily. C: the cockpit runs on real bot state and JARVIS
#   advises (never trades) -- the operator gets system-grounded, progressive guidance.
# =====================================================================
"""Read-only JARVIS bridge: GET /state + GET /council + the static HUD (FastAPI lazy)."""
from __future__ import annotations
import os


def create_app(provider=None):
    """Build the read-only FastAPI app. FastAPI is imported here, not at module load."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from src.jarvis.state_contract import build_state
    from src.jarvis.council import deliberate, answer
    from src.jarvis import knowledge as KB

    if provider is None:
        from src.jarvis.state_provider import StateProvider
        provider = StateProvider.from_synthetic()

    app = FastAPI(title="JARVIS bridge (read-only)")
    # GET-only CORS: the HUD may be opened from a file:// or another origin.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    @app.get("/state")
    def state():
        provider.step()                       # advance the sim one bar per poll (read-only)
        return build_state(provider.snapshot())

    @app.get("/council")
    def council(use_llm: str = "auto", chat: str = ""):
        """OMEGA -> JUSTICE -> JARVIS deliberation over the live state (grounded + progressive).

        `chat` is an optional URL-encoded JSON array of recent {role,text} turns so the council
        reasons WITH the conversation. Read-only: it only shapes advice, never a trade.
        """
        import json
        try:
            hist = json.loads(chat) if chat else None
            hist = hist if isinstance(hist, list) else None
        except Exception:
            hist = None
        return deliberate(build_state(provider.snapshot()), chat_history=hist, use_llm=use_llm)

    @app.get("/knowledge")
    def knowledge(q: str = ""):
        """How the bot works + the troubleshooting fixes (search with ?q=). Read-only."""
        return {"system": KB.SYSTEM_SUMMARY, "fixes": KB.search(q) if q else KB.TROUBLESHOOTING}

    @app.get("/ask")
    def ask(q: str, use_llm: str = "auto"):
        """Ask JARVIS how to fix something; grounded in the knowledge base + the live state."""
        return answer(q, state=build_state(provider.snapshot()), use_llm=use_llm)

    @app.get("/health")
    def health():
        return {"ok": True, "model_attached": provider.policy is not None}

    # serve the HUD + support.js from the repo root LAST (so /state etc. take priority)
    here = os.path.dirname(os.path.abspath(__file__))
    app.mount("/", StaticFiles(directory=here, html=True), name="static")
    return app


def main(host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "FastAPI/uvicorn not installed. The contract lives in src/jarvis/state_contract.py "
            "and the reasoning in src/jarvis/council.py (both pure, testable). "
            "Run `pip install fastapi uvicorn` to serve the cockpit, then "
            "`uvicorn jarvis_bridge:app --port 8000`.")
    uvicorn.run(create_app(), host=host, port=port)


# `uvicorn jarvis_bridge:app` looks for a module-level `app`; build it lazily on access.
def __getattr__(name):
    if name == "app":
        return create_app()
    raise AttributeError(name)


if __name__ == "__main__":   # pragma: no cover
    main()
