# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  The JARVIS COUNCIL: OMEGA -> JUSTICE -> JARVIS reason TOGETHER over the live
#      system + the chat history, talk to each other (each sees the prior speakers),
#      stay grounded in the logic of the system (every claim cites a real number from
#      consistency.analyze_consistency), and ALWAYS end on the single next improvement
#      toward passing the FTMO challenge CONSISTENTLY -- no matter how well we are doing.
# WHERE src/jarvis/council.py
# HOW  build_council_context() assembles {state, analysis, chat_history, directive,
#      roles}. deliberate() runs the three agents in order; each gets the context +
#      the transcript so far (agent-to-agent). Deterministic, system-grounded core
#      that ALWAYS works + is testable; an optional Anthropic LLM layer (claude-opus-4-8)
#      kicks in only when anthropic + ANTHROPIC_API_KEY are present, and falls back
#      cleanly to deterministic text. READ-ONLY: advises only, never trades.
# DEPENDS_ON: src/jarvis/consistency.py  (+ optional: anthropic SDK at runtime)
# USED_BY: jarvis_bridge.py (GET /council), tests
# CHANGE_NOTES(IRAC): I: operator wants the LLMs to see good info + chat history,
#   reason, talk to each other to fix consistency issues, advise from system logic,
#   and always be progressive. R: that spec, 2026-06-26. A: a council that feeds each
#   agent the full grounded context + chat history + prior speakers, forces a
#   progressive next-step, and degrades to deterministic when offline. C: trustworthy,
#   system-grounded, forward-looking advice toward a consistent pass -- testable today,
#   LLM-amplified when a key is present.
# =====================================================================
"""The JARVIS council: OMEGA -> JUSTICE -> JARVIS, grounded + progressive, with an optional LLM."""
from __future__ import annotations
import os
from src.jarvis.consistency import analyze_consistency
from src.jarvis import knowledge as KB
from src.jarvis import policy_registry as PR

# Each agent's lens. They share ONE goal: pass the FTMO challenge CONSISTENTLY.
ROLES = {
    "OMEGA": ("OMEGA — telemetry analyst. Read the system metrics coldly and surface the "
              "inefficiency or drift that matters, always citing the number."),
    "JUSTICE": ("JUSTICE — risk & constraints arbiter. Weigh OMEGA against the FTMO walls and "
                "the consistency goal; name the single binding constraint and rule if we are "
                "inside the lines or gambling."),
    "JARVIS": ("JARVIS — IRAC arbiter. Hear OMEGA and JUSTICE, weigh the whole picture and the "
               "chat history, and rule: Issue / Rule / Application / Conclusion, ending on the "
               "ONE next improvement toward a consistent pass."),
}

# The standing directive that keeps every reply forward-looking and grounded.
PROGRESSIVE_DIRECTIVE = (
    "STANDING DIRECTIVE: Passing the FTMO challenge CONSISTENTLY is the only goal. Reason ONLY "
    "from the system numbers provided (never invent a figure). ALWAYS end with the single "
    "highest-leverage next improvement — never say 'all good, nothing to do'. Even when we are "
    "green and inside every line, there is always a next refinement (steadier daily cadence, "
    "higher-consensus entries, less day-to-day concentration, more data/alphas to lift the edge). "
    "You are READ-ONLY: you advise and coach; you never place or modify a trade."
)

ORDER = ("OMEGA", "JUSTICE", "JARVIS")


def build_council_context(state: dict, chat_history=None, prior_messages=None, market_summary=None) -> dict:
    """Assemble everything the agents see: live PORTFOLIO state + grounded analysis + chat history +
    KNOWLEDGE + the MARKET heatmap + the POLICY roster (so JARVIS organizes policies by consistency)."""
    analysis = analyze_consistency(state)
    last_q = ""
    for m in reversed(list(chat_history or [])):
        if (m.get("role") in ("user", "monty")) and m.get("text"):
            last_q = m["text"]; break
    return {
        "state": state,
        "analysis": analysis,                       # the system-logic the agents cite
        "knowledge": KB.as_context(last_q or None, k=5),  # how the bot works + relevant fixes
        "policies": PR.summary(),                   # the policy roster, ranked by FTMO consistency
        "market": market_summary or "",             # the full-universe heatmap (portfolio trader)
        "chat_history": list(chat_history or [])[-12:],
        "prior_messages": list(prior_messages or [])[-8:],
        "directive": PROGRESSIVE_DIRECTIVE,
        "roles": ROLES,
        "goal": "pass the FTMO challenge consistently across the WHOLE FTMO portfolio (one pot, all symbols; +2.5%/day -> +10%, inside the walls)",
    }


# --------------------------------------------------------------------------- #
# Deterministic, system-grounded statements (always available + testable).    #
# --------------------------------------------------------------------------- #
def _omega(a) -> str:
    return (
        f"Telemetry — equity is {a['progress_to_target_pct']:.0f}% to the +{a['target_pct']:.0f}% pass; "
        f"daily-loss headroom {a['daily_headroom_pct']:.0f}%, max-drawdown headroom {a['maxdd_headroom_pct']:.0f}%; "
        f"net signal {a['net_signal']:+.2f} over the directional alphas (consensus {a['consensus_strength']:.2f}); "
        f"policy confidence {a['confidence']*100:.0f}%, entropy {a['entropy']:.2f}; "
        f"{a['consecutive_losses']} consecutive loss{'es' if a['consecutive_losses'] != 1 else ''}. "
        f"Flagging: {a['top_risk_to_consistency']}.")


def _justice(a) -> str:
    inside = a["binding_headroom_pct"] >= 40.0
    pace = ("on pace" if a["pace_ratio"] >= 1.0 else
            ("banked the day" if a["day_target_hit"] else f"behind pace ({a['day_made_pct']:.2f}% of {a['daily_target_pct']:.1f}%)"))
    return (
        f"Ruling — the binding constraint is {a['binding_constraint']} at {a['binding_headroom_pct']:.0f}% headroom; "
        f"we are {'inside the lines' if inside else 'tightening and must respect it'}. "
        f"We are {pace}, and the largest single day is {a['largest_day_share_pct']:.0f}% of the gains "
        f"({'balanced — that is consistency' if a['largest_day_share_pct'] < 45 else 'too concentrated — that is not consistency'}). "
        f"{'Discipline is holding.' if a['consecutive_losses'] < 3 else 'The losing streak is real — protect the account.'}")


def _jarvis(a) -> str:
    return (
        f"Issue — {a['top_risk_to_consistency']}. "
        f"Rule — a consistent pass is +{a['daily_target_pct']:.1f}%/day banked, inside the "
        f"{a['daily_headroom_pct']:.0f}%/{a['maxdd_headroom_pct']:.0f}% walls, with no single day carrying the account. "
        f"Application — p(pass) reads {a['p_pass_pct']}% at confidence {a['confidence']*100:.0f}% and "
        f"{a['progress_to_target_pct']:.0f}% of target. "
        f"Conclusion — {a['progressive_next_step']}. Posture: {a['posture']}.")


_DET = {"OMEGA": _omega, "JUSTICE": _justice, "JARVIS": _jarvis}


# --------------------------------------------------------------------------- #
# Optional LLM layer (anthropic). Falls back to deterministic on any problem.  #
# --------------------------------------------------------------------------- #
def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def build_agent_prompt(agent: str, ctx: dict, transcript_so_far) -> str:
    """The LLM prompt for one agent: role + goal + grounded analysis + chat history +
    the prior speakers (so the council talks to each other) + the progressive directive."""
    a = ctx["analysis"]
    lines = [
        f"YOU ARE {ctx['roles'][agent]}",
        f"GOAL: {ctx['goal']}.",
        ctx["directive"],
        "",
        "SYSTEM ANALYSIS (cite these numbers, do not invent any):",
        f"  progress_to_target={a['progress_to_target_pct']}%  p_pass={a['p_pass_pct']}%",
        f"  daily_headroom={a['daily_headroom_pct']}%  maxdd_headroom={a['maxdd_headroom_pct']}%  binding={a['binding_constraint']}",
        f"  day_made={a['day_made_pct']}% / target {a['daily_target_pct']}%  largest_day_share={a['largest_day_share_pct']}%",
        f"  net_signal={a['net_signal']}  consensus={a['consensus_strength']}  confidence={a['confidence']}  entropy={a['entropy']}",
        f"  consecutive_losses={a['consecutive_losses']}  top_risk={a['top_risk_to_consistency']}  suggested_next={a['progressive_next_step']}",
    ]
    if ctx.get("market"):
        lines += ["", "MARKET (the portfolio trades ALL of these at once):", ctx["market"]]
    if ctx.get("policies"):
        lines += ["", "POLICY ROSTER (you organize these by FTMO consistency):", ctx["policies"]]
    if ctx.get("knowledge"):
        lines += ["", "SYSTEM KNOWLEDGE (how the bot works + grounded fixes you may cite):", ctx["knowledge"]]
    if ctx["chat_history"]:
        lines += ["", "RECENT CHAT WITH MONTY (oldest first):"]
        for m in ctx["chat_history"]:
            who = "Monty" if m.get("role") in ("user", "monty") else "JARVIS"
            lines.append(f"  {who}: {m.get('text', '')}")
    if transcript_so_far:
        lines += ["", "THE COUNCIL SO FAR (respond to your colleagues):"]
        for m in transcript_so_far:
            lines.append(f"  {m['speaker']}: {m['text']}")
    lines += ["", f"Respond AS {agent} in 2-3 crisp sentences, grounded in the numbers above, "
                  f"and end on the next improvement toward a consistent pass."]
    return "\n".join(lines)


def _llm_raw(system: str, user: str, max_tokens: int = 360) -> str | None:
    """One Anthropic call; returns text or None on any problem (never raises)."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model="claude-opus-4-8", max_tokens=max_tokens,
                                     system=system, messages=[{"role": "user", "content": user}])
        parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
        return (" ".join(parts).strip()) or None
    except Exception:
        return None


def _call_llm(agent: str, ctx: dict, transcript_so_far) -> str | None:
    return _llm_raw(ctx["roles"][agent] + "\n" + ctx["directive"],
                    build_agent_prompt(agent, ctx, transcript_so_far))


# --------------------------------------------------------------------------- #
# The deliberation.                                                            #
# --------------------------------------------------------------------------- #
def deliberate(state: dict, chat_history=None, prior_messages=None, use_llm: str = "auto",
               market_summary=None) -> dict:
    """Run OMEGA -> JUSTICE -> JARVIS over the live PORTFOLIO + chat history + the policy roster.

    use_llm: "auto" (LLM if available, else deterministic), "off", or "on".
    Returns {transcript, ruling, analysis, llm_used, grounded_in, context_seen}.
    """
    ctx = build_council_context(state, chat_history=chat_history, prior_messages=prior_messages,
                                market_summary=market_summary)
    a = ctx["analysis"]
    want_llm = (use_llm == "on") or (use_llm == "auto" and llm_available())

    transcript = []
    llm_used = False
    for agent in ORDER:
        text = None
        if want_llm:
            text = _call_llm(agent, ctx, transcript)
            if text:
                llm_used = True
        if not text:
            text = _DET[agent](a)        # always-available grounded fallback
        transcript.append({"speaker": agent, "role": ROLES[agent].split(" — ")[0], "text": text})

    ruling = {
        "issue": a["top_risk_to_consistency"],
        "rule": (f"consistent pass = +{a['daily_target_pct']:.1f}%/day banked, inside the "
                 f"daily/max-DD walls, no single day dominating"),
        "application": (f"p(pass) {a['p_pass_pct']}%, {a['progress_to_target_pct']:.0f}% to target, "
                        f"binding {a['binding_constraint']} at {a['binding_headroom_pct']:.0f}% headroom"),
        "conclusion": a["progressive_next_step"],     # ALWAYS a forward step
        "posture": a["posture"],
        "p_pass_pct": a["p_pass_pct"],
        "progressive_next_step": a["progressive_next_step"],
        "top_risk_to_consistency": a["top_risk_to_consistency"],
    }
    return {
        "transcript": transcript,
        "ruling": ruling,
        "analysis": a,
        "llm_used": llm_used,
        "grounded_in": ["analysis." + k for k in (
            "p_pass_pct", "binding_headroom_pct", "progress_to_target_pct", "net_signal",
            "confidence", "largest_day_share_pct", "progressive_next_step")],
        "context_seen": {
            "chat_turns": len(ctx["chat_history"]),
            "prior_council_messages": len(ctx["prior_messages"]),
            "directive": ctx["directive"],
        },
    }


def answer(question: str, state: dict | None = None, chat_history=None, use_llm: str = "auto",
           market_summary=None) -> dict:
    """'Ask JARVIS how do I fix X / which policy do I run.' Grounded in the knowledge base, the
    POLICY roster, and the live portfolio. Always works (deterministic); LLM-amplified when a key
    is present. Returns {answer, fixes, used_llm, posture, progressive_next_step} -- read-only.
    """
    ql = (question or "").lower()
    is_policy_q = any(t in ql for t in ("policy", "policies", "which model", "champion", "best model"))
    picks = KB.search(question, k=5)
    a = analyze_consistency(state) if state else None
    if is_policy_q:
        det = ["On the policies, sir — ranked by how consistently each passes:", PR.summary()]
        champ = PR.champion()
        if champ:
            pr = champ.get("walk_forward_pass_rate")
            det.append(f"I would run '{champ['id']}': consistency {champ['consistency_score']}/100"
                       + (f", pass-rate {round(float(pr) * 100)}%" if pr is not None else "")
                       + ". Remember: only policies at the SAME env fingerprint are comparable.")
    else:
        det = ["Here is how to fix that, sir:"]
        for e in picks[:3]:
            det.append(f"- {e['fix']} (see {e['refs']})")
    if a:
        det.append(f"In the meantime: posture {a['posture']}, p(pass) {a['p_pass_pct']}%, binding "
                   f"{a['binding_constraint']} at {a['binding_headroom_pct']:.0f}% headroom. "
                   f"Next toward a consistent pass: {a['progressive_next_step']}")
    text, used_llm = "\n".join(det), False
    if (use_llm == "on") or (use_llm == "auto" and llm_available()):
        ctx = build_council_context(state or {}, market_summary=market_summary,
                                    chat_history=list(chat_history or []) + [{"role": "user", "text": question}])
        user = (f"Monty asks: \"{question}\"\n\nAnswer as JARVIS in 2-4 sentences: answer exactly (how to fix "
                f"it, OR which policy to run + why), cite the file(s) to edit, and end on the next step toward "
                f"passing CONSISTENTLY across the portfolio. Ground every claim in the context below; if a "
                f"figure is missing, say so.\n\n" + ctx["knowledge"] + "\n\n" + ctx["policies"]
                + (f"\n\nMARKET: {ctx['market']}" if ctx.get("market") else "")
                + (f"\n\nLIVE ANALYSIS: {a}" if a else ""))
        out = _llm_raw(ROLES["JARVIS"] + "\n" + PROGRESSIVE_DIRECTIVE, user, max_tokens=440)
        if out:
            text, used_llm = out, True
    return {
        "answer": text, "fixes": picks, "used_llm": used_llm,
        "posture": a["posture"] if a else None,
        "progressive_next_step": a["progressive_next_step"] if a else None,
    }
