# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  JARVIS: final judge -- synthesizes OMEGA + JUSTICE into one verdict.
# WHERE src/jarvis/jarvis_judge.py | HOW reads both agents' latest messages.
# DEPENDS_ON src/jarvis/{agent_bus,omega_agent,justice_agent}.py
# USED_BY src/jarvis/app.py.
"""JARVIS judge: synthesize OMEGA + JUSTICE into a final verdict (mock)."""
from __future__ import annotations
from src.jarvis.agent_bus import AgentBus, Message
from src.jarvis.omega_agent import OmegaAgent
from src.jarvis.justice_agent import JusticeAgent

class JarvisJudge:
    name = "JARVIS"
    def __init__(self) -> None:
        self.omega = OmegaAgent()
        self.justice = JusticeAgent()

    def cycle(self, obs, acc, bus: AgentBus, cfg=None) -> Message:
        om = self.omega.review(obs, bus)
        ju = self.justice.review(acc, bus, cfg)
        problems = [m for m in (om, ju) if m.kind == "warning"]
        if problems:
            return bus.post(self.name, "HOLD: " + " | ".join(m.text for m in problems),
                            kind="verdict")
        return bus.post(self.name, "CLEAR: pipeline consistent and within risk.",
                        kind="verdict")
