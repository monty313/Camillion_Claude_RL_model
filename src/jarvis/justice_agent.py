# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  JUSTICE: checks risk, FTMO constraints, pass/fail alignment.
# WHERE src/jarvis/justice_agent.py | HOW runs the breach detector on the
#      AccountState and posts a verdict. Field of view = risk/objective/
#      constraints/trading modules.
# DEPENDS_ON src/risk/breach_detector.py, src/jarvis/agent_bus.py
# USED_BY src/jarvis/jarvis_judge.py, src/jarvis/app.py.
"""JUSTICE agent: risk + FTMO constraint checker (mock)."""
from __future__ import annotations
from src.jarvis.agent_bus import AgentBus, Message
from src.risk import breach_detector as BD

FIELD_OF_VIEW = ["R", "O", "C", "06"]

class JusticeAgent:
    name = "JUSTICE"
    def review(self, acc, bus: AgentBus, cfg=None) -> Message:
        rep = BD.detect(acc, cfg)
        if rep.breached:
            return bus.post(self.name, f"BREACH ({rep.mode}): {', '.join(rep.reasons)}.",
                            kind="warning")
        if rep.daily_target_hit:
            return bus.post(self.name, f"Daily target hit ({rep.mode}); "
                            f"auto-flat={rep.should_auto_flat}.", kind="info")
        return bus.post(self.name, f"Within risk ({rep.mode}); no breach.", kind="info")
