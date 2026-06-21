# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  OMEGA: watches pipeline I/O, finds bugs, checks consistency.
# WHERE src/jarvis/omega_agent.py | HOW inspects a built observation via the
#      Feature Doctor and posts findings to the bus. Field of view = data/
#      alphas/combination/portfolio modules.
# DEPENDS_ON src/barbershop/feature_doctor.py, src/jarvis/agent_bus.py
# USED_BY src/jarvis/jarvis_judge.py, src/jarvis/app.py.
"""OMEGA agent: pipeline I/O + consistency watchdog (mock)."""
from __future__ import annotations
from src.jarvis.agent_bus import AgentBus, Message
from src.barbershop import feature_doctor as FD

FIELD_OF_VIEW = ["01", "02", "03", "04", "05"]

class OmegaAgent:
    name = "OMEGA"
    def review(self, obs, bus: AgentBus) -> Message:
        rep = FD.inspect(obs)
        if not rep["shape_ok"]:
            return bus.post(self.name, f"Observation shape {rep['shape']} != "
                            f"{rep['expected_shape']}.", kind="warning")
        if rep["nonfinite_count"] > 0:
            return bus.post(self.name, f"{rep['nonfinite_count']} non-finite features.",
                            kind="warning")
        return bus.post(self.name, "Pipeline I/O consistent: obs shape + finiteness OK.",
                        kind="info")
