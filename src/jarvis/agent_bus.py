# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Local, mock message bus for the IRAC agent loop -- NO paid API keys.
# WHERE src/jarvis/agent_bus.py | HOW in-memory list of messages + helpers.
# DEPENDS_ON dataclasses,time | USED_BY omega/justice/jarvis agents, app.py.
"""In-memory agent message bus (mock, no external API)."""
from __future__ import annotations
from dataclasses import dataclass, field
import time

@dataclass
class Message:
    sender: str
    text: str
    kind: str = "info"          # info | warning | verdict
    ts: float = field(default_factory=time.time)

class AgentBus:
    def __init__(self) -> None:
        self._msgs: list[Message] = []

    def post(self, sender: str, text: str, kind: str = "info") -> Message:
        m = Message(sender=sender, text=text, kind=kind)
        self._msgs.append(m)
        return m

    def history(self) -> list[Message]:
        return list(self._msgs)

    def clear(self) -> None:
        self._msgs.clear()
