# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Collect read-only introspection records (and any diagnostic snapshots)
#      so Policy Doctor / Jarvis can read 'what the policy was thinking'.
# WHERE src/interpret/telemetry.py | HOW append-only list + jsonl dump.
# DEPENDS_ON dataclasses,json | USED_BY policy_doctor, jarvis (Phase 2).
"""TelemetryLog: append-only store of introspection records (read-only diagnostics)."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import json


class TelemetryLog:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, record) -> None:
        self._rows.append(asdict(record) if is_dataclass(record) else dict(record))

    def rows(self) -> list[dict]:
        return list(self._rows)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self._rows:
                f.write(json.dumps(r) + "\n")

    def clear(self) -> None:
        self._rows.clear()
