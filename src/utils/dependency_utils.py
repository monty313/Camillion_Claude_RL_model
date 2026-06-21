# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Parse the WHEN/WHO/WHY/.../DEPENDS_ON/USED_BY headers out of source files
#      so docs/DEPENDENCY_MAP.md (and the Jarvis map) can be (re)built.
# WHERE src/utils/dependency_utils.py | HOW regex over the comment header.
# DEPENDS_ON re,pathlib | USED_BY tools/build_dependency_map (future), Jarvis.
"""Read the structured file headers (DEPENDS_ON / USED_BY / WHY ...)."""
from __future__ import annotations
import re
from pathlib import Path

FIELDS = ("WHEN", "WHO", "WHY", "WHERE", "HOW", "DEPENDS_ON", "USED_BY")

def parse_header(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")[:4000]
    out: dict[str, str] = {}
    for field in FIELDS:
        m = re.search(rf"{field}:\s*(.+)", text)
        if m:
            out[field] = m.group(1).strip()
    return out
