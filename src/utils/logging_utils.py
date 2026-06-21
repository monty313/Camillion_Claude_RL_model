# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty | WHY tiny shared logger.
# WHERE src/utils/logging_utils.py | HOW thin wrapper over stdlib logging.
# DEPENDS_ON logging | USED_BY most modules (optional).
"""Minimal shared logger."""
from __future__ import annotations
import logging

def get_logger(name: str = "camillion", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(level)
    return log
