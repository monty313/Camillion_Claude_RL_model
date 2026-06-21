# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  One entry point that picks the right rule set (FTMO or FREE) and
#      returns a clear breach report + whether the daily target is hit.
# WHERE src/risk/breach_detector.py
# HOW  Dispatch on the active config's mode; both rule classes share an API.
# DEPENDS_ON: config/ftmo_config.py, src/risk/{ftmo_rules,free_mode_rules}.py
# USED_BY: src/env/trading_env.py (Phase 1), src/barbershop/risk_doctor.py,
#          src/jarvis/justice_agent.py (Phase 2), tests/test_ftmo_free_mode.py
# CHANGE_NOTES(IRAC): I: env/UI need a single breach check for both modes. R:
#   spec breach detection + pass/fail tracking. A: dispatch + BreachReport.
#   C: consistent risk verdicts across env, Barbershop and the Justice agent.
# =====================================================================
"""Unified breach detection for FTMO and FREE modes."""
from __future__ import annotations
from dataclasses import dataclass, field
from config.ftmo_config import load_active_config
from src.risk.ftmo_rules import FTMORules
from src.risk.free_mode_rules import FreeModeRules
from src.account.account_state import AccountState


@dataclass
class BreachReport:
    breached: bool
    reasons: list[str] = field(default_factory=list)
    daily_target_hit: bool = False
    should_auto_flat: bool = False
    mode: str = ""

    def __bool__(self) -> bool:
        return self.breached


def make_rules(cfg=None):
    """Return the rule object matching the active (or given) config's mode."""
    cfg = cfg or load_active_config()
    return FTMORules(cfg) if getattr(cfg, "mode", "FTMO") == "FTMO" else FreeModeRules(cfg)


def detect(acc: AccountState, cfg=None) -> BreachReport:
    """Evaluate breaches + daily-target state for the active mode."""
    cfg = cfg or load_active_config()
    rules = make_rules(cfg)
    reasons = rules.reasons(acc)
    return BreachReport(
        breached=bool(reasons),
        reasons=reasons,
        daily_target_hit=rules.daily_target_hit(acc),
        should_auto_flat=rules.should_auto_flat(acc),
        mode=getattr(cfg, "mode", "FTMO"),
    )
