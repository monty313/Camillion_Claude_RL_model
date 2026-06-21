# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Barbershop #4: drawdown events, near-breaches, loss streaks, budget left.
# WHERE src/barbershop/risk_doctor.py | HOW reads breach_detector + AccountState.
# DEPENDS_ON src/risk/breach_detector.py, config/ftmo_config.py
# USED_BY src/jarvis/justice_agent.py (Phase 2), tests/test_barbershop_smoke.py.
"""Risk Doctor: drawdown/budget diagnostics for the active mode."""
from __future__ import annotations
from config.ftmo_config import load_active_config
from src.account.account_state import AccountState
from src.risk import breach_detector as BD


def diagnose(acc: AccountState, cfg=None) -> dict:
    cfg = cfg or load_active_config()
    rep = BD.detect(acc, cfg)
    peak = acc.episode_peak_equity or acc.starting_balance
    dd_from_peak = (peak - acc.equity) / peak * 100 if peak else 0.0
    return {
        "breached": rep.breached,
        "reasons": rep.reasons,
        "drawdown_from_peak_pct": dd_from_peak,
        "daily_consecutive_losses": acc.daily_consecutive_losses,
        "episode_consecutive_losses": acc.episode_consecutive_losses,
        "trailing_enabled": getattr(cfg, "trailing_enabled", False),
    }
