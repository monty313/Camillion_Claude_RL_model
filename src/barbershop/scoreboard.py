# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Barbershop #1: daily/episode PnL, target progress, drawdown, pass/fail.
# WHERE src/barbershop/scoreboard.py | HOW reads AccountState + breach_detector.
# DEPENDS_ON src/account/account_state.py, src/risk/breach_detector.py
# USED_BY src/jarvis/app.py (Phase 2), tests/test_barbershop_smoke.py.
"""Scoreboard: snapshot of daily + episode performance and pass/fail state."""
from __future__ import annotations
from src.account.account_state import AccountState
from src.risk import breach_detector as BD


def scoreboard(acc: AccountState, cfg=None) -> dict:
    rep = BD.detect(acc, cfg)
    bal0d = acc.day_start_balance or acc.starting_balance
    bal0e = acc.episode_start_balance or acc.starting_balance
    return {
        "mode": rep.mode,
        "daily": {
            "realized_pnl": acc.daily_realized_pnl,
            "pnl_pct": (acc.daily_realized_pnl / bal0d) * 100 if bal0d else 0.0,
            "trades": acc.daily_trades,
            "win_rate": acc.daily_win_rate,
            "target_hit": rep.daily_target_hit,
        },
        "episode": {
            "realized_pnl": acc.episode_realized_pnl,
            "pnl_pct": (acc.episode_realized_pnl / bal0e) * 100 if bal0e else 0.0,
            "trades": acc.episode_trades,
            "win_rate": acc.episode_win_rate,
            "passed": acc.episode_passed,
            "breached": acc.episode_breached or rep.breached,
        },
        "breach_reasons": rep.reasons,
        "should_auto_flat": rep.should_auto_flat,
    }
