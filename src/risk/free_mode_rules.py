# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  FREE-mode rule checks using the operator's own target/risk numbers.
# WHERE src/risk/free_mode_rules.py
# HOW  Mirrors FTMORules' method names so breach_detector can use either one.
#      Limits come from FreeModeConfig (variables.py) at call time.
# DEPENDS_ON: config/ftmo_config.py (FreeModeConfig), src/account/account_state.py
# USED_BY: src/risk/breach_detector.py, tests/test_ftmo_free_mode.py
# CHANGE_NOTES(IRAC): I: FREE mode needs the same interface as FTMO. R: spec
#   "Free mode: user sets any target and risk". A: same method names, FREE cfg.
#   C: one breach detector works for both modes; easy experimentation.
# =====================================================================
"""FREE-mode rule checks (same interface as FTMORules, operator-defined limits)."""
from __future__ import annotations
from config.ftmo_config import load_free_config
from src.account.account_state import AccountState


class FreeModeRules:
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_free_config()

    def daily_target_hit(self, acc: AccountState) -> bool:
        bal0 = acc.day_start_balance or acc.starting_balance
        return bal0 > 0 and (acc.daily_realized_pnl / bal0) >= self.cfg.daily_target_pct / 100.0

    def daily_drawdown_breached(self, acc: AccountState) -> bool:
        bal0 = acc.day_start_balance or acc.starting_balance
        return (bal0 - acc.equity) >= bal0 * self.cfg.max_daily_drawdown_pct / 100.0

    def total_drawdown_breached(self, acc: AccountState) -> bool:
        loss = acc.starting_balance - acc.equity
        return loss >= acc.starting_balance * self.cfg.max_total_drawdown_pct / 100.0

    def trailing_breached(self, acc: AccountState) -> bool:
        if not self.cfg.trailing_enabled:
            return False
        peak = acc.episode_peak_equity or acc.starting_balance
        return (peak - acc.equity) >= peak * self.cfg.trailing_drawdown_pct / 100.0

    def should_auto_flat(self, acc: AccountState) -> bool:
        return False  # FREE mode has no two-phase rule by default

    def reasons(self, acc: AccountState) -> list[str]:
        out = []
        if self.daily_drawdown_breached(acc):
            out.append("daily_drawdown")
        if self.total_drawdown_breached(acc):
            out.append("total_drawdown")
        if self.trailing_breached(acc):
            out.append("trailing_drawdown")
        return out
