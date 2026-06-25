# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  FTMO challenge rule checks (preserved from Quantra): 2.5%/day target,
#      daily loss limit, total loss limit, 4% trailing wall, two-phase flag.
# WHERE src/risk/ftmo_rules.py
# HOW  Reads the (editable) FTMOConfig and evaluates AccountState. All limits
#      come from config at call time -> live risk retunes work with no retrain.
# DEPENDS_ON: config/ftmo_config.py, src/account/account_state.py
# USED_BY: src/risk/breach_detector.py, src/env/trading_env.py (Phase 1),
#          tests/test_ftmo_free_mode.py
# CHANGE_NOTES(IRAC): I: must enforce FTMO without baking limits into the model.
#   R: Quantra 2.5%/4%/two-phase. A: config-driven boolean checks. C: faithful,
#   retunable FTMO enforcement -> training stays aligned to passing.
# =====================================================================
"""FTMO rule checks (target hit, daily/total/trailing breach, two-phase)."""
from __future__ import annotations
from config.ftmo_config import load_ftmo_config
from src.account.account_state import AccountState


class FTMORules:
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_ftmo_config()

    def daily_target_hit(self, acc: AccountState) -> bool:
        # Daily target = +2.5% of the INITIAL balance, measured as the DAY's gain on
        # EQUITY (open profit included) so the engine can auto-bank the moment you're up
        # 2.5% on the day. Fixed $/day -> ~4 such days ladder cleanly to the +10% pass.
        base = acc.starting_balance or acc.day_start_balance
        day0 = acc.day_start_balance if acc.day_start_balance is not None else base
        return base > 0 and (acc.equity - day0) >= base * self.cfg.daily_target_pct / 100.0

    def daily_drawdown_breached(self, acc: AccountState) -> bool:
        bal0 = acc.day_start_balance or acc.starting_balance
        daily_loss = bal0 - acc.equity
        return daily_loss >= bal0 * self.cfg.daily_drawdown_pct / 100.0

    def total_drawdown_breached(self, acc: AccountState) -> bool:
        loss = acc.starting_balance - acc.equity
        return loss >= acc.starting_balance * self.cfg.max_total_drawdown_pct / 100.0

    def trailing_breached(self, acc: AccountState) -> bool:
        if not self.cfg.trailing_enabled:
            return False
        peak = acc.episode_peak_equity or acc.starting_balance
        return (peak - acc.equity) >= peak * self.cfg.trailing_drawdown_pct / 100.0

    def should_auto_flat(self, acc: AccountState) -> bool:
        """Two-phase: once +2.5% is hit, flatten all and start a fresh trail."""
        return self.cfg.two_phase_enabled and self.daily_target_hit(acc)

    def reasons(self, acc: AccountState) -> list[str]:
        out = []
        if self.daily_drawdown_breached(acc):
            out.append("daily_drawdown")
        if self.total_drawdown_breached(acc):
            out.append("total_drawdown")
        if self.trailing_breached(acc):
            out.append("trailing_drawdown")
        return out
