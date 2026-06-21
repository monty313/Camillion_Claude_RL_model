# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Record closed trades and keep win/loss tallies + loss streaks in sync
#      on the AccountState (daily and episode).
# WHERE src/account/trade_history.py
# HOW  record_close(acc, pnl) updates balances, win/loss counts, and the
#      consecutive-loss streaks. Day Replay / Trade Autopsy (Phase 2) read it.
# DEPENDS_ON: src/account/account_state.py
# USED_BY: src/barbershop/{day_replay,trade_autopsy}.py (Phase 2), env (Phase 1)
# CHANGE_NOTES(IRAC): I: need a tidy closed-trade ledger. R: spec win/loss
#   history features + Barbershop autopsy. A: append-only list + tally updater.
#   C: accurate streak/win-rate inputs for the account block.
# =====================================================================
"""TradeHistory: append-only closed-trade ledger + win/loss tally updates."""
from __future__ import annotations
from dataclasses import dataclass, field
from src.account.account_state import AccountState


@dataclass
class ClosedTrade:
    pnl: float
    is_win: bool
    bar_index: int = -1
    symbol: str = ""
    direction: int = 0  # +1 long / -1 short


@dataclass
class TradeHistory:
    trades: list[ClosedTrade] = field(default_factory=list)

    def record_close(self, acc: AccountState, pnl: float, *, bar_index: int = -1,
                     symbol: str = "", direction: int = 0) -> ClosedTrade:
        """Append a closed trade and update AccountState tallies/streaks."""
        is_win = pnl > 0.0
        t = ClosedTrade(pnl=float(pnl), is_win=is_win, bar_index=bar_index,
                        symbol=symbol, direction=direction)
        self.trades.append(t)
        # balances
        acc.balance += pnl
        acc.daily_realized_pnl += pnl
        acc.episode_realized_pnl += pnl
        acc.mark_equity(acc.balance)
        # tallies
        acc.daily_trades += 1
        acc.episode_trades += 1
        if is_win:
            acc.daily_wins += 1
            acc.episode_wins += 1
            acc.daily_consecutive_losses = 0
            acc.episode_consecutive_losses = 0
        else:
            acc.daily_losses += 1
            acc.episode_losses += 1
            acc.daily_consecutive_losses += 1
            acc.episode_consecutive_losses += 1
        return t

    def reset(self) -> None:
        self.trades.clear()
