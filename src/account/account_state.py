# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Hold the live account picture (balance/equity, daily & episode tallies,
#      aggregate open-position state) that the account/portfolio features read.
# WHERE src/account/account_state.py
# HOW  A mutable dataclass with reset_day()/reset_episode() and win-rate props.
#      These are CONTEXT only -- the bot is never rewarded directly from them.
# DEPENDS_ON: (stdlib only)
# USED_BY: src/account/win_loss_features.py, src/account/trade_history.py,
#          src/risk/*, src/observation/builder.py
# CHANGE_NOTES(IRAC): I: features need a tidy state source. R: spec daily +
#   episode account block + portfolio block. A: one dataclass w/ resets.
#   C: clean state -> correct, leak-free account features for FTMO awareness.
# =====================================================================
"""AccountState: balance/equity + daily & episode tallies + open-position state."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AccountState:
    starting_balance: float = 100_000.0
    balance: float | None = None
    equity: float | None = None

    # --- daily ---
    day_start_balance: float | None = None
    day_peak_equity: float | None = None
    daily_realized_pnl: float = 0.0
    daily_wins: int = 0
    daily_losses: int = 0
    daily_trades: int = 0
    daily_consecutive_losses: int = 0

    # --- episode (full challenge) ---
    episode_start_balance: float | None = None
    episode_peak_equity: float | None = None
    episode_realized_pnl: float = 0.0
    episode_wins: int = 0
    episode_losses: int = 0
    episode_trades: int = 0
    episode_consecutive_losses: int = 0
    episode_breached: bool = False
    episode_passed: bool = False

    # --- aggregate open-position state (portfolio block) ---
    open_positions: int = 0
    net_exposure: float = 0.0       # signed, normalised to [-1, +1] upstream
    gross_exposure: float = 0.0     # fraction of max exposure [0, 1]
    unrealized_pnl: float = 0.0
    avg_position_age_bars: float = 0.0
    largest_position_dir: int = 0   # -1 short / 0 flat / +1 long

    def __post_init__(self) -> None:
        if self.balance is None:
            self.balance = self.starting_balance
        if self.equity is None:
            self.equity = self.balance
        if self.day_start_balance is None:
            self.day_start_balance = self.balance
        if self.day_peak_equity is None:
            self.day_peak_equity = self.equity
        if self.episode_start_balance is None:
            self.episode_start_balance = self.balance
        if self.episode_peak_equity is None:
            self.episode_peak_equity = self.equity

    # --- helpers ---
    @property
    def daily_win_rate(self) -> float:
        return self.daily_wins / self.daily_trades if self.daily_trades else 0.0

    @property
    def episode_win_rate(self) -> float:
        return self.episode_wins / self.episode_trades if self.episode_trades else 0.0

    def mark_equity(self, equity: float) -> None:
        """Update equity and rolling peaks (for trailing-drawdown math)."""
        self.equity = float(equity)
        self.day_peak_equity = max(self.day_peak_equity, self.equity)
        self.episode_peak_equity = max(self.episode_peak_equity, self.equity)

    def reset_day(self) -> None:
        self.day_start_balance = self.balance
        self.day_peak_equity = self.equity
        self.daily_realized_pnl = 0.0
        self.daily_wins = self.daily_losses = self.daily_trades = 0
        self.daily_consecutive_losses = 0

    def reset_episode(self) -> None:
        self.balance = self.starting_balance
        self.equity = self.starting_balance
        self.episode_start_balance = self.starting_balance
        self.episode_peak_equity = self.starting_balance
        self.episode_realized_pnl = 0.0
        self.episode_wins = self.episode_losses = self.episode_trades = 0
        self.episode_consecutive_losses = 0
        self.episode_breached = self.episode_passed = False
        self.open_positions = 0
        self.net_exposure = self.gross_exposure = self.unrealized_pnl = 0.0
        self.avg_position_age_bars = 0.0
        self.largest_position_dir = 0
        self.reset_day()
