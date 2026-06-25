# =====================================================================
# WHEN 2026-06-21 (Phase 0 stub; Phase 1 real) | WHO Claude for Monty
# WHY  The RL environment. Observation = the locked 367 contract assembled from
#      the CACHE (no TA-Lib/MT5/pandas in step). Actions {HOLD,BUY,SELL,CLOSE}.
#      REWARD = change in account equity only (the real objective) -- NEVER any
#      alpha/accuracy term. FTMO state is tracked day by day.
# WHERE src/env/trading_env.py
# HOW  __init__ precomputes alpha_matrix, net-signal, leak-free signal accuracy,
#      and time features once. step() reads cached float32, marks the position
#      to market, updates the FTMO account, checks breach, and returns obs.
# DEPENDS_ON: cache (indicators/close/time), AlphaRegistry, observation builder,
#             account_state/trade_history/win_loss_features, risk/breach_detector
# USED_BY: src/training/* (Phase 1), src/barbershop/* + introspector (eval), tests
# CHANGE_NOTES(IRAC): I: env is the #1 spot for leakage/day-reset/reward bugs.
#   R: operator guardrails 2026-06-21 (reward objective-only, no leakage, clean
#   eval separation). A: precompute-then-cached-step; reward = equity delta only;
#   per-day FTMO reset; breach terminates. C: honest training signal aligned to
#   passing FTMO, fast enough to train on CPU.
# =====================================================================
"""TradingEnv: cached-observation FTMO env. Reward = equity change only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_active_config
from config import variables as V
from src.indicators.base import ALL_INDICATOR_COLUMNS
from src.strategies.context import MarketContext
from src.signals.signal_summary import summarize, net_balance
from src.signals.signal_memory import last5_from_series
from src.signals.signal_accuracy import accuracy_features
from src.account.account_state import AccountState
from src.account.trade_history import TradeHistory
from src.account import win_loss_features as WL
from src.risk import breach_detector as BD
from src.observation import builder as OB


class TradingEnv:
    """Single-position FTMO env over one symbol. Gym-style API (import-safe)."""

    observation_shape = C.OBS_SHAPE
    n_actions = C.N_ACTIONS

    def __init__(self, indicators, close, time_ns, alpha_registry, *, cfg=None,
                 position_size: float = 100000.0, breach_penalty: float = 1.0,
                 reward_scale: float = 1.0, open_gate: bool = False,
                 cost_frac: float | None = None, pass_bonus: float | None = None,
                 window: int | None = None, warmup: int = 200,
                 random_window: bool = False, seed: int | None = None):
        self.ind = np.asarray(indicators, dtype=np.float32)
        self.close = np.asarray(close, dtype=np.float64).ravel()
        self.time_ns = np.asarray(time_ns).astype("int64").ravel()
        self.T = self.close.shape[0]
        self.cfg = cfg or load_active_config()
        self.position_size = float(position_size)
        self.breach_penalty = float(breach_penalty)
        self.reward_scale = float(reward_scale)   # F2: condition learning signal w/o oversizing
        self.open_gate = bool(open_gate)          # 5m CCI open-gate (off by default)
        self.cost_frac = float(V.TRANSACTION_COST_FRAC_PER_SIDE if cost_frac is None else cost_frac)
        self.pass_bonus = float(self.breach_penalty if pass_bonus is None else pass_bonus)
        self.profit_target_frac = float(getattr(self.cfg, 'profit_target_total_pct', 10.0)) / 100.0
        self.warmup = int(warmup)
        self.window = window
        self.random_window = bool(random_window)
        self.rng = np.random.default_rng(seed)
        self._dates = pd.to_datetime(self.time_ns).normalize().values  # day boundaries
        self._precompute(alpha_registry)

    # ---- precompute (once): alphas, net signal, leak-free accuracy, time ----
    def _precompute(self, registry):
        T = self.T
        self.alpha_matrix = np.zeros((T, C.MAX_STRATEGIES), dtype=np.float32)
        for i in range(T):
            ctx = MarketContext(close=float(self.close[i]),
                                indicators=dict(zip(ALL_INDICATOR_COLUMNS,
                                                    self.ind[i].tolist())),
                                bar_index=i)
            self.alpha_matrix[i] = registry.collect_alphas(ctx)
        self.occupancy = registry.occupancy_mask()
        self.net_signal = np.array([net_balance(self.alpha_matrix[i]) for i in range(T)],
                                   dtype=np.float32)
        self.sig_acc = accuracy_features(self.net_signal, self.close)        # (T,2) leak-free
        self.time_feats = np.stack([OB.time_features(pd.Timestamp(self.time_ns[i]))
                                    for i in range(T)]).astype(np.float32)    # (T,6)
        # 5m CCI open-gate mask: True where EITHER 5m CCI sits in [-50, 50] (flat/undecided
        # short-term market) -> new directional opens are forbidden when self.open_gate is on.
        try:
            j30 = ALL_INDICATOR_COLUMNS.index("5m__cci30_raw")
            j100 = ALL_INDICATOR_COLUMNS.index("5m__cci100_raw")
            self.open_gate_blocked = (np.abs(self.ind[:, j30]) <= 50.0) | (np.abs(self.ind[:, j100]) <= 50.0)
        except ValueError:
            self.open_gate_blocked = np.zeros(self.T, dtype=bool)
        # v1.2.0: per-alpha signal streak (consecutive bars, same non-zero signal) -- leak-free
        am = self.alpha_matrix
        nz = am != 0
        cont = np.zeros_like(am, dtype=bool); cont[1:] = (am[1:] == am[:-1]) & nz[1:]
        self.streak_matrix = np.zeros_like(am, dtype=np.float32)
        s = np.zeros(am.shape[1], dtype=np.float32)
        for i in range(am.shape[0]):
            s = np.where(cont[i], s + 1.0, np.where(nz[i], 1.0, 0.0))
            self.streak_matrix[i] = s

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.random_window and self.window:
            hi = max(self.warmup + 1, self.T - self.window - 1)
            self.start = int(self.rng.integers(self.warmup, hi))
            self.end = min(self.T - 1, self.start + self.window)
        else:
            self.start, self.end = self.warmup, self.T - 1
        self.ptr = self.start
        self.acc = AccountState(starting_balance=self.cfg.starting_balance)
        self.th = TradeHistory()
        self.position = 0
        self.entry_price = float(self.close[self.ptr])
        self._cur_date = self._dates[self.ptr]
        return self._obs(), {"ptr": self.ptr}

    def step(self, action: int):
        a = int(action)
        t = self.ptr
        equity_before = self.acc.equity

        # 1) act at close[t]: realize on any position change, set new position
        target = {C.ACTION_HOLD: self.position, C.ACTION_BUY: 1,
                  C.ACTION_SELL: -1, C.ACTION_CLOSE: 0}[a]
        # 5m CCI open-gate: forbid establishing a NEW direction when the 5m market is
        # neutral (EITHER 5m CCI in [-50,50]). A flip just closes; holds/closes pass through.
        if self.open_gate and self.open_gate_blocked[t] and target != 0 and target != self.position:
            target = 0
        if target != self.position:
            if self.position != 0:
                realized = self.position * (self.close[t] - self.entry_price) * self.position_size
                realized -= self.cost_frac * self.close[t] * self.position_size      # exit cost
                # record_close is the SINGLE source of truth: it updates balance +
                # daily/episode realized PnL + equity + tallies. Do NOT also add them
                # here -- that double-counted every closed trade (banked 2x the PnL).
                self.th.record_close(self.acc, realized, bar_index=t)
            self.position = target
            if target != 0:
                self.entry_price = float(self.close[t])
                ecost = self.cost_frac * self.close[t] * self.position_size           # entry cost
                self.acc.balance -= ecost
                self.acc.daily_realized_pnl -= ecost
                self.acc.episode_realized_pnl -= ecost

        # 2) advance one bar, mark to market at close[t+1]
        self.ptr = min(t + 1, self.T - 1)
        unrealized = self.position * (self.close[self.ptr] - self.entry_price) * self.position_size
        self.acc.equity = self.acc.balance + unrealized
        self.acc.mark_equity(self.acc.equity)

        # 3) REWARD = equity change as a fraction of starting balance (objective only)
        reward = float((self.acc.equity - equity_before) / self.cfg.starting_balance) * self.reward_scale

        # 4) day boundary -> reset daily FTMO state (after reward)
        if self._dates[self.ptr] != self._cur_date:
            self.acc.reset_day()
            self._cur_date = self._dates[self.ptr]

        # 5) breach check (FTMO/free) -> terminate with penalty (still objective)
        rep = BD.detect(self.acc, self.cfg)
        terminated = bool(rep.breached)
        if terminated:
            self.acc.episode_breached = True
            reward -= self.breach_penalty
        elif self.acc.equity >= self.cfg.starting_balance * (1.0 + self.profit_target_frac):
            terminated = True                        # +10% FTMO Challenge target reached -> PASS
            self.acc.episode_passed = True
            reward += self.pass_bonus
        # daily target -> two-phase auto-flat (FTMO)
        if rep.should_auto_flat and self.position != 0:
            realized = self.position * (self.close[self.ptr] - self.entry_price) * self.position_size
            self.th.record_close(self.acc, realized, bar_index=self.ptr)  # single source of truth (no manual +=)
            self.position = 0
            self.acc.equity = self.acc.balance

        truncated = bool(self.ptr >= self.end)
        info = {"ptr": self.ptr, "equity": self.acc.equity, "position": self.position,
                "breach_reasons": rep.reasons, "daily_target_hit": rep.daily_target_hit,
                "action": a, "alpha_streaks": self.streak_matrix[self.ptr].astype(int)}
        return self._obs(), reward, terminated, truncated, info

    # ---- observation assembly from cache (no recompute) ----
    def _obs(self):
        i = self.ptr
        return OB.build_from_blocks({
            "indicators": self.ind[i],
            "alpha_values": self.alpha_matrix[i],
            "alpha_mask": self.occupancy,
            "alpha_summary": summarize(self.alpha_matrix[i], self.occupancy),
            "signal_memory": last5_from_series(self.net_signal, i),
            "signal_accuracy": self.sig_acc[i],
            "account_daily": WL.daily_features(self.acc, self.cfg),
            "account_episode": WL.episode_features(self.acc, self.cfg),
            "time": self.time_feats[i],
            "portfolio": self._portfolio_block(),
            "alpha_streak": np.minimum(self.streak_matrix[i], C.ALPHA_STREAK_CAP) / float(C.ALPHA_STREAK_CAP),
        })

    def _portfolio_block(self):
        self.acc.open_positions = 1 if self.position != 0 else 0
        self.acc.net_exposure = float(self.position)
        self.acc.gross_exposure = float(abs(self.position))
        self.acc.unrealized_pnl = self.acc.equity - self.acc.balance
        self.acc.largest_position_dir = int(np.sign(self.position))
        return WL.portfolio_features(self.acc)
