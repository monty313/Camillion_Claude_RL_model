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
"""TradingEnv: cached-observation FTMO env. Reward = equity change + deliberate FTMO shaping
(breach penalty, +10% pass bonus, and the operator's NY-session index bonus)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_active_config
from config import variables as V
from config import asset_specs as A
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
                 open_gate_threshold: float | None = None,
                 cost_frac: float | None = None, pass_bonus: float | None = None,
                 symbol: str | None = None, value_per_point: float | None = None,
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
        # |cci| must exceed this for BOTH 5m CCIs to allow a new open (config-driven, tunable)
        self.open_gate_threshold = float(
            V.OPEN_GATE_CCI_THRESHOLD if open_gate_threshold is None else open_gate_threshold)
        self.cost_frac = float(V.TRANSACTION_COST_FRAC_PER_SIDE if cost_frac is None else cost_frac)
        self.pass_bonus = float(self.breach_penalty if pass_bonus is None else pass_bonus)
        self.profit_target_frac = float(getattr(self.cfg, 'profit_target_total_pct', 10.0)) / 100.0
        # v1.3.0 sizing block: account $ per 1.0 PRICE move per 1 lot. From the asset spec if
        # known, else treat the active position_size as "1 lot" (active_lots = 1) so the block
        # stays sane for any data. position_size = value_per_point * active_lots.
        self.symbol = symbol
        if value_per_point is not None:
            self.value_per_point = float(value_per_point)
        elif symbol and symbol in A.SPECS:
            self.value_per_point = float(A.SPECS[symbol].contract_size)
        else:
            self.value_per_point = float(self.position_size) or 1.0
        self._typical_atr = A.typical_atr(symbol)   # per-asset "how it normally moves" baseline (or None)
        self._typical_range = (A.SPECS[symbol].typical_daily_range
                               if (symbol and symbol in A.SPECS) else None)  # symbol's long-run daily range
        self.warmup = int(warmup)
        self.window = window
        self.random_window = bool(random_window)
        self.rng = np.random.default_rng(seed)
        _ts = pd.to_datetime(self.time_ns)
        self._dates = _ts.normalize().values                          # day boundaries
        self._minute_of_day = (_ts.hour * 60 + _ts.minute).to_numpy().astype(np.int32)  # UTC, for NY session
        self._is_index = (A.asset_class(symbol) == "index")           # ORB + NY bonus apply to indices
        self._precompute(alpha_registry)

    # ---- precompute (once): alphas, net signal, leak-free accuracy, time ----
    def _precompute(self, registry):
        T = self.T
        self.alpha_matrix = np.zeros((T, C.MAX_STRATEGIES), dtype=np.float32)
        for i in range(T):
            ctx = MarketContext(close=float(self.close[i]),
                                indicators=dict(zip(ALL_INDICATOR_COLUMNS,
                                                    self.ind[i].tolist())),
                                bar_index=i, symbol=self.symbol or "",
                                minute_of_day=int(self._minute_of_day[i]))
            self.alpha_matrix[i] = registry.collect_alphas(ctx)
        self.occupancy = registry.occupancy_mask()
        self.net_signal = np.array([net_balance(self.alpha_matrix[i]) for i in range(T)],
                                   dtype=np.float32)
        self.sig_acc = accuracy_features(self.net_signal, self.close)        # (T,2) leak-free
        self.time_feats = np.stack([OB.time_features(pd.Timestamp(self.time_ns[i]))
                                    for i in range(T)]).astype(np.float32)    # (T,6)
        # 5m CCI open-gate mask: True where EITHER 5m CCI sits within +/-threshold
        # (flat/undecided short-term market) -> new directional opens are forbidden when
        # self.open_gate is on. Allowed only when BOTH |cci| > threshold (a strong move).
        thr = self.open_gate_threshold
        try:
            j30 = ALL_INDICATOR_COLUMNS.index("5m__cci30_raw")
            j100 = ALL_INDICATOR_COLUMNS.index("5m__cci100_raw")
            self.open_gate_blocked = (np.abs(self.ind[:, j30]) <= thr) | (np.abs(self.ind[:, j100]) <= thr)
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
        # v1.3.0: recent typical PRICE move (leak-free) for the sizing block = realized range
        # over the last ~240 bars (uses close[:i+1] only). pandas here is precompute, NEVER step().
        cser = pd.Series(self.close)
        self.ref_move = (cser.rolling(240, min_periods=1).max()
                         - cser.rolling(240, min_periods=1).min()).to_numpy()
        self._precompute_cross_asset()
        self._precompute_recent_context()

    # ---- v1.5.0: recent daily-movement history (leak-free), precomputed once ----
    def _precompute_recent_context(self):
        """Daily RANGE history for the recent_context block: prior-day ranges + last-week avg
        + today's range so far. Daily range = (max - min) of close per calendar day. Leak-free:
        prior days are complete; today's range is expanding up to t; week-avg uses prior days."""
        df = pd.DataFrame({"close": self.close, "day": self._dates})
        gb = df.groupby("day", sort=False)["close"]
        self._today_sofar = (gb.cummax() - gb.cummin()).to_numpy()              # expanding intraday range
        day_full = (gb.max() - gb.min()).to_numpy()                            # full-day range, day order
        p = df.groupby("day", sort=False).ngroup().to_numpy()                  # day index per bar
        n = len(day_full)
        prev_d = np.zeros(n);  prev_d[1:] = day_full[:-1]                      # day i -> range of day i-1
        prev2_d = np.zeros(n); prev2_d[2:] = day_full[:-2]
        week_d = np.array([day_full[max(0, i - 5):i].mean() if i >= 1 else day_full[i]
                           for i in range(n)])                                 # mean of up to 5 PRIOR days
        self._prev_day = prev_d[p]
        self._prev2_day = prev2_d[p]
        # day 0 has no prior days -> use today's expanding range (leak-free) instead of its full range
        self._week_avg = np.where(p == 0, self._today_sofar, week_d[p])

    # ---- v1.4.0: cross-asset perception (leak-free), precomputed once ----
    def _precompute_cross_asset(self):
        """asset-class one-hot + ATR-normalized movement + sessions -> (T, OBS_BLOCK_CROSS_ASSET).
        ATR-normalized features are SCALE-FREE so one policy reads any FTMO instrument the same."""
        T = self.T
        onehot = np.array(A.class_one_hot(self.symbol), dtype=np.float64)        # constant per symbol
        # ATR (1m, raw) from the cache; where it is missing / zero / non-finite, use the recent
        # realized range as a volatility proxy so these features are always meaningful.
        try:
            atr = self.ind[:, ALL_INDICATOR_COLUMNS.index("1m__atr14_raw")].astype(np.float64)
        except ValueError:
            atr = self.ref_move.astype(np.float64)
        # fallback when the cache has no ATR: this asset's typical 1m ATR (how it normally moves)
        # if the symbol is known, else the recent realized range.
        fb = float(self._typical_atr) if self._typical_atr else None
        bad = ~(np.isfinite(atr) & (atr > 0))
        atr = np.where(bad, (fb if fb else self.ref_move.astype(np.float64)), atr)
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)   # final guard
        c = self.close
        mv = c - pd.Series(c).shift(30).to_numpy()                              # signed recent move
        move_in_atr = np.clip(np.nan_to_num(mv / atr) / 3.0, -1.0, 1.0)         # ~[-1,1], +/-3 ATR caps
        atr_pct_price = np.clip(np.nan_to_num(atr / c) * 100.0, 0.0, 1.0)       # vol as % (scale-free)
        # vol REGIME = current ATR vs how THIS asset normally moves (per-asset typical ATR if the
        # symbol is known, else a rolling average). ~0.33 = normal, higher = unusually volatile.
        ref = np.full(T, fb) if fb else pd.Series(atr).rolling(240, min_periods=1).mean().to_numpy()
        atr_regime = np.clip(np.nan_to_num(atr / ref) / 3.0, 0.0, 1.0)
        hours = pd.to_datetime(self.time_ns).hour.to_numpy()
        asian = ((hours >= 0) & (hours < 9)).astype(np.float64)                 # ~Tokyo/Sydney
        overlap = ((hours >= 12) & (hours < 16)).astype(np.float64)            # London-NY overlap (prime)
        self.cross_asset_matrix = np.column_stack([
            np.tile(onehot, (T, 1)), move_in_atr, atr_pct_price, atr_regime, asian, overlap,
        ]).astype(np.float32)

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
        self._days_elapsed = 0          # trading days into this episode (for the time-to-pass pace)
        self._ny_reset()                # NY-session index-bonus state (per day)
        self._reset_phase2()
        return self._obs(), {"ptr": self.ptr}

    def _reset_phase2(self) -> None:
        """Reset the per-DAY two-phase state (called on reset + each midnight)."""
        self._phase2_active = False    # banked +2.5% today AND chose to keep trading (1% trail)
        self._phase2_peak = 0.0        # equity peak since the phase-2 (1%) trail started
        self._day_locked = False       # done trading for the day -> no new opens

    # ---- NY-session index bonus (DELIBERATE reward shaping; operator decision) ----
    def _ny_reset(self) -> None:
        """Per-day NY-bonus state (called on reset + each midnight)."""
        self._ny_start_realized = None     # realized PnL banked at NY session open today
        self._ny_half_qualified = False    # session closed >=50% of daily target within 2h
        self._ny_full_qualified = False    # session closed >=100% of daily target within 3h

    def _ny_qualify(self) -> None:
        """On an INDEX during the NY session (open 13:30 UTC), mark the half/full bonus QUALIFIED
        once the session's CLOSED-IN-PROFIT P&L reaches >=50% (within 2h, 13:30-15:30) / >=100%
        (within 3h, 13:30-16:30) of the daily target. Index share of closed session P&L is 1.0 in
        the single-symbol env (>=50% ok). Paid only at day-end IF the day passes (_ny_day_end_bonus)."""
        if not self._is_index:
            return
        mod = int(self._minute_of_day[self.ptr])
        if mod >= 810 and self._ny_start_realized is None:                 # 13:30 UTC -> session start
            self._ny_start_realized = self.acc.daily_realized_pnl
        if self._ny_start_realized is None:
            return
        session_closed = self.acc.daily_realized_pnl - self._ny_start_realized
        if session_closed <= 0:                                            # must be CLOSED in profit
            return
        target = self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance   # 2.5% of initial
        if not self._ny_half_qualified and 810 <= mod < 930 and session_closed >= 0.5 * target:
            self._ny_half_qualified = True
        if not self._ny_full_qualified and 810 <= mod < 990 and session_closed >= target:
            self._ny_full_qualified = True

    def _ny_day_end_bonus(self) -> float:
        """Pay the qualified NY bonus IFF the ENDING day PASSED (closed >= +2.5% of initial). If the
        day failed (or breached -> the episode already terminated before here), the bonus is erased."""
        if not (self._ny_half_qualified or self._ny_full_qualified):
            return 0.0
        target = self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance
        if self.acc.daily_realized_pnl < target:                           # day did NOT pass -> no bonus
            return 0.0
        bonus = float(getattr(V, "FTMO_NY_HALF_TARGET_BONUS", 0.0)) if self._ny_half_qualified else 0.0
        bonus += float(getattr(V, "FTMO_NY_FULL_TARGET_BONUS", 0.0)) if self._ny_full_qualified else 0.0
        return bonus

    def _flatten(self) -> None:
        """Close any open position at close[ptr] and bank it via record_close (the single
        source of truth). Used to bank the +2.5% target and the phase-2 protective stop."""
        if self.position != 0:
            realized = self.position * (self.close[self.ptr] - self.entry_price) * self.position_size
            realized -= self.cost_frac * self.close[self.ptr] * self.position_size   # exit cost
            self.th.record_close(self.acc, realized, bar_index=self.ptr)
            self.position = 0
        self.acc.equity = self.acc.balance
        self.acc.mark_equity(self.acc.equity)

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
        # two-phase day-lock: once the day is done (banked +2.5% then stopped, or hit the
        # phase-2 1% protective stop), block NEW opens until tomorrow. Holds/closes pass.
        if self._day_locked and target != 0 and target != self.position:
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

        # 4) day boundary -> PAY any qualified NY index bonus IF the ending day passed (closed
        #    >= +2.5% of initial), then reset daily FTMO + two-phase + NY-bonus state.
        if self._dates[self.ptr] != self._cur_date:
            reward += self._ny_day_end_bonus()   # 0 unless the ending day passed (else erased)
            self.acc.reset_day()
            self._reset_phase2()
            self._ny_reset()
            self._cur_date = self._dates[self.ptr]
            self._days_elapsed += 1     # one more trading day into the episode
        # 4b) NY-session index bonus QUALIFY (closed-in-profit during the 13:30-UTC NY session)
        self._ny_qualify()

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
        # 6) DAILY ENGINE (two-phase): hit +2.5% of initial -> close ALL & bank it. Then
        #    either STOP for the day (default) or, if phase2_continue, keep trading under a
        #    tight 1% trailing wall from the banked peak (give it back -> bank & stop). This
        #    is a PROTECTIVE overlay, never an episode breach. Skipped once terminated.
        if not terminated and self.cfg.two_phase_enabled:
            if rep.should_auto_flat and not self._phase2_active and not self._day_locked:
                self._flatten()                              # +2.5% reached -> bank the day
                if getattr(self.cfg, "phase2_continue", False):
                    self._phase2_active = True
                    self._phase2_peak = self.acc.equity      # fresh 1% trail starts here
                else:
                    self._day_locked = True                  # default: done for the day
            elif self._phase2_active:
                self._phase2_peak = max(self._phase2_peak, self.acc.equity)
                give_back = self._phase2_peak - self.acc.equity
                if give_back >= self._phase2_peak * self.cfg.phase2_trailing_pct / 100.0:
                    self._flatten()                          # gave back the 1% -> bank & stop
                    self._phase2_active = False
                    self._day_locked = True

        truncated = bool(self.ptr >= self.end)
        info = {"ptr": self.ptr, "equity": self.acc.equity, "position": self.position,
                "breach_reasons": rep.reasons, "daily_target_hit": rep.daily_target_hit,
                "day_locked": self._day_locked, "phase2_active": self._phase2_active,
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
            "sizing": WL.sizing_features(self.acc, self.cfg, value_per_point=self.value_per_point,
                                         ref_move=float(self.ref_move[i]), position_size=self.position_size),
            "cross_asset": self.cross_asset_matrix[i],
            "recent_context": WL.recent_context_features(
                self.acc, self.cfg, week_avg=float(self._week_avg[i]), prev_day=float(self._prev_day[i]),
                prev2=float(self._prev2_day[i]), today_sofar=float(self._today_sofar[i]),
                typical_range=self._typical_range, days_elapsed=self._days_elapsed),
            "alpha_streak": np.minimum(self.streak_matrix[i], C.ALPHA_STREAK_CAP) / float(C.ALPHA_STREAK_CAP),
        })

    def _portfolio_block(self):
        self.acc.open_positions = 1 if self.position != 0 else 0
        self.acc.net_exposure = float(self.position)
        self.acc.gross_exposure = float(abs(self.position))
        self.acc.unrealized_pnl = self.acc.equity - self.acc.balance
        self.acc.largest_position_dir = int(np.sign(self.position))
        return WL.portfolio_features(self.acc)
