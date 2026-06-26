# =====================================================================
# WHEN 2026-06-26 (Phase 2) | WHO Claude for Monty
# WHY  ONE bot trading the WHOLE FTMO book from ONE shared equity/drawdown pot --
#      the core goal. The policy decides ONE symbol at a time while SEEING how exposed
#      the shared pot already is (the account + portfolio observation blocks), so it
#      learns to BALANCE risk across simultaneous positions. Because decisions are
#      per-symbol, this scales from 4 symbols to the full FTMO broker list (130+)
#      WITHOUT changing the locked 479 observation.
# WHERE src/env/portfolio_env.py
# HOW  Reuse one TradingEnv per symbol for its PRECOMPUTED per-symbol blocks (alphas,
#      streaks, cross-asset, etc.); maintain ONE AccountState (the pot) + per-symbol
#      positions; cycle symbol-by-symbol each bar; mark the pot, run the FTMO engine
#      (breach / +10% pass / per-day reset) at the POT level; reward = pot equity change.
# DEPENDS_ON: src/env/trading_env.py, src/account/*, src/risk/breach_detector.py,
#             src/observation/builder.py, src/signals/*, config/*
# USED_BY: src/training (train one portfolio policy), src/training/daily_report.py
# CHANGE_NOTES(IRAC): I: single-symbol envs can't learn portfolio RISK ALLOCATION across
#   one pot. R: operator 2026-06-26 -- one bot on all four symbols, generalizing to the
#   full live broker list. A: a shared-pot PortfolioEnv with per-symbol decisions + a
#   portfolio-aggregated obs; obs stays 479, actions stay {HOLD,BUY,SELL,CLOSE}, so the
#   existing MlpPolicy/trainer/fingerprint all apply. C: one policy learns to balance the
#   book toward a consistent portfolio pass, and scales to the whole FTMO universe live.
# =====================================================================
"""PortfolioEnv: one policy trades ALL symbols from ONE shared pot (obs stays 479)."""
from __future__ import annotations
import numpy as np
from config import constants as C
from config.ftmo_config import load_active_config
from config import asset_specs as A
from src.account.account_state import AccountState
from src.account.trade_history import TradeHistory
from src.account import win_loss_features as WL
from src.risk import breach_detector as BD
from src.observation import builder as OB
from src.signals.signal_summary import summarize, net_balance
from src.signals.signal_memory import last5_from_series
from src.env.trading_env import TradingEnv

_TARGET = {C.ACTION_HOLD: None, C.ACTION_BUY: 1, C.ACTION_SELL: -1, C.ACTION_CLOSE: 0}


def align_symbol_data(symbol_data: dict) -> dict:
    """Inner-join symbols on their timestamps so PortfolioEnv gets equal-length, aligned arrays.

    Real caches differ in length (FX trades ~24/5, an index has its own hours). This keeps only the
    bars all symbols share, so positions move on the same clock. Returns {symbol: (ind, close, time_ns)}.
    """
    keys = list(symbol_data)
    times = {k: np.asarray(symbol_data[k][2]).astype(np.int64).ravel() for k in keys}
    common = sorted(set.intersection(*[set(times[k].tolist()) for k in keys]))
    if not common:
        raise ValueError("symbols share no common timestamps -- cannot align the portfolio")
    common_arr = np.array(common, dtype=np.int64)
    out = {}
    for k in keys:
        ind, close, _ = symbol_data[k]
        pos = {int(ts): i for i, ts in enumerate(times[k].tolist())}
        idx = np.array([pos[int(ts)] for ts in common], dtype=np.int64)
        out[k] = (np.asarray(ind)[idx], np.asarray(close)[idx], common_arr)
    return out


class PortfolioEnv:
    """One shared pot, many symbols, per-symbol decisions. Gym-style single obs(479)/action(4)."""

    observation_shape = C.OBS_SHAPE
    n_actions = C.N_ACTIONS

    def __init__(self, symbol_data: dict, registry_factory, *, cfg=None, warmup: int = 200,
                 breach_penalty: float = 1.0, reward_scale: float = 1.0, pass_bonus: float | None = None):
        self.cfg = cfg or load_active_config()
        self.symbols = list(symbol_data)
        if not self.symbols:
            raise ValueError("PortfolioEnv needs at least one {symbol: (indicators, close, time_ns)}")
        self.breach_penalty = float(breach_penalty)
        self.reward_scale = float(reward_scale)
        self.pass_bonus = float(self.breach_penalty if pass_bonus is None else pass_bonus)
        self.profit_target_frac = float(getattr(self.cfg, "profit_target_total_pct", 10.0)) / 100.0
        # one TradingEnv per symbol -> its PRECOMPUTED per-symbol arrays (we never call its step)
        self.subs: dict[str, TradingEnv] = {}
        T = None
        for sym, (ind, close, time_ns) in symbol_data.items():
            ps = A.calibrated_position_size(sym) if sym in A.SPECS else 100_000.0
            sub = TradingEnv(np.asarray(ind), np.asarray(close), np.asarray(time_ns),
                             registry_factory(), cfg=self.cfg, symbol=sym, position_size=ps, warmup=warmup)
            self.subs[sym] = sub
            if T is None:
                T = sub.T
            elif sub.T != T:
                raise ValueError("all symbols must be time-aligned (same length / timestamps)")
        self.T = int(T)
        self.warmup = int(warmup)
        self._dates = next(iter(self.subs.values()))._dates    # assumes aligned timestamps
        self.reset()

    # ---- episode ----
    def reset(self, *, seed=None, options=None):
        self.acc = AccountState(starting_balance=self.cfg.starting_balance)
        self.th = TradeHistory()
        self.t = int(self.warmup)
        self.j = 0                                             # current symbol index
        self.position = {s: 0 for s in self.symbols}
        self.entry = {s: float(self.subs[s].close[self.t]) for s in self.symbols}
        self._cur_date = self._dates[self.t]
        self._days_elapsed = 0
        self._mark()
        return self._obs(), {"t": self.t}

    @property
    def ptr(self) -> int:                                      # so daily_report / introspect work unchanged
        return self.t

    def _cur(self):
        return self.symbols[self.j], self.subs[self.symbols[self.j]]

    # ---- shared pot bookkeeping ----
    def _mark(self):
        unreal = 0.0
        for s in self.symbols:
            if self.position[s] != 0:
                sub = self.subs[s]
                unreal += self.position[s] * (sub.close[self.t] - self.entry[s]) * sub.position_size
        self.acc.equity = self.acc.balance + unreal
        self.acc.mark_equity(self.acc.equity)

    def _set_aggregates(self):
        pos = self.position
        S = max(1, len(self.symbols))
        self.acc.open_positions = sum(1 for p in pos.values() if p != 0)
        self.acc.net_exposure = float(np.clip(sum(pos.values()) / S, -1.0, 1.0))
        self.acc.gross_exposure = float(np.clip(sum(abs(p) for p in pos.values()) / S, 0.0, 1.0))
        self.acc.unrealized_pnl = self.acc.equity - self.acc.balance
        big = max(self.symbols, key=lambda s: abs(pos[s]))
        self.acc.largest_position_dir = int(np.sign(pos[big]))

    # ---- observation (479; per-symbol blocks + SHARED account/portfolio blocks) ----
    def _obs(self):
        sym, sub = self._cur()
        i = self.t
        self._set_aggregates()
        return OB.build_from_blocks({
            "indicators": sub.ind[i],
            "alpha_values": sub.alpha_matrix[i],
            "alpha_mask": sub.occupancy,
            "alpha_summary": summarize(sub.alpha_matrix[i], sub.occupancy),
            "signal_memory": last5_from_series(sub.net_signal, i),
            "signal_accuracy": sub.sig_acc[i],
            "account_daily": WL.daily_features(self.acc, self.cfg),         # SHARED pot
            "account_episode": WL.episode_features(self.acc, self.cfg),     # SHARED pot
            "time": sub.time_feats[i],
            "portfolio": WL.portfolio_features(self.acc),                   # aggregated over ALL symbols
            "sizing": WL.sizing_features(self.acc, self.cfg, value_per_point=sub.value_per_point,
                                         ref_move=float(sub.ref_move[i]), position_size=sub.position_size),
            "cross_asset": sub.cross_asset_matrix[i],
            "recent_context": WL.recent_context_features(
                self.acc, self.cfg, week_avg=float(sub._week_avg[i]), prev_day=float(sub._prev_day[i]),
                prev2=float(sub._prev2_day[i]), today_sofar=float(sub._today_sofar[i]),
                typical_range=sub._typical_range, days_elapsed=self._days_elapsed),
            "alpha_streak": np.minimum(sub.streak_matrix[i], C.ALPHA_STREAK_CAP) / float(C.ALPHA_STREAK_CAP),
        })

    # ---- step: decide ONE symbol, advance the cursor, mark the pot ----
    def step(self, action: int):
        sym, sub = self._cur()
        t = self.t
        eq_before = self.acc.equity
        target = _TARGET[int(action)]
        if target is None:
            target = self.position[sym]                        # HOLD
        if target != self.position[sym]:
            if self.position[sym] != 0:                        # realize the closing leg into the POT
                realized = self.position[sym] * (sub.close[t] - self.entry[sym]) * sub.position_size
                realized -= sub.cost_frac * sub.close[t] * sub.position_size
                self.th.record_close(self.acc, realized, bar_index=t)   # single P&L truth
            self.position[sym] = target
            if target != 0:
                self.entry[sym] = float(sub.close[t])
                ecost = sub.cost_frac * sub.close[t] * sub.position_size
                self.acc.balance -= ecost
                self.acc.daily_realized_pnl -= ecost
                self.acc.episode_realized_pnl -= ecost

        # advance the cursor: next symbol; when it wraps, advance the bar
        self.j += 1
        bar_advanced = False
        if self.j >= len(self.symbols):
            self.j = 0
            self.t = min(t + 1, self.T - 1)
            bar_advanced = True
        self._mark()
        reward = float((self.acc.equity - eq_before) / self.cfg.starting_balance) * self.reward_scale

        terminated = False
        if bar_advanced:
            if self._dates[self.t] != self._cur_date:          # midnight -> per-day FTMO reset
                self.acc.reset_day()
                self._cur_date = self._dates[self.t]
                self._days_elapsed += 1
            rep = BD.detect(self.acc, self.cfg)                # breach on the SHARED pot
            if rep.breached:
                self.acc.episode_breached = True
                terminated = True
                reward -= self.breach_penalty
            elif self.acc.equity >= self.cfg.starting_balance * (1.0 + self.profit_target_frac):
                self.acc.episode_passed = True                 # +10% on the pot -> PASS
                terminated = True
                reward += self.pass_bonus
            if self.t >= self.T - 1:
                terminated = True

        info = {"symbol": sym, "t": t, "equity": self.acc.equity,
                "open_positions": self.acc.open_positions, "positions": dict(self.position)}
        return self._obs(), reward, terminated, False, info
