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
#   [2026-06-26b] I: the portfolio path (1) replayed ONE identical trajectory across all vec
#   workers (reset ignored seed, no window) = no exploration diversity, and (2) had NO two-phase
#   +2.5% bank-and-stop (it lived only in single-symbol TradingEnv) so the trained bot ignored a
#   documented FTMO rule. R: keep obs(479)/FTMO numbers; add diversity + pot-level two-phase.
#   A: random_window/seed per worker (DIFFERENT stretches) + pot-level two-phase (bank ALL at
#   +2.5%, stop or 1% trail) mirroring TradingEnv; episode/window end is now truncated (breach/pass
#   stay terminated). C: parallel envs actually diversify, and the portfolio bot banks +2.5% & stops.
#   [2026-06-27] ALPHA-SHAPING (ON by default — operator decision). DELIBERATE departure from the
#   "reward = equity only / NEVER alpha" rule (still true for single-symbol TradingEnv + its test).
#   I: operator wants the bot to (a) USE the alphas and (b) BEAT them. A: small reward terms tied to the
#   firing-alpha consensus -- a bonus for a profitable close that agreed with >=50% of alphas, a bonus for
#   a closed trade that OUT-EARNED following the consensus, and a penalty for OPENING against >=50%; every
#   bonus is CAPPED at the trade's own PnL and only pays when the day is net up; toggle via cfg.alpha_reward_enabled.
#   C: reward is no longer alpha-independent in PortfolioEnv by default (documented, reversible).
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


def build_portfolio_subs(symbol_data: dict, registry_factory, *, cfg=None, warmup: int = 200,
                         progress: bool = True, feature_cache_dir: str | None = None) -> dict:
    """Build ONE TradingEnv per symbol -- the expensive precompute (alphas/streaks/cross-asset over the
    whole history) -- so the result can be SHARED across every vectorised PortfolioEnv worker.

    The per-symbol arrays are READ-ONLY after this (PortfolioEnv only reads them; it never calls sub.step
    or mutates sub state), so sharing one copy across all workers is safe AND avoids rebuilding them once
    per worker -- which was 4 workers x 4 symbols = 16 redundant builds over 1.8M bars (the multi-hour
    "stuck building" hang). Build once here, share everywhere.

    If `feature_cache_dir` is given, each symbol's features are LOADED from disk (Google Drive on Colab)
    when the fingerprint matches exactly, else BUILT and SAVED -- so re-runs skip the slow precompute
    with ZERO risk of loading stale features (see src/data/feature_cache.py).
    """
    cfg = cfg or load_active_config()
    bal = float(getattr(cfg, "starting_balance", 100_000.0))   # position sizes SCALE with the account size
    fc = None
    if feature_cache_dir:
        from src.data import feature_cache as fc
    subs: dict = {}
    n = len(symbol_data)
    for k, (sym, (ind, close, time_ns)) in enumerate(symbol_data.items(), 1):
        ind = np.asarray(ind); close = np.asarray(close); time_ns = np.asarray(time_ns)
        # calibrate lots to ~2.5%/day FOR THIS ACCOUNT SIZE (so behavior is identical at $10k or $200k);
        # unknown symbols fall back to "1 lot = the account" which also scales with the balance.
        ps = A.calibrated_position_size(sym, account=bal) if sym in A.SPECS else bal
        reg = registry_factory()
        cached = fc.load(feature_cache_dir, sym, ind, close, time_ns, reg) if fc else None
        if cached is not None:
            if progress:
                print(f"      [{k}/{n}] {sym}: loaded saved features ✓ (skipped the rebuild)", flush=True)
            subs[sym] = TradingEnv(ind, close, time_ns, reg, cfg=cfg, symbol=sym,
                                   position_size=ps, warmup=warmup, precomputed=cached)
            continue
        if progress:
            print(f"      [{k}/{n}] building features for {sym} ...", flush=True)
        sub = TradingEnv(ind, close, time_ns, reg, cfg=cfg, symbol=sym, position_size=ps,
                         warmup=warmup, progress=progress)
        if fc:
            try:
                path = fc.save(feature_cache_dir, sym, ind, close, time_ns, reg, sub)
                if progress:
                    print(f"      [{k}/{n}] {sym}: saved features -> {path}", flush=True)
            except Exception as e:   # caching is best-effort; never block training on a save failure
                if progress:
                    print(f"      [{k}/{n}] {sym}: (could not save features: {e})", flush=True)
        subs[sym] = sub
    return subs


class PortfolioEnv:
    """One shared pot, many symbols, per-symbol decisions. Gym-style single obs(479)/action(4)."""

    observation_shape = C.OBS_SHAPE
    n_actions = C.N_ACTIONS

    def __init__(self, symbol_data: dict | None = None, registry_factory=None, *, cfg=None, warmup: int = 200,
                 breach_penalty: float = 1.0, reward_scale: float = 1.0, pass_bonus: float | None = None,
                 window: int | None = None, random_window: bool = False, seed: int | None = None,
                 subs: dict | None = None, continue_after_pass: bool = False):
        self.cfg = cfg or load_active_config()
        self.breach_penalty = float(breach_penalty)
        self.reward_scale = float(reward_scale)
        self.pass_bonus = float(self.breach_penalty if pass_bonus is None else pass_bonus)
        # TRAINING keeps going past +10% (continue_after_pass=True) to learn CONSISTENCY (string 4 daily
        # passes in a row, over and over). EVAL / a real challenge ends at +10% (default False).
        self.continue_after_pass = bool(continue_after_pass)
        self.profit_target_frac = float(getattr(self.cfg, "profit_target_total_pct", 10.0)) / 100.0
        # ALPHA-SHAPING (opt-in via cfg; ON by default 2026-06-27). Small terms tied to the alpha consensus;
        # every bonus is CAPPED at the trade's own PnL (amplify a real win, never fabricate reward).
        self._alpha_on = bool(getattr(self.cfg, "alpha_reward_enabled", False))
        self._alpha_agree = float(getattr(self.cfg, "alpha_agree_bonus", 0.0))
        self._alpha_against = float(getattr(self.cfg, "alpha_against_penalty", 0.0))
        self._alpha_beat = float(getattr(self.cfg, "alpha_beat_bonus", 0.0))
        # one TradingEnv per symbol -> its PRECOMPUTED per-symbol arrays (we never call its step). These can
        # be PRE-BUILT and SHARED across vec workers (read-only after precompute), so the heavy precompute
        # runs ONCE for all workers instead of once each -- see build_portfolio_subs().
        if subs is not None:
            self.subs = subs
            self.symbols = list(subs)
        elif symbol_data:
            self.subs = build_portfolio_subs(symbol_data, registry_factory, cfg=self.cfg,
                                             warmup=warmup, progress=False)
            self.symbols = list(self.subs)
        else:
            raise ValueError("PortfolioEnv needs symbol_data {symbol:(ind,close,time)} OR pre-built subs=")
        T = None
        for sym, sub in self.subs.items():
            if T is None:
                T = sub.T
            elif sub.T != T:
                raise ValueError("all symbols must be time-aligned (same length / timestamps)")
        self.T = int(T)
        self.warmup = int(warmup)
        # Per-worker EPISODE DIVERSITY: with random_window, each reset() starts at a RANDOM bar and
        # runs `window` bars. Vectorised workers get DIFFERENT seeds -> they explore DIFFERENT stretches
        # of history instead of replaying the same trajectory (no diversity = wasted parallel envs).
        self.window = int(window) if window else None
        self.random_window = bool(random_window)
        self.rng = np.random.default_rng(seed)
        self._dates = next(iter(self.subs.values()))._dates    # assumes aligned timestamps
        self.reset()

    # ---- episode ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.acc = AccountState(starting_balance=self.cfg.starting_balance)
        self.th = TradeHistory()
        if self.random_window and self.window:
            # Clamp the window to AT MOST half the usable span so there is ALWAYS room to sample a
            # VARIED start. Otherwise, on a SHORT history (e.g. a trimmed --from/--to first run) where
            # the configured window is >= the data, every worker pins to `warmup` and the parallel
            # copies become identical again -- the silent zero-diversity trap this whole fix removes.
            span = self.T - self.warmup
            eff_window = min(self.window, max(1, span // 2))
            hi = max(self.warmup + 1, self.T - eff_window)
            self.t = int(self.rng.integers(self.warmup, hi))    # random start -> per-worker diversity
            self.end = min(self.T - 1, self.t + eff_window)
        else:
            self.t = int(self.warmup)                           # full walk-forward (eval / daily report)
            self.end = self.T - 1
        self.j = 0                                             # current symbol index
        self.position = {s: 0 for s in self.symbols}
        self.entry = {s: float(self.subs[s].close[self.t]) for s in self.symbols}
        self._cur_date = self._dates[self.t]
        self._days_elapsed = 0
        self._reset_phase2()                                   # per-day two-phase state on the shared pot
        self._daily_pass_streak = 0                            # consecutive days that banked +2.5% (consistency)
        self._day_passed = False                               # did TODAY bank +2.5% (net of fees)?
        self._entry_agreed = {}                                # per-open: did the entry agree with >=50% firing alphas?
        self._entry_alpha_dir = {}                             # per-open: the alpha consensus direction at entry
        self._mark()
        return self._obs(), {"t": self.t}

    def _alpha_consensus(self, sub, t, d):
        """(agree_frac, disagree_frac, net_dir) among the FIRING, UNMASKED alphas at bar t, for direction d.
        agree = share of firing alphas pointing WITH d; net_dir = the alphas' overall lean (+1/-1/0)."""
        am = sub.alpha_matrix[t]
        occ = np.asarray(sub.occupancy, dtype=bool)
        fired = (am != 0) & occ
        nf = int(fired.sum())
        if nf == 0:
            return 0.0, 0.0, 0
        buys = int(((am > 0) & fired).sum())
        sells = int(((am < 0) & fired).sum())
        net_dir = 1 if buys > sells else (-1 if sells > buys else 0)
        if d > 0:
            return buys / nf, sells / nf, net_dir
        if d < 0:
            return sells / nf, buys / nf, net_dir
        return 0.0, 0.0, net_dir

    def _reset_phase2(self) -> None:
        """Reset the per-DAY two-phase state on the SHARED pot (reset + each midnight)."""
        self._phase2_active = False    # banked +2.5% today AND kept trading (1% trail)
        self._phase2_peak = 0.0        # equity peak since the phase-2 (1%) trail started
        self._day_locked = False       # day done (banked +2.5% then stopped) -> no new opens

    def _flatten_all(self) -> None:
        """Close EVERY open position at the current bar and bank it into the shared pot via
        record_close (the single P&L truth). Used to bank +2.5% and the phase-2 protective stop."""
        for s in self.symbols:
            if self.position[s] != 0:
                sub = self.subs[s]
                realized = self.position[s] * (sub.close[self.t] - self.entry[s]) * sub.position_size
                realized -= sub.cost_frac * sub.close[self.t] * sub.position_size   # exit cost
                self.th.record_close(self.acc, realized, bar_index=self.t)
                self.position[s] = 0
        self.acc.equity = self.acc.balance
        self.acc.mark_equity(self.acc.equity)

    def _pending_exit_cost(self) -> float:
        """Total transaction cost to close ALL open positions right now -- so the +2.5% bank triggers on
        TRUE post-fee equity (what you'd actually keep), not the gross open-profit figure."""
        c = 0.0
        for s in self.symbols:
            if self.position[s] != 0:
                sub = self.subs[s]
                c += sub.cost_frac * sub.close[self.t] * sub.position_size
        return c

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
        # two-phase day-lock: once the pot banked +2.5% (then stopped), block NEW opens on EVERY
        # symbol until tomorrow. Holds (target==current) and closes (target==0) still pass through.
        if self._day_locked and target != 0 and target != self.position[sym]:
            target = 0
        alpha_shaping = 0.0
        if target != self.position[sym]:
            if self.position[sym] != 0:                        # realize the closing leg into the POT
                old_dir = self.position[sym]
                realized = old_dir * (sub.close[t] - self.entry[sym]) * sub.position_size
                realized -= sub.cost_frac * sub.close[t] * sub.position_size
                self.th.record_close(self.acc, realized, bar_index=t)   # single P&L truth
                # ALPHA-SHAPING bonuses (paid only on a profitable close with the DAY net up; capped at PnL)
                day0 = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
                if self._alpha_on and realized > 0.0 and self.acc.balance > day0:
                    pnl_frac = realized / self.cfg.starting_balance
                    bonus = 0.0
                    if self._entry_agreed.get(sym, False):                  # USE the alphas: agreed >=50% and won
                        bonus += self._alpha_agree
                    move = sub.close[t] - self.entry[sym]                   # BEAT the alphas: out-earned a follow
                    alpha_gross = self._entry_alpha_dir.get(sym, 0) * move * sub.position_size
                    bot_gross = old_dir * move * sub.position_size
                    if bot_gross > alpha_gross:
                        bonus += min(self._alpha_beat, (bot_gross - alpha_gross) / self.cfg.starting_balance)
                    alpha_shaping += min(bonus, pnl_frac)                   # CAP: bonus can never exceed the trade PnL
                self._entry_agreed.pop(sym, None); self._entry_alpha_dir.pop(sym, None)
            self.position[sym] = target
            if target != 0:
                self.entry[sym] = float(sub.close[t])
                ecost = sub.cost_frac * sub.close[t] * sub.position_size
                self.acc.balance -= ecost
                self.acc.daily_realized_pnl -= ecost
                self.acc.episode_realized_pnl -= ecost
                # record the alpha consensus AT ENTRY (for the close-time bonuses) + penalise fighting it
                agree, disagree, net_dir = self._alpha_consensus(sub, t, target)
                self._entry_agreed[sym] = (agree >= 0.5)
                self._entry_alpha_dir[sym] = net_dir
                if self._alpha_on and disagree >= 0.5:                      # opened AGAINST >=50% firing alphas
                    alpha_shaping -= self._alpha_against

        # advance the cursor: next symbol; when it wraps, advance the bar
        self.j += 1
        bar_advanced = False
        if self.j >= len(self.symbols):
            self.j = 0
            self.t = min(t + 1, self.T - 1)
            bar_advanced = True
        self._mark()
        reward = float((self.acc.equity - eq_before) / self.cfg.starting_balance) * self.reward_scale
        reward += alpha_shaping            # alpha-shaping (0 unless enabled AND an alpha consensus event fired)

        terminated = False
        truncated = False
        daily_target_hit = False
        if bar_advanced:
            if self._dates[self.t] != self._cur_date:          # midnight -> per-day FTMO reset
                # consistency reward: did the day that just ENDED bank +2.5%? 4 such days IN A ROW = a BIG
                # bonus (and we KEEP training for more streaks). A failed day breaks the streak.
                if self._day_passed:
                    self._daily_pass_streak += 1
                    if self._daily_pass_streak % 4 == 0:
                        reward += self.pass_bonus              # +10% / 4-in-a-row achieved -> big bonus
                else:
                    self._daily_pass_streak = 0
                self.acc.reset_day()
                self._reset_phase2()                           # new day -> clear the day-lock / phase-2
                self._day_passed = False
                self._cur_date = self._dates[self.t]
                self._days_elapsed += 1
            rep = BD.detect(self.acc, self.cfg)                # breach on the SHARED pot
            daily_target_hit = rep.daily_target_hit
            if rep.breached:
                self.acc.episode_breached = True
                terminated = True
                reward -= self.breach_penalty
            elif self.acc.equity >= self.cfg.starting_balance * (1.0 + self.profit_target_frac):
                self.acc.episode_passed = True                 # +10% on the pot -> PASS
                if not self.continue_after_pass:               # eval / real challenge ENDS at +10%
                    terminated = True
                    reward += self.pass_bonus
                # training (continue_after_pass): keep going for CONSISTENCY; the 4-in-a-row bonus rewards it
            # DAILY ENGINE (two-phase) on the SHARED pot: bank +2.5% of initial NET OF FEES -> close ALL,
            # then STOP for the day (phase2_continue=False) or keep trading under a 1% trailing wall from
            # the banked peak. A PROTECTIVE overlay, never an episode breach. Skipped if breached.
            if not terminated and self.cfg.two_phase_enabled:
                day0 = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
                target_amt = self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance
                net_equity = self.acc.equity - self._pending_exit_cost()   # what you'd actually KEEP after closing
                hit_net = (net_equity - day0) >= target_amt
                if hit_net and not self._phase2_active and not self._day_locked:
                    self._flatten_all()                        # +2.5% (net) reached -> bank the whole book
                    self._day_passed = True                    # counts toward the 4-in-a-row streak
                    if getattr(self.cfg, "phase2_continue", False):
                        self._phase2_active = True
                        self._phase2_peak = self.acc.equity    # fresh 1% trail starts here
                    else:
                        self._day_locked = True                # stop for the day
                elif self._phase2_active:
                    self._phase2_peak = max(self._phase2_peak, self.acc.equity)
                    give_back = self._phase2_peak - self.acc.equity
                    if give_back >= self._phase2_peak * self.cfg.phase2_trailing_pct / 100.0:
                        self._flatten_all()                    # gave back the 1% -> bank & stop
                        self._phase2_active = False
                        self._day_locked = True
            if self.t >= self.end:                             # reached the (windowed) episode end
                truncated = True

        info = {"symbol": sym, "t": t, "equity": self.acc.equity,
                "open_positions": self.acc.open_positions, "positions": dict(self.position),
                "day_locked": self._day_locked, "phase2_active": self._phase2_active,
                "daily_target_hit": daily_target_hit, "daily_pass_streak": self._daily_pass_streak}
        return self._obs(), reward, terminated, truncated, info
