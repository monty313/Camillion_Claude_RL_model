# =====================================================================
# WHEN 2026-06-26 (Phase 2) | WHO Claude for Monty
# WHY  ONE bot trading the WHOLE FTMO book from ONE shared equity/drawdown pot --
#      the core goal. The policy decides ONE symbol at a time while SEEING how exposed
#      the shared pot already is (the account + portfolio observation blocks), so it
#      learns to BALANCE risk across simultaneous positions. Because decisions are
#      per-symbol, this scales from 4 symbols to the full FTMO broker list (130+)
#      WITHOUT changing the locked 499 observation.
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
#   portfolio-aggregated obs; obs stays 499, actions stay {HOLD,BUY,SELL,CLOSE}, so the
#   existing MlpPolicy/trainer/fingerprint all apply. C: one policy learns to balance the
#   book toward a consistent portfolio pass, and scales to the whole FTMO universe live.
#   [2026-06-26b] I: the portfolio path (1) replayed ONE identical trajectory across all vec
#   workers (reset ignored seed, no window) = no exploration diversity, and (2) had NO two-phase
#   +2.5% bank-and-stop (it lived only in single-symbol TradingEnv) so the trained bot ignored a
#   documented FTMO rule. R: keep obs(499)/FTMO numbers; add diversity + pot-level two-phase.
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
"""PortfolioEnv: one policy trades ALL symbols from ONE shared pot (obs stays 499)."""
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
from src.observation import trade_risk as TR
from src.observation.hug_pressure import (IDX_DOMINANT_SIDE as _HUG_DOM, IDX_CONTINUATION_3PLUS as _HUG_CONT3,
                                          HUG_EXH_THR, HUG_DECAY_THR, HUG_LOC_THR)
from src.observation.momentum_scores import (IDX_EXHAUSTION as _MOM_EXH, IDX_LOCATION as _MOM_LOC,
                                             IDX_DECAY as _MOM_DEC)
from src.strategies.alpha_pack import CONVICTION_ALIGN_CAP
from src.env.trading_env import TradingEnv

_TARGET = {C.ACTION_HOLD: None, C.ACTION_BUY: 1, C.ACTION_SELL: -1, C.ACTION_CLOSE: 0}


def unpack_symbol(v):
    """A symbol entry is (ind, close, time_ns) OR (ind, close, time_ns, aux). Return all four,
    aux=None for the legacy 3-tuple (synthetic tests / old caches). aux (T,32) = OHLC obs block + DI."""
    return v[0], v[1], v[2], (v[3] if len(v) > 3 else None)


def align_symbol_data(symbol_data: dict) -> dict:
    """Inner-join symbols on their timestamps so PortfolioEnv gets equal-length, aligned arrays.

    Real caches differ in length (FX trades ~24/5, an index has its own hours). This keeps only the
    bars all symbols share, so positions move on the same clock. Returns {symbol: (ind, close, time_ns)}
    -- or (ind, close, time_ns, aux) when an aux array (OHLC+DI) is present, trimmed the SAME way.
    """
    keys = list(symbol_data)
    times = {k: np.asarray(symbol_data[k][2]).astype(np.int64).ravel() for k in keys}
    common = sorted(set.intersection(*[set(times[k].tolist()) for k in keys]))
    if not common:
        raise ValueError("symbols share no common timestamps -- cannot align the portfolio")
    common_arr = np.array(common, dtype=np.int64)
    out = {}
    for k in keys:
        ind, close, _t, aux = unpack_symbol(symbol_data[k])
        pos = {int(ts): i for i, ts in enumerate(times[k].tolist())}
        idx = np.array([pos[int(ts)] for ts in common], dtype=np.int64)
        trimmed = (np.asarray(ind)[idx], np.asarray(close)[idx], common_arr)
        if aux is not None:
            trimmed = trimmed + (np.asarray(aux)[idx],)   # trim aux on the SAME shared bars
        out[k] = trimmed
    return out


def build_portfolio_subs(symbol_data: dict, registry_factory, *, cfg=None, warmup: int = 200,
                         progress: bool = True, feature_cache_dir: str | None = None,
                         risk_pct: float | None = None) -> dict:
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
    for k, (sym, v) in enumerate(symbol_data.items(), 1):
        ind, close, time_ns, aux = unpack_symbol(v)
        ind = np.asarray(ind); close = np.asarray(close); time_ns = np.asarray(time_ns)
        # calibrate lots so that ONE typical full day on this symbol = `risk_pct`% of the account (default
        # 2.5%). For a MULTI-symbol book the per-symbol risk should be SMALLER (positions stack): e.g. 4
        # symbols x 0.6% = ~2.4% worst-case day, inside the 4% wall, while still able to sum to the +2.5%
        # target. `risk_pct` is the per-trade/day risk knob (operator 2026-06-28). Unknown symbols fall back
        # to "1 lot = the account" scaled by risk_pct/2.5 so the knob still shrinks them.
        tp = 2.5 if risk_pct is None else float(risk_pct)
        ps = (A.calibrated_position_size(sym, account=bal, target_pct=tp) if sym in A.SPECS
              else bal * (tp / 2.5))
        reg = registry_factory()
        cached = fc.load(feature_cache_dir, sym, ind, close, time_ns, reg, aux=aux) if fc else None
        if cached is not None:
            if progress:
                print(f"      [{k}/{n}] {sym}: loaded saved features ✓ (skipped the rebuild)", flush=True)
            subs[sym] = TradingEnv(ind, close, time_ns, reg, cfg=cfg, symbol=sym,
                                   position_size=ps, warmup=warmup, precomputed=cached, aux=aux)
            continue
        if progress:
            print(f"      [{k}/{n}] building features for {sym} ...", flush=True)
        sub = TradingEnv(ind, close, time_ns, reg, cfg=cfg, symbol=sym, position_size=ps,
                         warmup=warmup, progress=progress, aux=aux)
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
    """One shared pot, many symbols, per-symbol decisions. Gym-style single obs(499)/action(4)."""

    observation_shape = C.OBS_SHAPE
    n_actions = C.N_ACTIONS

    def __init__(self, symbol_data: dict | None = None, registry_factory=None, *, cfg=None, warmup: int = 200,
                 breach_penalty: float | None = None, reward_scale: float = 1.0, pass_bonus: float | None = None,
                 window: int | None = None, random_window: bool = False, seed: int | None = None,
                 subs: dict | None = None, continue_after_pass: bool = False,
                 bb_stop_enabled: bool = False, risk_per_trade_pct: float | None = None,
                 open_gate: bool = False):
        self.cfg = cfg or load_active_config()
        # breach cliff + pass bonus now default from the config (breach 0.2, pass 1.0); explicit args override.
        self.breach_penalty = float(breach_penalty if breach_penalty is not None
                                    else getattr(self.cfg, "breach_penalty", 1.0))
        self.reward_scale = float(reward_scale)
        self.pass_bonus = float(pass_bonus if pass_bonus is not None
                                else getattr(self.cfg, "pass_bonus", self.breach_penalty))
        self._dd_coef = float(getattr(self.cfg, "dd_proximity_coef", 0.0))   # drawdown-proximity penalty
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
        # per-DAY consistency: a "won day" ENDS >= +2.5% of initial (scored at midnight, after any give-back)
        self._day_pass_reward = float(getattr(self.cfg, "day_pass_reward", 0.0))
        self._day_fail_penalty = float(getattr(self.cfg, "day_fail_penalty", 0.0))
        # ESCALATING streak bonus (operator 2026-06-29): every ADDITIONAL consecutive won day pays more (capped),
        # replacing the old every-4th jackpot (no lumpy multiple to gamble for).
        self._streak_bonus = float(getattr(self.cfg, "streak_bonus", 0.0))
        self._streak_bonus_cap = float(getattr(self.cfg, "streak_bonus_cap", 0.0))
        # SEEK-THE-TARGET vs HIDE rebalance (operator 2026-06-28): a dense reward for NEW progress toward
        # the +2.5%/day target (high-water-mark) so the bot actively SEEKS profit, + a penalty for a day
        # with ZERO trades so "hiding" isn't free. Both 0.0 = pre-rebalance reward. (See config/variables.py.)
        self._seek_w = float(getattr(self.cfg, "target_seek_weight", 0.0))
        self._idle_pen = float(getattr(self.cfg, "idle_day_penalty", 0.0))
        # v1.7.0 TRADE-RISK behaviours (all DEFAULT OFF -> existing trajectories byte-identical; the
        # training path turns them on). bb_stop = auto-close at the 1m BB(10,1) opposite band; risk-based
        # sizing = size each entry so a stop-out loses ~risk_per_trade_pct% of the pot (capped at the
        # configured size); band_stack_bonus / reentry_bonus = small PnL-capped CLOSE bonuses (cfg coefs).
        self._bb_stop_on = bool(bb_stop_enabled)
        self._risk_pct = None if risk_per_trade_pct is None else float(risk_per_trade_pct)
        # 5m CCI open-gate: block a NEW directional open when the 5m is FLAT (both CCI30 & CCI100 in +/-50)
        # -> "don't trade the chop" (operator 2026-06-29). OFF by default; the training path turns it on.
        self._open_gate = bool(open_gate)
        self._band_bonus = float(getattr(self.cfg, "band_stack_bonus", 0.0))
        self._reentry_bonus = float(getattr(self.cfg, "reentry_bonus", 0.0))
        # CONVICTION bonus (operator 2026-06-29): paid when >=2 of the 3 strong-setup alphas (slots
        # CONVICTION_SLOTS) confirmed the trade direction at entry AND it closes in profit (day net up).
        self._conviction_bonus = float(getattr(self.cfg, "conviction_bonus", 0.0))
        self._conviction_cap = float(CONVICTION_ALIGN_CAP)
        # v1.10.0 HUGGING-PRESSURE reward (operator's heavy momentum agent): per-step bonus for RIDING a >=2-TF
        # shifted-SMA hug (continuation), and a heavier MISS-PENALTY for sitting out a CLEAN one on an
        # INDEX/METAL. Both gated to NOT hard-force when exhaustion/extension/decay conflict (momentum block).
        self._hug_bonus = float(getattr(self.cfg, "hug_pressure_bonus", 0.0))
        self._hug_miss_pen = float(getattr(self.cfg, "hug_miss_penalty", 0.0))
        self._HUG_ZERO = np.zeros(C.OBS_BLOCK_HUG_PRESSURE, dtype=np.float32)   # muted hug obs after the daily goal
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
        # v1.10.0: which symbols are an INDEX or METAL (the hug miss-penalty only applies to these momentum
        # instruments, per the operator). Inferred from the symbol (SPECS override, else roots).
        self._is_index_metal = {s: 1.0 if A.asset_class(s) in ("index", "metal") else 0.0 for s in self.symbols}
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
        self._day_progress_hwm = 0.0                           # high-water mark of progress toward +2.5% today (seek reward)
        self._day_had_exposure = False                         # did the bot hold ANY position today? (anti-hide)
        self._daily_target_reached = False                     # v1.10.0: hit +2.5% today? -> mute hug penalty + obs
        self._daily_pass_streak = 0                            # consecutive WON days (ended >= +2.5%) -> consistency
        self._days_won = 0                                     # cumulative WON days this episode (v1.8.0 consistency obs)
        self._entry_agreed = {}                                # per-open: did the entry agree with >=50% firing alphas?
        self._entry_alpha_dir = {}                             # per-open: the alpha consensus direction at entry
        self._init_trade_risk_state()                          # v1.7.0 per-symbol open-trade risk state
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

    def restart_account(self) -> None:
        """START OVER: reset the pot + positions to a FRESH challenge attempt, KEEPING the current bar/cursor.
        Used after a breach so the bot doesn't trade a DEAD account -- it restarts and keeps going (operator
        2026-06-28: 'if it fails one day, restart the following day'). The day-by-day report calls this on a
        breach so each attempt is a clean slate; training's rollout does the equivalent on-device."""
        self.acc = AccountState(starting_balance=self.cfg.starting_balance)
        self.th = TradeHistory()
        self.position = {s: 0 for s in self.symbols}
        self.entry = {s: float(self.subs[s].close[self.t]) for s in self.symbols}
        self._reset_phase2()
        self._day_progress_hwm = 0.0
        self._day_had_exposure = False
        self._daily_target_reached = False
        self._daily_pass_streak = 0
        self._days_won = 0
        self._entry_agreed = {}
        self._entry_alpha_dir = {}
        self._init_trade_risk_state()                          # v1.7.0: fresh per-symbol trade-risk state
        self._mark()

    # ---- v1.7.0: per-symbol per-trade RISK state (trade-risk obs block + BB hard stop + risk sizing) ----
    def _init_trade_risk_state(self) -> None:
        """Reset every symbol's open-trade risk trackers (called on reset + restart_account)."""
        p = self.t
        self._trade_size = {s: float(self.subs[s].position_size) for s in self.symbols}
        self._entry_bar = {s: int(p) for s in self.symbols}
        self._entry_atr = {s: 0.0 for s in self.symbols}
        self._entry_stop_band = {s: 0.0 for s in self.symbols}
        self._mfe_atr = {s: 0.0 for s in self.symbols}
        self._mae_atr = {s: 0.0 for s in self.symbols}
        self._last_close_bar = {s: int(p) for s in self.symbols}
        self._last_close_dir = {s: 0 for s in self.symbols}
        self._last_exit_px = {s: float(self.subs[s].close[p]) for s in self.symbols}
        self._entry_band_long = {s: False for s in self.symbols}    # band-stack at entry (enter-bonus)
        self._entry_band_short = {s: False for s in self.symbols}
        self._entry_reentry = {s: 0 for s in self.symbols}          # entered as a with-trend re-entry?
        self._entry_confirms = {s: 0 for s in self.symbols}         # # of the 3 strong-setup alphas confirming at entry

    @staticmethod
    def _atr_at(sub, t: int) -> float:
        v = float(sub._atr1m[t])
        return v if np.isfinite(v) else 0.0

    def _update_excursions(self) -> None:
        """At the (advanced) bar, update each open trade's max favorable / adverse excursion (ATR units)."""
        i = self.t
        for s in self.symbols:
            if self.position[s] != 0 and self._entry_atr[s] > 0.0:
                sub = self.subs[s]
                exc = self.position[s] * (sub.close[i] - self.entry[s]) / self._entry_atr[s]
                self._mfe_atr[s] = max(self._mfe_atr[s], exc)
                self._mae_atr[s] = max(self._mae_atr[s], -exc)

    def _trade_risk_block(self, sym, sub) -> np.ndarray:
        """The 14-float v1.7.0 trade-risk block for the CURRENT symbol's open position (or re-entry context)."""
        i = self.t
        return TR.build(
            pos=self.position[sym], entry_px=self.entry[sym], price=float(sub.close[i]),
            trade_size=self._trade_size[sym], equity=self.acc.equity,
            entry_atr=self._entry_atr[sym], atr_now=self._atr_at(sub, i),
            entry_stop_band=self._entry_stop_band[sym],
            bars_held=(i - self._entry_bar[sym]), mfe_atr=self._mfe_atr[sym], mae_atr=self._mae_atr[sym],
            bars_since_close=(i - self._last_close_bar[sym]), last_dir=self._last_close_dir[sym],
            last_exit_px=self._last_exit_px[sym],
            bb200_1m_up=float(sub._bb200_1m_up[i]), bb200_1m_lo=float(sub._bb200_1m_lo[i]),
            bb200_5m_up=float(sub._bb200_5m_up[i]), bb200_5m_lo=float(sub._bb200_5m_lo[i]),
            bb10_1m_up=float(sub.bb10_1m_up[i]), bb10_1m_lo=float(sub.bb10_1m_lo[i]),
            bb10_5m_up=float(sub.bb10_5m_up[i]), bb10_5m_lo=float(sub.bb10_5m_lo[i]))

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
                ts = self._trade_size[s]                                        # v1.7.0 actual (risk-based) size
                realized = self.position[s] * (sub.close[self.t] - self.entry[s]) * ts
                realized -= sub.cost_frac * sub.close[self.t] * ts             # exit cost
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
                c += sub.cost_frac * sub.close[self.t] * self._trade_size[s]
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
                unreal += self.position[s] * (sub.close[self.t] - self.entry[s]) * self._trade_size[s]
        self.acc.equity = self.acc.balance + unreal
        self.acc.mark_equity(self.acc.equity)

    def _apply_bb_hard_stop(self) -> None:
        """v1.7.0 BB HARD STOP: auto-close any open position whose price has crossed the 1m BB(10,1) opposite
        band (long below the lower band, short above the upper). Protective -> BYPASSES the day-lock; realizes
        at the current bar via record_close (with the trade's actual size), tallies, and records re-entry
        context (a mechanical exit, like the two-phase flatten -> no shaping bonus). OFF unless bb_stop_enabled."""
        if not self._bb_stop_on:
            return
        t = self.t
        for s in self.symbols:
            p = self.position[s]
            if p == 0:
                continue
            sub = self.subs[s]
            lo = float(sub.bb10_1m_lo[t]); up = float(sub.bb10_1m_up[t])
            px = float(sub.close[t])
            hit = ((p > 0 and np.isfinite(lo) and px < lo) or (p < 0 and np.isfinite(up) and px > up))
            if not hit:
                continue
            ts = self._trade_size[s]
            realized = p * (px - self.entry[s]) * ts - sub.cost_frac * px * ts
            self.th.record_close(self.acc, realized, bar_index=t)
            self._last_close_bar[s] = int(t)
            self._last_close_dir[s] = int(p)
            self._last_exit_px[s] = px
            self.position[s] = 0
        self._mark()

    def _set_aggregates(self):
        pos = self.position
        S = max(1, len(self.symbols))
        self.acc.open_positions = sum(1 for p in pos.values() if p != 0)
        self.acc.net_exposure = float(np.clip(sum(pos.values()) / S, -1.0, 1.0))
        self.acc.gross_exposure = float(np.clip(sum(abs(p) for p in pos.values()) / S, 0.0, 1.0))
        self.acc.unrealized_pnl = self.acc.equity - self.acc.balance
        big = max(self.symbols, key=lambda s: abs(pos[s]))
        self.acc.largest_position_dir = int(np.sign(pos[big]))

    # ---- observation (499; per-symbol blocks + SHARED account/portfolio blocks) ----
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
            "ohlc": sub.ohlc_matrix[i],   # v1.6.0: per-symbol raw O/H/L/C per timeframe
            "trade_risk": self._trade_risk_block(sym, sub),   # v1.7.0: current symbol's open-trade risk state
            "consistency": WL.consistency_features(            # v1.8.0: multi-day FTMO standing (won-day streak)
                self._daily_pass_streak, self._days_won, self._days_elapsed),
            "momentum": sub.momentum_matrix[i],                # v1.9.0: 9 momentum-perception scores (static)
            # v1.10.0: 15 hugging-pressure scores. ZEROED once today's +2.5% goal is reached (operator: stop
            # observing the agent after the goal so it doesn't tempt a give-back). Re-armed next day.
            "hug_pressure": (self._HUG_ZERO if self._daily_target_reached else sub.hug_pressure_matrix[i]),
            "bb_interactions": sub.bb_interactions_matrix[i],   # v1.11.0: 12 dual-BB interaction scores (static)
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
        # 5m CCI open-gate: block a NEW directional open when the 5m is FLAT (both CCIs in +/-50). Chop filter.
        if self._open_gate and sub.open_gate_blocked[t] and target != 0 and target != self.position[sym]:
            target = 0
        alpha_shaping = 0.0
        if target != self.position[sym]:
            if self.position[sym] != 0:                        # realize the closing leg into the POT
                old_dir = self.position[sym]
                ts_old = self._trade_size[sym]                          # v1.7.0 actual (risk-based) size
                realized = old_dir * (sub.close[t] - self.entry[sym]) * ts_old
                realized -= sub.cost_frac * sub.close[t] * ts_old
                self.th.record_close(self.acc, realized, bar_index=t)   # single P&L truth
                self._last_close_bar[sym] = int(t)                      # v1.7.0 re-entry context
                self._last_close_dir[sym] = int(old_dir)
                self._last_exit_px[sym] = float(sub.close[t])
                # SHAPING bonuses (paid only on a profitable close with the DAY net up; capped at the trade PnL):
                #   alpha USE+BEAT (cfg.alpha_*), band-stack ENTER bonus (cfg.band_stack_bonus), re-entry nudge.
                day0 = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
                if realized > 0.0 and self.acc.balance > day0:
                    pnl_frac = realized / self.cfg.starting_balance
                    bonus = 0.0
                    if self._alpha_on:
                        if self._entry_agreed.get(sym, False):              # USE the alphas: agreed >=50% and won
                            bonus += self._alpha_agree
                        move = sub.close[t] - self.entry[sym]               # BEAT the alphas: out-earned a follow
                        alpha_gross = self._entry_alpha_dir.get(sym, 0) * move * ts_old
                        bot_gross = old_dir * move * ts_old
                        if bot_gross > alpha_gross:
                            bonus += min(self._alpha_beat, (bot_gross - alpha_gross) / self.cfg.starting_balance)
                    # BAND-STACK enter bonus: entered with price stacked above (long) / below (short) BB200 & BB10
                    # on BOTH 1m and 5m, and the trade closed in profit with the day net up (operator 2026-06-29).
                    if self._band_bonus > 0.0 and ((old_dir > 0 and self._entry_band_long.get(sym, False))
                                                   or (old_dir < 0 and self._entry_band_short.get(sym, False))):
                        bonus += self._band_bonus
                    # RE-ENTRY nudge: this was a with-trend re-entry that paid off.
                    if self._reentry_bonus > 0.0 and self._entry_reentry.get(sym, 0):
                        bonus += self._reentry_bonus
                    # CONVICTION (selectivity): reward SCALES with how many signals aligned with this entry
                    # (the greatest one-directional consensus pays most), but ONLY if the bot traded WITH the
                    # majority (entry agreed). Capped at CONVICTION_ALIGN_CAP aligned signals.
                    if self._conviction_bonus > 0.0 and self._entry_agreed.get(sym, False):
                        strength = min(self._entry_confirms.get(sym, 0), self._conviction_cap) / self._conviction_cap
                        bonus += self._conviction_bonus * strength
                    if bonus > 0.0:
                        alpha_shaping += min(bonus, pnl_frac)              # CAP: bonus can never exceed the trade PnL
                self._entry_agreed.pop(sym, None); self._entry_alpha_dir.pop(sym, None)
            self.position[sym] = target
            if target != 0:
                self.entry[sym] = float(sub.close[t])
                # v1.7.0 RISK-BASED sizing (gated): size so a BB(10,1) stop-out loses ~risk_per_trade_pct% of
                # the pot, capped at the configured size. OFF (risk_per_trade_pct None) -> trade_size == base.
                base = float(sub.position_size)
                sb = float(sub.bb10_1m_lo[t] if target > 0 else sub.bb10_1m_up[t])   # the hard-stop band at entry
                ts = base
                if self._risk_pct is not None and np.isfinite(sb):
                    dist = abs(float(sub.close[t]) - sb)
                    if dist > 1e-12:
                        risk_dollars = self.cfg.starting_balance * (self._risk_pct / 100.0)
                        ts = min(base, risk_dollars / dist)
                self._trade_size[sym] = ts
                ecost = sub.cost_frac * sub.close[t] * ts
                self.acc.balance -= ecost
                self.acc.daily_realized_pnl -= ecost
                self.acc.episode_realized_pnl -= ecost
                # record the alpha consensus AT ENTRY (for the close-time bonuses) + penalise fighting it
                agree, disagree, net_dir = self._alpha_consensus(sub, t, target)
                self._entry_agreed[sym] = (agree >= 0.5)
                self._entry_alpha_dir[sym] = net_dir
                if self._alpha_on and disagree >= 0.5:                      # opened AGAINST >=50% firing alphas
                    alpha_shaping -= self._alpha_against
                # v1.7.0 per-trade risk state: entry bar/ATR, the BB(10,1) hard-stop band, reset MFE/MAE, the
                # band-stack-at-entry flags (enter bonus), and whether this is a with-trend RE-ENTRY.
                self._entry_bar[sym] = int(t)
                self._entry_atr[sym] = self._atr_at(sub, t)
                self._entry_stop_band[sym] = sb
                self._mfe_atr[sym] = 0.0
                self._mae_atr[sym] = 0.0
                self._entry_band_long[sym] = TR.band_stack_long(
                    sub.close[t], sub._bb200_1m_up[t], sub.bb10_1m_up[t], sub._bb200_5m_up[t], sub.bb10_5m_up[t])
                self._entry_band_short[sym] = TR.band_stack_short(
                    sub.close[t], sub._bb200_1m_lo[t], sub.bb10_1m_lo[t], sub._bb200_5m_lo[t], sub.bb10_5m_lo[t])
                ld = self._last_close_dir[sym]
                self._entry_reentry[sym] = int(
                    ld != 0 and target == ld and ld * (float(sub.close[t]) - self._last_exit_px[sym]) > 0.0)
                # SELECTIVITY: count ALL firing alphas pointing the SAME way as this entry (the "amount of
                # signals in one direction"). Empty/inactive slots are 0 != +/-1, so this counts only firing,
                # aligned alphas. The close-time conviction reward scales with this (capped), gated on agreeing.
                self._entry_confirms[sym] = int((sub.alpha_matrix[t] == target).sum())

        # advance the cursor: next symbol; when it wraps, advance the bar
        self.j += 1
        bar_advanced = False
        if self.j >= len(self.symbols):
            self.j = 0
            self.t = min(t + 1, self.T - 1)
            bar_advanced = True
        self._mark()
        self._update_excursions()                              # v1.7.0: track each open trade's MFE/MAE (ATR)
        if any(self.position[s] != 0 for s in self.symbols):   # anti-hide: was the bot exposed today?
            self._day_had_exposure = True
        # =================================================================================================
        # REWARD MODEL — the COMPLETE reward logic for the portfolio bot (keep this comment in sync with the
        # code; full write-up in docs/UPDATE_LOG.md). Everything is a fraction of the INITIAL balance.
        #
        #   (1) BASE (every step):  (equity_now - equity_before)/starting_balance * reward_scale
        #       The shared pot's per-step equity change. Transaction costs are EMBEDDED (entry/exit costs
        #       reduce equity), so this already rewards real, fee-net money. This is ~99% of the signal.
        #
        #   (2) ALPHA-SHAPING (this step; `alpha_shaping`, computed above; ON by default, cfg.alpha_reward_enabled).
        #       DELIBERATE departure from "reward = equity only" (still true for single-symbol TradingEnv).
        #       Every bonus is CAPPED at the trade's own PnL and only paid on a profitable close with the DAY net up:
        #         - AGAINST  (at OPEN):  open vs >=50% of FIRING alphas        -> - alpha_against (0.001)
        #         - USE      (at CLOSE): agreed w/ >=50% & closed in profit     -> + alpha_agree   (0.001)
        #         - BEAT     (at CLOSE): out-earned following the consensus      -> + alpha_beat    (0.002 = 2x, so a
        #                                                                            divergent WIN isn't cancelled by AGAINST)
        #
        #   (2b) SEEK-THE-TARGET (this step; ON by default, cfg.target_seek_weight): a DENSE reward for NEW
        #        progress toward today's +2.5% target (high-water-mark, can't be farmed by churning). Makes
        #        the gradient point AT the target so the bot SEEKS profit instead of "hiding" to dodge the
        #        breach cliff. Total <= seek_weight (0.10) per won day.
        #
        #   (2c) DRAWDOWN-PROXIMITY (this step; ON by default, cfg.dd_proximity_coef): a DENSE quadratic penalty
        #        -coef*(dd/wall)^2 as equity falls from its peak toward the 4% trailing wall -> nearing the wall
        #        is no longer "free" until the cliff; the gradient says "ease off as you approach" (coef 0.02).
        #
        #   (3) ON BAR-ADVANCE (every len(symbols) steps, below):
        #         - PER-DAY at midnight: a "WON day" = the day ENDS >= +2.5% of initial (AFTER any give-back).
        #                        won  -> + day_pass_reward (0.025)   failed -> - day_fail_penalty (0.025)
        #         - ANTI-HIDE: a day FLAT the whole day -> - idle_day_penalty (0.02)  (hiding is no longer free)
        #         - CONSISTENCY: 4 WON days IN A ROW                     -> + pass_bonus (1.0)   [rarely fires until
        #                        the window is long enough to contain 4 days — Batch A, staged]
        #         - BREACH: 4% trailing / 5% daily / 10% total (LIVE pot) -> - breach_penalty (0.2) + episode ENDS
        #         - +10% PASS: eval -> ENDS + pass_bonus; training(continue_after_pass) -> keep going (streak rewards it)
        #
        #   RAILS (behaviour, NOT reward): two-phase banks +2.5% NET-of-fees -> 1% leash (phase2_continue) or day-lock.
        #
        #   HORIZON: gamma=0.9995 (~a full trading day) so the bot plans toward the midnight +2.5% target and the
        #   wall PROACTIVELY (was 0.997 ~1.4h -> walls were only avoided reactively). DD-proximity is now LIVE (2c).
        # =================================================================================================
        reward = float((self.acc.equity - eq_before) / self.cfg.starting_balance) * self.reward_scale
        reward += alpha_shaping            # alpha-shaping (0 unless enabled AND an alpha consensus event fired)
        # SEEK-THE-TARGET (dense): reward NEW progress toward today's +2.5% target (high-water-mark so it
        # can't be farmed by churning). Makes "move toward the day's target" the gradient -> the bot SEEKS
        # profit instead of hiding. Capped at the target (progress in [0,1]); total <= seek_w per won day.
        if self._seek_w > 0.0:
            d0s = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
            target_amt = self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance
            day_progress = 0.0 if target_amt <= 0 else min(1.0, max(0.0, (self.acc.equity - d0s) / target_amt))
            reward += self._seek_w * max(0.0, day_progress - self._day_progress_hwm)
            self._day_progress_hwm = max(self._day_progress_hwm, day_progress)
        # DRAWDOWN-PROXIMITY (dense): a GRADUAL penalty that grows as equity nears the trailing wall, so the
        # bot plans AWAY from the wall instead of only feeling it at the breach cliff. penalty = coef*(dd/wall)^2.
        if self._dd_coef > 0.0:
            peak = self.acc.episode_peak_equity or self.acc.starting_balance
            wall = self.cfg.trailing_drawdown_pct / 100.0
            dd_frac = max(0.0, (peak - self.acc.equity) / peak) if peak else 0.0
            reward -= self._dd_coef * (min(dd_frac / wall, 1.0) if wall > 0 else 0.0) ** 2
        # DAILY GOAL REACHED (operator 2026-06-30): once the pot hits +2.5% today, MUTE the hug agent entirely
        # (no penalty, no bonus) AND zero its observation (in _obs) so it can't tempt a give-back -> "stop
        # observing it so it doesn't mess up the thinking." Persists until midnight; re-armed next day.
        d0h = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
        if (self.acc.equity - d0h) >= (self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance):
            self._daily_target_reached = True
        # HUGGING-PRESSURE (v1.10.0; operator's heavy momentum agent): for the CURRENT symbol at bar t, reward
        # RIDING a >=3-TF shifted-SMA hug (dominant side, continuation) and PENALISE sitting out a CLEAN one on
        # an INDEX/METAL. "Clean" = ALL 3 TFs agree AND momentum is NOT exhausted / extended-in-direction /
        # decaying (the carve-out that prevents hard-forcing). MUTED once today's +2.5% goal is reached.
        if (self._hug_bonus > 0.0 or self._hug_miss_pen > 0.0) and not self._daily_target_reached:
            hp = sub.hug_pressure_matrix[t]; mm = sub.momentum_matrix[t]
            dom = float(hp[_HUG_DOM])                                  # +1 / -1 / 0  (net hug side across TFs)
            cont3 = float(hp[_HUG_CONT3])                            # 1.0 if ALL 3 TFs agree
            loc = float(mm[_MOM_LOC])
            conflict = (mm[_MOM_EXH] > HUG_EXH_THR or mm[_MOM_DEC] > HUG_DECAY_THR
                        or (abs(loc) > HUG_LOC_THR and (loc > 0) == (dom > 0) and dom != 0.0))
            clean = (cont3 > 0.5) and (dom != 0.0) and (not conflict)
            if clean:
                aligned = (self.position[sym] == dom)
                if aligned and self._hug_bonus > 0.0:
                    reward += self._hug_bonus                          # RIDE the continuation (heavy prior)
                elif (not aligned) and self._hug_miss_pen > 0.0 and self._is_index_metal[sym]:
                    reward -= self._hug_miss_pen                       # sat out a clean index/metal momentum

        terminated = False
        truncated = False
        daily_target_hit = False
        if bar_advanced:
            if self._dates[self.t] != self._cur_date:          # midnight -> SCORE the day that just ENDED, then reset
                # A "WON day" = the day ENDS at >= +2.5% of INITIAL (measured here at midnight, AFTER any
                # give-back -- so banking +2.5% then leaking it back to +1.5% counts as a FAIL). A won day pays
                # day_pass_reward PLUS an ESCALATING streak bonus (every ADDITIONAL consecutive won day is worth
                # more, capped); a failed day penalises and RESETS the streak (operator 2026-06-29).
                d0 = self.acc.day_start_balance if self.acc.day_start_balance is not None else self.acc.starting_balance
                day_end_gain = self.acc.equity - d0
                won = day_end_gain >= (self.cfg.daily_target_pct / 100.0 * self.acc.starting_balance)
                if won:
                    self._daily_pass_streak += 1
                    self._days_won += 1                            # v1.8.0 consistency obs (cumulative won days)
                    streak_bonus = self._streak_bonus * min(self._daily_pass_streak - 1, self._streak_bonus_cap)
                    reward += self._day_pass_reward + streak_bonus   # won day + ESCALATING streak bonus
                else:
                    reward -= self._day_fail_penalty           # consistency: penalise a FAILED day
                    self._daily_pass_streak = 0
                # ANTI-HIDE: a day the bot was FLAT the WHOLE day (never held a position) is penalised, so
                # "hiding" (staying out to dodge the breach penalty) is no longer free -> it must ENGAGE the
                # market. (Exposure, not close-count: opening + holding across midnight is NOT hiding.)
                if self._idle_pen > 0.0 and not self._day_had_exposure:
                    reward -= self._idle_pen
                self.acc.reset_day()
                self._reset_phase2()                           # new day -> clear the day-lock / phase-2
                self._day_progress_hwm = 0.0                   # new day -> reset the seek high-water mark
                self._daily_target_reached = False             # new day -> hug penalty/obs un-muted again
                # re-seed exposure for the NEW day from the carried position (holding across midnight = exposed)
                self._day_had_exposure = any(self.position[s] != 0 for s in self.symbols)
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
            # v1.7.0 BB HARD STOP (protective): auto-close any position past the 1m BB(10,1) opposite band.
            # Runs BEFORE two-phase banking so a stopped trade isn't double-handled. Equity-neutral (closes at
            # the marked bar) -> does not retroactively change this step's reward. OFF unless bb_stop_enabled.
            if not terminated:
                self._apply_bb_hard_stop()
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
                    # NOTE: "won day" is scored at MIDNIGHT on the day's ENDING equity (not here), so a bank
                    # that later gives back under the 1% leash correctly counts as a FAIL.
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
