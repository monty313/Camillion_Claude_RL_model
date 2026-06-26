# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  Produce a live, HONEST snapshot of the running bot for the JARVIS bridge:
#      account, position, alphas, policy, perf -- read straight off a headless
#      TradingEnv + AccountState + PolicyIntrospector. Works with NO trained model
#      (an honest alpha-consensus fallback) so the cockpit lights up out of the box.
# WHERE src/jarvis/state_provider.py
# HOW  StateProvider holds an env + registry + optional policy callable. step()
#      advances read-only and tracks entry-bar (age) + per-day realized P&L
#      (day_history). snapshot() reads live state -> the dict build_state() shapes.
#      Directional handling is DEFENSIVE: uses registry.directional_mask() if the
#      dual-gate branch is present, else treats every occupied slot as directional
#      (so a future movement gate's 1 is never counted as a bullish vote).
# DEPENDS_ON: numpy, pandas, src/env/trading_env.py, src/strategies/*,
#             src/interpret/policy_introspector.py, src/jarvis/state_contract.py
# USED_BY: jarvis_bridge.py, tests
# CHANGE_NOTES(IRAC): I: the HUD needs real numbers but there may be no model/data
#   yet. R: HANDOFF "never fabricate" + read-only. A: headless provider with a
#   no-model net-signal fallback + provider-side age/day_history tracking + the
#   directional-mask fallback. C: an honest live feed (or a clearly-flagged
#   fallback) so JARVIS only ever advises on real system state.
# =====================================================================
"""StateProvider: a read-only live snapshot of the bot for the JARVIS /state bridge."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import constants as C

_DIR = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def directional_mask(registry) -> np.ndarray:
    """Boolean per-slot 'this slot votes in the directional consensus' mask.

    Prefers registry.directional_mask() (present once the dual-gate branch merges);
    falls back to per-slot DIRECTIONAL attr (default True) so non-directional gates
    are excluded the moment they exist, and every legacy alpha counts today.
    """
    fn = getattr(registry, "directional_mask", None)
    if callable(fn):
        return np.asarray(fn(), dtype=bool)
    out = np.zeros(registry.max_slots, dtype=bool)
    for i, s in enumerate(registry._slots):
        out[i] = (s is not None) and bool(getattr(s, "DIRECTIONAL", True))
    return out


class StateProvider:
    """Read-only snapshot source. NEVER places or modifies a trade."""

    def __init__(self, env, registry, policy=None):
        self.env = env
        self.registry = registry
        self.policy = policy                 # callable obs->(logits,value) or None
        self._entry_ptr = None
        self._day_history: list[float] = []
        self._dmask = directional_mask(registry)

    # ---- construction helpers --------------------------------------------------
    @classmethod
    def from_synthetic(cls, n: int = 600, seed: int = 0, symbol: str = "EURUSD"):
        """A tiny self-contained env (no trained model, no real data) so the bridge
        runs and is testable anywhere. Deterministic given seed."""
        from src.env.trading_env import TradingEnv
        from src.strategies.registry import AlphaRegistry
        from src.strategies.alpha_pack import register_all
        rng = np.random.default_rng(seed)
        close = (100.0 + np.cumsum(rng.standard_normal(n) * 0.05)).astype(np.float32)
        ind = np.zeros((n, C.N_INDICATORS_TOTAL), dtype=np.float32)
        idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
        time_ns = idx.values.astype("datetime64[ns]").astype(np.int64)
        reg = AlphaRegistry(); register_all(reg)
        env = TradingEnv(ind, close, time_ns, reg, warmup=min(200, n // 3),
                         symbol=symbol)
        env.reset()
        return cls(env, reg, policy=None)

    # ---- read-only stepping (drives the sim forward; never forces orders) ------
    def step(self, action: int | None = None):
        """Advance one bar. With a policy, act greedily; else HOLD. Tracks age + day P&L."""
        env = self.env
        if action is None:
            action = self._greedy_action() if self.policy is not None else C.ACTION_HOLD
        pos0, t0 = int(env.position), int(env.ptr)
        d0 = env._dates[env.ptr]
        pnl0 = float(env.acc.daily_realized_pnl)
        env.step(int(action))
        # entry-bar tracking for honest position age
        pos1 = int(env.position)
        if pos0 == 0 and pos1 != 0:
            self._entry_ptr = t0
        elif pos1 == 0:
            self._entry_ptr = None
        # per-day realized P&L ledger: capture the ending day's realized P&L at the boundary
        if env._dates[env.ptr] != d0:
            self._day_history.append(round(pnl0, 2))
        return self

    def _greedy_action(self) -> int:
        from src.interpret.policy_introspector import introspect
        rec = introspect(self.policy, self.env._obs(), ablate=False, bar_index=int(self.env.ptr))
        return int(rec.chosen_action)

    # ---- the snapshot --------------------------------------------------------
    def snapshot(self) -> dict:
        env, acc, cfg = self.env, self.env.acc, self.env.cfg
        ptr = int(env.ptr)

        # alphas (occupied slots only) + directional-only net signal
        sig = env.alpha_matrix[ptr]
        streak = env.streak_matrix[ptr]
        occ = env.occupancy
        alphas = []
        for i, s in enumerate(self.registry._slots):
            if s is None:
                continue
            alphas.append({"name": s.name, "signal": int(sig[i]),
                           "streak": int(streak[i]), "directional": bool(self._dmask[i])})
        keep = self._dmask & (np.asarray(occ) > 0)
        dir_idx = np.where(keep)[0]
        n_dir = int(max(1, dir_idx.size))
        net = float(sig[dir_idx].sum()) / n_dir if dir_idx.size else 0.0

        # policy (real introspection, or None -> contract uses the alpha fallback)
        policy_raw = None
        if self.policy is not None:
            from src.interpret.policy_introspector import introspect
            rec = introspect(self.policy, env._obs(), ablate=False, bar_index=ptr)
            policy_raw = {"action_probs": list(rec.action_probs), "value": rec.value,
                          "entropy": rec.entropy, "chosen_action": rec.chosen_action,
                          "chosen_action_name": rec.chosen_action_name}

        lots = 0.0 if env.position == 0 else float(env.position_size) / float(env.value_per_point or 1.0)
        age_known = (env.position == 0) or (self._entry_ptr is not None)
        age_min = 0 if env.position == 0 or self._entry_ptr is None else int(ptr - self._entry_ptr)

        return {
            "account": {
                "balance": float(acc.balance), "equity": float(acc.equity),
                "day_start_equity": float(acc.day_start_balance),
                "episode_start_equity": float(acc.episode_start_balance),
                "peak_equity": float(acc.episode_peak_equity),
            },
            "ftmo": {
                "daily_loss_limit_pct": float(getattr(cfg, "daily_drawdown_pct",
                                                      getattr(cfg, "max_daily_drawdown_pct", 5.0))),
                "max_drawdown_limit_pct": float(getattr(cfg, "max_total_drawdown_pct", 10.0)),
                "profit_target_pct": float(getattr(cfg, "profit_target_total_pct",
                                                   getattr(cfg, "daily_target_pct", 2.5))),
                "daily_target_pct": float(getattr(cfg, "daily_target_pct", 2.5)),
            },
            "position": {
                "dir": _DIR[int(env.position)], "symbol": env.symbol or "n/a",
                "lots": lots, "entry": float(env.entry_price) if env.position != 0 else 0.0,
                "price": float(env.close[ptr]), "age_min": age_min, "age_known": age_known,
            },
            "alphas": alphas,
            "policy_raw": policy_raw,
            "policy_extra": {},                       # advantage/regime/etc. -> contract defaults+flags
            "perf": {
                "win_rate_pct": float(acc.episode_win_rate * 100.0),
                "trades": int(acc.episode_trades),
                "consecutive_losses": int(acc.episode_consecutive_losses),
                "day_history": list(self._day_history),
            },
            "news": [],
            "human": {"overrides": 0, "panic_closes": 0, "discipline_pct": 0},
            "clock": pd.Timestamp(int(env.time_ns[ptr])).strftime("%H:%M:%S"),
            "net_signal": net,
            "n_directional": n_dir,
            "mode": str(getattr(cfg, "mode", "FTMO")),
            "model_attached": self.policy is not None,
        }
