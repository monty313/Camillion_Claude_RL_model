# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Build, ONCE on the host, the things the JAX env indexes instead of
#      recomputing: (1) the (T, 513) STATIC observation tensor with the 9 per-bar
#      blocks placed at their exact contract indices (dynamic slots zeroed), and
#      (2) the per-bar + per-symbol scalar arrays the branchless step needs
#      (close, is_new_day, minute_of_day, ref_move, recent-context ranges, ...).
#      The static blocks are taken straight from a PRECOMPUTED CPU TradingEnv, so
#      459/499 obs floats are BYTE-IDENTICAL to the CPU env -> near-zero parity risk.
#      This tensor is SHARED read-only across all parallel envs (the "build once +
#      share" plan in config/constants.py / project memory).
# WHERE jax_tpu/jax_static_features.py
# HOW   Vectorized numpy assembly (no per-bar Python in the hot path). Block ranges
#       are derived from C.OBS_BLOCK_ORDER so they never drift from the contract.
# DEPENDS_ON: numpy, config.constants, src.env.trading_env (TradingEnv), signal helpers
# USED_BY: jax_tpu/jax_env.py (StaticData), jax_tpu/jax_trainer.py, the notebook, tests
# CHANGE_NOTES(IRAC): I: rewriting indicators/alphas in jnp is huge + risky; they're
#   already precomputed on the host. R: CLAUDE.md #3 (no TA-Lib/pandas in step) +
#   the shared-table scaling plan. A: lift the 9 static blocks from the CPU env into a
#   (T,513) tensor, share it; recompute only the 40 dynamic floats in jnp. C: exact
#   static obs + a tiny per-env state -> thousands of envs fit on a TPU.
# =====================================================================
"""Host builder for the shared (T,513) static obs tensor + per-bar/per-symbol scalars."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from config import constants as C
from src.signals.signal_summary import summarize  # for cross-check only


# --- contract block ranges (start index of each block in the 499 vector) ---
def _block_ranges() -> dict[str, tuple[int, int]]:
    ranges, start = {}, 0
    for name, size in C.OBS_BLOCK_ORDER:
        ranges[name] = (start, start + size)
        start += size
    assert start == C.OBS_TOTAL_SIZE, (start, C.OBS_TOTAL_SIZE)
    return ranges


BLOCK_RANGES = _block_ranges()
# blocks the JAX env recomputes every step (everything else is static, placed here). v1.7.0 adds
# trade_risk: it depends on the EVOLVING open-position state, so it is recomputed, not placed.
DYNAMIC_BLOCKS = ("account_daily", "account_episode", "portfolio", "sizing", "recent_context", "trade_risk")
DYNAMIC_SLICES = {b: BLOCK_RANGES[b] for b in DYNAMIC_BLOCKS}


@dataclass
class StaticData:
    """Everything the JAX env needs that is FIXED per (symbol, bar). Arrays are numpy
    here; jax_env converts to device arrays once and shares them read-only."""
    static_obs: np.ndarray      # (T, 513) float32 — static blocks placed, dynamic = 0
    close: np.ndarray           # (T,) float64 — price for P&L + mark
    is_new_day: np.ndarray      # (T,) float32 — 1.0 if bar t starts a new day vs t-1
    open_gate_blocked: np.ndarray  # (T,) float32 — 1.0 where a new directional open is gated (5m CCI neutral)
    minute_of_day: np.ndarray   # (T,) int32 — UTC minute (NY-bonus windows)
    ref_move: np.ndarray        # (T,) float32 — sizing block input
    week_avg: np.ndarray        # (T,) float32 — recent_context input
    prev_day: np.ndarray        # (T,) float32
    prev2_day: np.ndarray       # (T,) float32
    today_sofar: np.ndarray     # (T,) float32
    # v1.7.0 trade-risk per-bar references: clean 1m ATR(14) (NaN->0, matches _atr_at) + BB200(dev1)
    # and BB10(dev1) upper/lower on 1m & 5m (raw; NaN warmup -> band-stack flag False, like the CPU block)
    atr1m: np.ndarray           # (T,) float32 (NaN->0)
    bb200_1m_up: np.ndarray     # (T,) float32
    bb200_1m_lo: np.ndarray     # (T,) float32
    bb200_5m_up: np.ndarray     # (T,) float32
    bb200_5m_lo: np.ndarray     # (T,) float32
    bb10_1m_up: np.ndarray      # (T,) float32
    bb10_1m_lo: np.ndarray      # (T,) float32
    bb10_5m_up: np.ndarray      # (T,) float32
    bb10_5m_lo: np.ndarray      # (T,) float32
    # per-symbol scalars
    T: int
    starting_balance: float
    position_size: float
    value_per_point: float
    cost_frac: float
    typical_range: float        # 0.0 means "None" (CPU uses week_avg fallback)
    is_index: float             # 1.0/0.0 — NY-session index bonus applies
    warmup: int


def _last5_matrix(net_series: np.ndarray, lags: int) -> np.ndarray:
    """(T, lags) where col k = net[t-k], zeros before the start. Vectorized last5_from_series."""
    T = net_series.shape[0]
    out = np.zeros((T, lags), dtype=np.float32)
    for k in range(lags):
        if k == 0:
            out[:, 0] = net_series
        else:
            out[k:, k] = net_series[:-k]
    return out


def _alpha_summary_matrix(alpha_matrix: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """(T,4) [buy%, sell%, active%, net%] — vectorized signal_summary.summarize."""
    av = np.asarray(alpha_matrix, dtype=np.float32)
    om = np.asarray(occupancy, dtype=np.float32).ravel()
    assigned = float(om.sum())
    buy = (av == 1.0).sum(axis=1).astype(np.float32)
    sell = (av == -1.0).sum(axis=1).astype(np.float32)
    active = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_pct = np.where(active > 0, buy / active, 0.0)
        sell_pct = np.where(active > 0, sell / active, 0.0)
        active_pct = np.where(assigned > 0, active / assigned, 0.0) if assigned > 0 else np.zeros_like(active)
        net_pct = np.where(active > 0, (buy - sell) / active, 0.0)
    return np.stack([buy_pct, sell_pct, active_pct, net_pct], axis=1).astype(np.float32)


def build_static_data(env) -> StaticData:
    """Lift the 9 STATIC obs blocks + per-bar scalars out of a PRECOMPUTED CPU TradingEnv.

    `env` is a src.env.trading_env.TradingEnv whose __init__ already ran _precompute (or loaded
    a feature cache), so alpha_matrix / occupancy / net_signal / sig_acc / time_feats /
    cross_asset_matrix / streak_matrix / ref_move / recent-context arrays are all present.
    """
    T = int(env.T)
    so = np.zeros((T, C.OBS_TOTAL_SIZE), dtype=np.float32)

    def place(name, arr):
        a, b = BLOCK_RANGES[name]
        arr = np.asarray(arr, dtype=np.float32)
        assert arr.shape == (T, b - a), f"{name}: {arr.shape} != {(T, b - a)}"
        so[:, a:b] = arr

    # static, per-bar blocks (identical to TradingEnv._obs's static parts)
    place("indicators", env.ind)
    place("alpha_values", env.alpha_matrix)
    place("alpha_mask", np.tile(np.asarray(env.occupancy, np.float32), (T, 1)))
    place("alpha_summary", _alpha_summary_matrix(env.alpha_matrix, env.occupancy))
    place("signal_memory", _last5_matrix(np.asarray(env.net_signal, np.float32), C.SIGNAL_MEMORY_LAGS))
    place("signal_accuracy", env.sig_acc)
    place("time", env.time_feats)
    place("alpha_streak", np.minimum(env.streak_matrix, C.ALPHA_STREAK_CAP) / float(C.ALPHA_STREAK_CAP))
    place("cross_asset", env.cross_asset_matrix)
    place("ohlc", env.ohlc_matrix)   # v1.6.0: raw O/H/L/C per timeframe (static; zeros if env has no aux)
    # dynamic blocks (account_daily/episode, portfolio, sizing, recent_context) stay 0 here.

    # sanitize EXACTLY like the CPU builder (np.nan_to_num on the whole vector)
    so = np.nan_to_num(so, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # per-bar scalars
    dates = np.asarray(env._dates)
    is_new_day = np.zeros(T, dtype=np.float32)
    is_new_day[1:] = (dates[1:] != dates[:-1]).astype(np.float32)

    typ = env._typical_range
    return StaticData(
        static_obs=so,
        close=np.asarray(env.close, dtype=np.float64).ravel(),
        is_new_day=is_new_day,
        open_gate_blocked=np.asarray(env.open_gate_blocked, dtype=np.float32).ravel(),
        minute_of_day=np.asarray(env._minute_of_day, dtype=np.int32).ravel(),
        ref_move=np.asarray(env.ref_move, dtype=np.float32).ravel(),
        week_avg=np.asarray(env._week_avg, dtype=np.float32).ravel(),
        prev_day=np.asarray(env._prev_day, dtype=np.float32).ravel(),
        prev2_day=np.asarray(env._prev2_day, dtype=np.float32).ravel(),
        today_sofar=np.asarray(env._today_sofar, dtype=np.float32).ravel(),
        # v1.7.0 trade-risk per-bar refs (atr1m NaN->0 to match TradingEnv._atr_at; bands raw)
        atr1m=np.nan_to_num(np.asarray(env._atr1m, np.float32).ravel(), nan=0.0),
        bb200_1m_up=np.asarray(env._bb200_1m_up, np.float32).ravel(),
        bb200_1m_lo=np.asarray(env._bb200_1m_lo, np.float32).ravel(),
        bb200_5m_up=np.asarray(env._bb200_5m_up, np.float32).ravel(),
        bb200_5m_lo=np.asarray(env._bb200_5m_lo, np.float32).ravel(),
        bb10_1m_up=np.asarray(env.bb10_1m_up, np.float32).ravel(),
        bb10_1m_lo=np.asarray(env.bb10_1m_lo, np.float32).ravel(),
        bb10_5m_up=np.asarray(env.bb10_5m_up, np.float32).ravel(),
        bb10_5m_lo=np.asarray(env.bb10_5m_lo, np.float32).ravel(),
        T=T,
        starting_balance=float(env.cfg.starting_balance),
        position_size=float(env.position_size),
        value_per_point=float(env.value_per_point),
        cost_frac=float(env.cost_frac),
        typical_range=float(typ) if typ else 0.0,
        is_index=1.0 if getattr(env, "_is_index", False) else 0.0,
        warmup=int(env.warmup),
    )


@dataclass
class PortfolioStaticData:
    """Per-symbol stacked static data for the SHARED-POT portfolio env. Leading axis = symbol.

    The per-symbol obs blocks are identical to the single-symbol env (so static_obs[j] is reused for
    symbol j); only the SHARED account/portfolio/sizing/recent-context blocks are recomputed from the
    pot. alpha_matrix + occupancy are kept raw so the portfolio's alpha-shaping reward can recompute
    the firing-alpha consensus on-device."""
    symbols: tuple
    static_obs: np.ndarray      # (N, T, 499)
    close: np.ndarray           # (N, T) float64
    is_new_day: np.ndarray      # (T,)   float32 (shared clock — symbols are time-aligned)
    ref_move: np.ndarray        # (N, T)
    week_avg: np.ndarray        # (N, T)
    prev_day: np.ndarray        # (N, T)
    prev2_day: np.ndarray       # (N, T)
    today_sofar: np.ndarray     # (N, T)
    alpha_matrix: np.ndarray    # (N, T, 64) — +1/-1/0 per alpha slot
    occupancy: np.ndarray       # (N, 64)    — 1 assigned / 0 empty
    position_size: np.ndarray   # (N,)
    value_per_point: np.ndarray # (N,)
    typical_range: np.ndarray   # (N,) 0.0 == None
    cost_frac: np.ndarray       # (N,)
    # v1.7.0 trade-risk per-bar refs (per symbol)
    atr1m: np.ndarray           # (N, T)
    bb200_1m_up: np.ndarray     # (N, T)
    bb200_1m_lo: np.ndarray     # (N, T)
    bb200_5m_up: np.ndarray     # (N, T)
    bb200_5m_lo: np.ndarray     # (N, T)
    bb10_1m_up: np.ndarray      # (N, T)
    bb10_1m_lo: np.ndarray      # (N, T)
    bb10_5m_up: np.ndarray      # (N, T)
    bb10_5m_lo: np.ndarray      # (N, T)
    N: int
    T: int
    starting_balance: float
    warmup: int


def build_portfolio_static(subs: dict) -> PortfolioStaticData:
    """Stack per-symbol StaticData (+ raw alpha tables) from a dict {symbol: TradingEnv} (time-aligned).

    `subs` is exactly what src.env.portfolio_env.build_portfolio_subs returns. All symbols must share
    the same length T and timestamps (use portfolio_env.align_symbol_data first)."""
    symbols = list(subs)
    sds = {s: build_static_data(subs[s]) for s in symbols}
    T = sds[symbols[0]].T
    for s in symbols:
        assert sds[s].T == T, f"{s}: T={sds[s].T} != {T} (symbols must be time-aligned)"
    st = lambda attr: np.stack([getattr(sds[s], attr) for s in symbols], axis=0)
    return PortfolioStaticData(
        symbols=tuple(symbols),
        static_obs=st("static_obs"),
        close=st("close"),
        is_new_day=sds[symbols[0]].is_new_day,            # shared clock
        ref_move=st("ref_move"), week_avg=st("week_avg"), prev_day=st("prev_day"),
        prev2_day=st("prev2_day"), today_sofar=st("today_sofar"),
        alpha_matrix=np.stack([np.asarray(subs[s].alpha_matrix, np.float32) for s in symbols], axis=0),
        occupancy=np.stack([np.asarray(subs[s].occupancy, np.float32) for s in symbols], axis=0),
        position_size=np.array([sds[s].position_size for s in symbols], np.float64),
        value_per_point=np.array([sds[s].value_per_point for s in symbols], np.float64),
        typical_range=np.array([sds[s].typical_range for s in symbols], np.float32),
        cost_frac=np.array([sds[s].cost_frac for s in symbols], np.float64),
        atr1m=st("atr1m"), bb200_1m_up=st("bb200_1m_up"), bb200_1m_lo=st("bb200_1m_lo"),
        bb200_5m_up=st("bb200_5m_up"), bb200_5m_lo=st("bb200_5m_lo"),
        bb10_1m_up=st("bb10_1m_up"), bb10_1m_lo=st("bb10_1m_lo"),
        bb10_5m_up=st("bb10_5m_up"), bb10_5m_lo=st("bb10_5m_lo"),
        N=len(symbols), T=T,
        starting_balance=float(sds[symbols[0]].starting_balance),
        warmup=int(sds[symbols[0]].warmup),
    )
