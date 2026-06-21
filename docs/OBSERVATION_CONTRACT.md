# OBSERVATION CONTRACT — v1.1.0  (367 float32)

> Defined in `config/constants.py` (sizes) and `src/observation/observation_contract.py`
> (names). Built by `src/observation/builder.py`. **Changing this = a deliberate
> version bump.** Adding *strategies* does NOT change it (that is the whole point).

## Block order (concatenated in this exact order)

| # | Block | Size | Meaning |
|---|-------|------|---------|
| 1 | `indicators` | 200 | Raw values, 5 TFs × 40. NOT normalized. (real in Phase 1: numpy/pandas, TA-Lib optional.) |
| 2 | `alpha_values` | 64 | Strategy outputs `+1` buy / `-1` sell / `0` inactive. Fixed slots. |
| 3 | `alpha_mask` | 64 | Occupancy: `1` strategy assigned, `0` empty slot. Distinguishes empty vs inactive. |
| 4 | `alpha_summary` | 4 | `buy_pct, sell_pct, active_pct, net_signal_pct`. |
| 5 | `signal_memory` | 5 | Net signal balance, last 5 bars (`lag_0`=current … `lag_4`). |
| 6 | `signal_accuracy` | 2 | Rolling 1-bar / 3-bar accuracy (no look-ahead). |
| 7 | `account_daily` | 7 | win%, pnl%, dd_used%, target_progress%, risk_remaining%, trades%, streak%. |
| 8 | `account_episode` | 7 | win%, pnl%, dd_used%, target_progress%, pass_progress%, risk_remaining%, streak%. |
| 9 | `time` | 6 | tod sin/cos, dow sin/cos, London flag, New York flag. |
| 10 | `portfolio` | 8 | open%, net exposure, gross exposure, unrealized pnl%, avg age%, largest dir, equity ratio, balance ratio. |

**Total = 367.**

## Per-timeframe indicator block (40), repeated for 1m, 5m, 30m, 4h, 1d
- **SMA (6):** p1/s0, p2/s1, p3/s2, p4/s3, p50/s0, p200/s0.
- **CCI (4):** cci30 raw + cci30 SMA(2)-shift-4; cci100 raw + cci100 SMA(2)-shift-4.
- **RSI (4):** rsi4 raw + rsi4 SMA(2)-shift-2; rsi14 raw + rsi14 SMA(2)-shift-2.
- **ATR (2):** atr14 raw + atr14 SMA(2)-shift-4 (volatility; shift = compare vs ~4 bars ago).
- **Bollinger (24):** periods {20,200} × devs {0.5,1,2,4} × {upper,middle,lower}.

## Why percentages (scale-stability)
Summaries and account features are **fractions**, not raw counts. So (a) adding
strategies never confuses the bot, and (b) changing the FTMO/FREE target, trailing
drawdown amount, or trailing on/off **needs no retrain** — a fraction like
"target_progress" or "risk_remaining" means the same thing to the policy regardless
of the absolute limit. See `FTMO_AND_FREE_MODE.md`.

## Hybrid design (raw + alphas)
The bot sees **both** raw indicators (so it can trade on its own even when no alpha
fires) **and** the alpha signals (suggestions it can weight or ignore). `0` is an
*inactive strategy*, never a HOLD action. The RL action space {HOLD, BUY, SELL, CLOSE}
is separate from alpha outputs.

## Ablation
Blocks are contiguous and named (`observation_contract.BLOCK_SLICES`). The
`alpha_mask` block (64, near-static within an episode) is the first ablation
candidate if training is noisy. Removing/adding any block is a **new contract version**.

## Strategy vs Alpha vs Policy (LOCKED 2026-06-21)
- **strategy** = the *internal* logic that generates a signal (any indicators /
  rules). Lives in `src/strategies/`. **The policy never sees a strategy's
  internals.**
- **alpha** = a strategy's *exposed output* for its slot, exactly one of:
  `+1` active buy · `-1` active sell · `0` assigned-but-inactive ·
  *empty slot* = no alpha assigned (occupancy mask = 0). These four are
  distinct and **none equals the action-space HOLD** (alpha `0` lives in
  alpha-space; ACTION `HOLD` lives in action-space — they merely share the int 0).
- **policy** = the RL agent. It sees: raw indicators (market reality) + alpha
  outputs (expert suggestions) + alpha-context + account/FTMO/portfolio. It is
  rewarded ONLY on the real objective, never for "matching an alpha."
- Code naming: `AlphaRegistry`, `collect_alphas()`, `alpha_values/alpha_mask/
  alpha_summary`; example logic classes are `...Strategy`.

## Per-alpha reliability — design decision (anti-bloat)
Per-alpha 1/3/10-bar accuracy is tracked in **diagnostics / Policy Doctor**, NOT
the observation (3 × 64 slots = 192 mostly-empty features would wreck training
speed + scale-stability). The observation stays scale-stable as alphas are added.
A compact aggregate reliability block (≈4 values: reliability-weighted net, mean,
best, dispersion) may be added later as a deliberate v1.2.0 bump **only if
diagnostics prove it helps** — evidence-driven, not speculative.
