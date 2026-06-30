# OBSERVATION CONTRACT — v1.10.0  (541 float32)

> Defined in `config/constants.py` (sizes) and `src/observation/observation_contract.py`
> (names). Built by `src/observation/builder.py`. **Changing this = a deliberate
> version bump.** Adding *strategies* does NOT change it (that is the whole point).
>
> **Version history:** v1.1.0 (367) → v1.2.0 (451: indicators 200→220 + 64-wide `alpha_streak`)
> → v1.3.0 (461: 10-float `sizing`) → v1.4.0 (471: 10-float `cross_asset`) → v1.5.0 (479: 8-float
> `recent_context`) → v1.6.0 (499: 20-float raw `ohlc` block) → v1.7.0 (513: 14-float `trade_risk` block)
> → v1.8.0 (517): appended the 4-float `consistency` block (the bot's multi-day FTMO standing / won-day
> streak, so it can VALUE and PROTECT the streak with the stretched discount horizon)
> → v1.9.0 (526): appended the 9-float `momentum` block — momentum-PERCEPTION scores (one per the
> operator's momentum decision tree), so the policy learns the PRINCIPLE of momentum, not hard-coded CCI
> rules (see `JORDAN_PRINCIPLES.md`)
> → **v1.10.0 (541): appended the 15-float `hug_pressure` block — the operator's "Shifted SMA Hugging
> Pressure" agent (heavy): shifted-SMA(2)-on-High/Low envelope hug across 5m/15m/1h (15m & 1h resampled).**
> Appended blocks leave indices 0..(prev-1) unchanged (a v1.9.0 policy must retrain — shape changed).

## Block order (concatenated in this exact order)

| # | Block | Size | Meaning |
|---|-------|------|---------|
| 1 | `indicators` | 220 | Raw values, 5 TFs × 44 (v1.2.0 added 4 extras/TF). NOT normalized. |
| 2 | `alpha_values` | 64 | Strategy outputs `+1` buy / `-1` sell / `0` inactive. Fixed slots. |
| 3 | `alpha_mask` | 64 | Occupancy: `1` strategy assigned, `0` empty slot. Distinguishes empty vs inactive. |
| 4 | `alpha_summary` | 4 | `buy_pct, sell_pct, active_pct, net_signal_pct`. |
| 5 | `signal_memory` | 5 | Net signal balance, last 5 bars (`lag_0`=current … `lag_4`). |
| 6 | `signal_accuracy` | 2 | Rolling 1-bar / 3-bar accuracy (no look-ahead). |
| 7 | `account_daily` | 7 | win%, pnl%, dd_used%, target_progress%, risk_remaining%, trades%, streak%. |
| 8 | `account_episode` | 7 | win%, pnl%, dd_used%, target_progress%, pass_progress%, risk_remaining%, streak%. |
| 9 | `time` | 6 | tod sin/cos, dow sin/cos, London flag, New York flag. |
| 10 | `portfolio` | 8 | open%, net exposure, gross exposure, unrealized pnl%, avg age%, largest dir, equity ratio, balance ratio. |
| 11 | `alpha_streak` | 64 | Per-alpha consecutive-signal streak, normalized by `ALPHA_STREAK_CAP` (v1.2.0). |
| 12 | `sizing` | 10 | **v1.3.0**, all fractions of INITIAL balance: 6-rung what-if lot ladder (0.01/0.1/0.5/1/2/4 → account-% a typical move is worth), `daily_target_remaining`, `dd_room`, `active_lots_norm`, `active_move_value`. OBSERVATION ONLY — sizing is not an action yet. |
| 13 | `cross_asset` | 10 | **v1.4.0**: asset-class one-hot (`pair/index/metal/energy/crypto`, classifier covers the full FTMO broker) + ATR-normalized movement (`move_in_atr`, `atr_pct_price`, `atr_regime` — **scale-free**, comparable across all instruments) + sessions (`asian`, `london_ny_overlap`). Lets one policy generalize across the universe. |
| 14 | `recent_context` | 8 | **v1.5.0**: recent daily movement RELATIVE to the symbol's average — `week_avg_range_vs_typical`, `prev_day/prev2_day/today_range_vs_week` — + TIME-to-pass pace: `days_elapsed_norm`, `episode_return_so_far`, `pace_vs_2_5pct_plan` (0.5 = on plan), `challenge_target_remaining`. |
| 15 | `ohlc` | 20 | **v1.6.0**: RAW Open/High/Low/Close of the last CLOSED bar on each of the 5 TFs (`{tf}__open/high/low/close`, TF-major). The policy finally sees High/Low/Open, not just close. NOT normalized (like block 1). Built at cache time from the resampled bars (the env carries only `close`) and threaded in via `src/data/aux_features.py`; leak-free (last-closed-bar). |
| 16 | `trade_risk` | 14 | **v1.7.0**: the CURRENT symbol's OPEN-TRADE risk state, so the policy can MANAGE the trade and learn to RE-ENTER a winner. `tr_in_trade`, `tr_direction`, unrealized P&L in ATR units (`tr_unrealized_pnl_atr`) and as % of the pot (`tr_unrealized_pnl_pct`), distance to the 2×-ATR(14) SOFT stop (`tr_dist_to_soft_stop_2atr`) and to the 1m BB(10,1) opposite-band HARD stop (`tr_dist_to_hard_stop_bb`, 0→1), `tr_bars_held_norm`, max favorable / adverse excursion in ATR (`tr_max_favorable_atr`, `tr_max_adverse_atr`), re-entry context (`tr_bars_since_last_close`, `tr_last_trade_dir`, `tr_price_vs_last_exit_atr`), and the band-stack flags `tr_band_stack_long` / `tr_band_stack_short` (price above/below BB200(dev1) AND BB10(dev1) on BOTH 1m & 5m). DYNAMIC (recomputed each step from the position state). Shared builder `src/observation/trade_risk.py` (jnp twin in `jax_tpu/jax_obs_blocks.trade_risk_features`). |

| 17 | `consistency` | 4 | **v1.8.0**: the bot's MULTI-DAY FTMO standing, so it can VALUE and PROTECT the won-day STREAK (paired with the stretched discount horizon, gamma 0.9999). `won_day_streak_norm` (current consecutive won days / 40), `days_won_norm` (cumulative won days / 40), `won_day_rate` (days won / days elapsed = consistency %), `days_into_journey_norm` (days elapsed / 40). DYNAMIC. The shared-pot `PortfolioEnv` fills these from its midnight day-scoring; the single-symbol env (no streak logic) emits zeros (only `days_into_journey` is real). Shared builder `src/account/win_loss_features.consistency_features` (jnp twin in `jax_obs_blocks.consistency_features`). |

| 18 | `momentum` | 9 | **v1.9.0**: momentum-PERCEPTION scores — the operator's "momentum" decomposed into learnable sub-problems (a decision tree), one per-bar SCORE each, so the policy CONSUMES momentum (it doesn't reinvent it from raw indicators) and learns the PRINCIPLE, not a hard CCI threshold (see `JORDAN_PRINCIPLES.md`). `mom_tradeability` (is momentum present? graded 0..1), `mom_bias` (higher-TF direction, ±1), `mom_alignment` (do 5m/30m/4h agree? ±1), `mom_strength` (graded |CCI| ladder 0..1), `mom_exhaustion` (blow-off risk past ~160), `mom_location` (extension vs pullback = 5m band position, ±1), `mom_structure` (position in the recent range → near a breakout, 0..1), `mom_persistence` (recent follow-through, 0..1), `mom_decay` (momentum dying = CCI rolled back from its peak, 0..1). **STATIC** (market-only, per-bar) → computed once in `src/observation/momentum_scores.py`, lifted byte-identical into the JAX env (auto parity; no jnp twin). Levels/windows are TUNABLE — the policy learns the weighting. |

| 19 | `hug_pressure` | 15 | **v1.10.0**: the operator's "Shifted SMA Hugging Pressure" agent (HEAVY). Across **5m / 15m / 1h**, a fast `SMA(2)` of High and Low **shifted forward 1 bar** forms an envelope; price that keeps HUGGING one side (never touching the opposite band) for consecutive bars = sustained directional pressure; **2+ TFs agreeing = strong continuation**. Per-TF (×3): `hug_{tf}_side` (±1 bull/bear hug), `hug_{tf}_count` (consecutive no-opposite-touch bars / 20), `hug_{tf}_respect` (current bar still on side). Aggregate (×6): `hug_agree_bull`, `hug_agree_bear`, `hug_net_pressure` (signed, count-weighted), `hug_strength` (0..1), `hug_continuation_2plus` (≥2 TFs agree), `hug_dominant_side` (±1). **15m & 1h are a RESAMPLED side-channel from the 1m High/Low** (NOT new full obs timeframes — engine still runs 1m/5m/30m/4h/1d). **STATIC** (market-only, per-bar) → computed once in `src/observation/hug_pressure.py` from the OHLC aux, lifted byte-identical into the JAX env (auto parity; no jnp twin). The HEAVY action prior + indices/metals miss-penalty live in the reward (`portfolio_env`), not here. |

**Total = 541.**

> **The trade-risk block is where the BB hard stop + risk-based sizing + band-stack/re-entry bonuses live.**
> The 1m+5m BB(10,1) bands it needs are NOT in the 220-indicator cache (BB periods there are 20 & 200 only);
> they are precomputed leak-free from `close`+`time` in `TradingEnv._precompute` (`compute_bb10_bands`) — no
> cache-format change. The BEHAVIOURS (auto-close at the 1m BB(10,1) band, size each entry so a stop-out
> loses ~`risk_per_trade_pct`% of the pot, PnL-capped band-stack/re-entry CLOSE bonuses) live in
> `PortfolioEnv` and default OFF; the training path turns them on. The block itself (and the per-trade
> MFE/MAE state) is always present and parity-clean (CPU ↔ JAX bar-for-bar).

> **OHLC is built where the OHLC exists.** The env only carries `close`, so the 20-float raw OHLC
> block (and the ADX-DI alpha side-channel) are precomputed at cache time into a `{symbol}_aux.npy`
> array (`build_aligned_aux`), trimmed alongside the indicators by `align_symbol_data`, and handed to
> the env as `aux=`. The aux array is **NOT** itself in the observation — only its OHLC half is (block
> 15); its DI half feeds the two ADX-DI alphas (slots 16/17). The aux content is part of the feature-cache
> fingerprint, so a no-OHLC cache is never loaded as an OHLC one.

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
