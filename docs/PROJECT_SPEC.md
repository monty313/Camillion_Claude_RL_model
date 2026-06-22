# PROJECT SPEC — Camillion RL Trading Bot

> Mission: Build an RL bot that passes the FTMO challenge consistently.
> Principle: Safety first, profit second.

---

## 1. THE FTMO-PASS MISSION

**Goal:** Achieve a high, stable pass rate on FTMO-style prop firm challenges.

**Success metric:** Pass rate (percentage of simulated challenges that complete without breach and reach profit target), NOT maximum PnL.

**FTMO challenge parameters (from `config/variables.py`):**
- Daily drawdown limit: 5%
- Total drawdown limit: 10%
- Trailing drawdown wall: 4% (toggleable)
- Daily target: 2.5% (toggleable)
- Two-phase option: Hit +2.5% → auto-flat → fresh 1% trailing wall

**Runtime-editable:** Target, trailing-DD amount, and trailing on/off can be changed at runtime WITHOUT retraining (observation uses percentages, not absolute values).

---

## 2. THE 367 OBSERVATION CONTRACT (SACRED)

**Contract version:** v1.1.0

**Total size:** 367 float32 values

**Design principle:** The observation shape is LOCKED. Adding strategies fills pre-allocated slots; the shape NEVER changes. Old models continue to work with new alphas.

### 2.1 Block Structure

| Block | Size | Purpose |
|-------|------|---------|
| indicators | 200 | Raw multi-timeframe indicators (5 TFs × 40 each) — NOT normalized |
| alpha_values | 64 | Strategy outputs: +1 (buy), -1 (sell), 0 (inactive) |
| alpha_mask | 64 | Occupancy: 1 (strategy assigned), 0 (empty slot) |
| alpha_summary | 4 | buy%, sell%, active%, net% — scale-stable percentages |
| signal_memory | 5 | Net signal balance for last 5 bars (lag_0 to lag_4) |
| signal_accuracy | 2 | Rolling 1-bar and 3-bar directional accuracy (leak-free) |
| account_daily | 7 | Daily win%, pnl%, dd_used%, target_progress%, risk_remaining%, trades%, streak% |
| account_episode | 7 | Episode-level versions of daily features |
| time | 6 | tod_sin, tod_cos, dow_sin, dow_cos, session_london, session_newyork |
| portfolio | 8 | open_positions_pct, net_exposure_signed, gross_exposure_pct, unrealized_pnl_pct, avg_position_age_pct, largest_position_dir, equity_ratio, balance_ratio |

### 2.2 Source Files

| Component | File |
|-----------|------|
| Block sizes | `config/constants.py` |
| Feature names | `src/observation/observation_contract.py` |
| Builder | `src/observation/builder.py` |
| Documentation | `docs/OBSERVATION_CONTRACT.md` |

### 2.3 Current Audit Finding (Unresolved)

**Verified source-tree inconsistency:**
- `config/constants.py` defines `OBS_TOTAL_SIZE = 357` and `N_INDICATORS_TOTAL = 190`
- Actual indicator count is 200 (5 TFs × 40 per TF)
- Expected total is 367
- Tests assert against 367

**Status:** High-confidence stale-constants bug. Constants file was not updated when ATR was added. Requires verification of actual runtime shape.

---

## 3. ALPHA PHILOSOPHY

**Core principle:** Strategies emit suggestions; the policy decides actions.

### 3.1 Terminology (Locked 2026-06-21)

| Term | Definition |
|------|------------|
| **Strategy** | Internal logic that generates a signal. Lives in `src/strategies/`. The policy NEVER sees strategy internals. |
| **Alpha** | A strategy's exposed output for its slot: +1 (active buy), -1 (active sell), 0 (assigned but inactive). Empty slot = no alpha assigned (mask = 0). |
| **Policy** | The RL agent. Sees raw indicators + alpha outputs + alpha-context + account/FTMO/portfolio. Rewarded ONLY on real objective, never for "matching an alpha." |

### 3.2 Alpha States (Distinct from Actions)

| Alpha State | Meaning | NOT the same as |
|-------------|---------|-----------------|
| +1 | Active buy signal | ACTION_BUY (policy decision) |
| -1 | Active sell signal | ACTION_SELL (policy decision) |
| 0 | Assigned but no current setup | ACTION_HOLD (policy decision) |
| Empty (mask=0) | No strategy in this slot | Not a signal at all |

### 3.3 Alpha vs Policy Relationship

- Alphas are **suggestions**, not commands
- Policy can weight, ignore, or override any alpha
- Reward is based on **equity change only**, never on alpha-matching
- Diagnostics track alpha reliability separately (not in observation, not in reward)

---

## 4. GRAVITY ALPHA (SLOT 0)

**Status:** First real alpha, currently assigned to slot 0. All other 63 slots are empty.

### 4.1 What It Reads (RAW Values Only)

| Indicator Family | Columns | Notes |
|------------------|---------|-------|
| CCI | `cci30_raw`, `cci100_raw` | Both timeframes (30m, 4h) |
| RSI | `rsi4_raw`, `rsi14_raw` | Both timeframes |
| Bollinger Bands | 8 configs: periods {20, 200} × devs {0.5, 1, 2, 4} × {upper, middle, lower} | Both timeframes |
| SMA Fan | `sma_p1_s0`, `sma_p2_s1`, `sma_p3_s2`, `sma_p4_s3` | Both timeframes |

### 4.2 Voting Logic

**Per-timeframe majority vote:**
- Each detector votes: +1 (bullish), -1 (bearish), 0 (neutral/dead-zone)
- **Family mode (default):** CCI casts 1 vote, RSI casts 1 vote, BB casts 1 vote, SMA casts 1 vote
- No single family can outvote the other three
- **Flat mode (override):** Every detector votes (BB has 8 of 13 votes — heavier)

**Confluence requirement:**
- 30m AND 4h must agree on the same non-zero direction
- If either TF is neutral or they conflict, output is 0 (inactive)

### 4.3 Dead Zones

| Detector | Dead Zone | Output |
|----------|-----------|--------|
| CCI | Between -25 and +25 | 0 (neutral) |
| RSI | Between 45 and 55 | 0 (neutral) |
| BB position | Between -0.25 and +0.25 (near middle) | 0 (neutral) |
| SMA fan | No dead zone | Pure trend direction |

### 4.4 Source File

`src/strategies/gravity_30m_4h_alpha.py`

---

## 5. 5m OPEN GATE

**Purpose:** Risk filter that blocks new directional opens when 5m momentum is neutral.

### 5.1 Rule

Block new BUY/SELL opens if **EITHER**:
- `5m__cci30_raw` is between -50 and +50, **OR**
- `5m__cci100_raw` is between -50 and +50

### 5.2 Semantics

| Action | When Gate is Blocked |
|--------|---------------------|
| HOLD | ✓ Always allowed |
| CLOSE | ✓ Always allowed |
| BUY (new open) | ✗ Blocked |
| SELL (new open) | ✗ Blocked |
| Flip long→short | ✗ Blocked (just closes, no opposite open) |

**OR logic:** BOTH CCIs must be outside the neutral band for a new open to proceed.

### 5.3 Configuration

- Default: **OFF** (gate disabled)
- Enable: `open_gate=True` in TradingEnv constructor

### 5.4 Source Files

- Implementation: `src/env/trading_env.py` (not directly auditable)
- Test: `tests/test_open_gate.py`

---

## 6. REWARD STRUCTURE (NOT FINALIZED, APPROVAL-GATED)

### 6.1 Current Implementation

**Indirectly verified from tests/docs:**

```
reward = (equity_{t+1} - equity_t) / starting_balance - breach_penalty * 1[breached]
```

**Properties:**
- Real money made/lost this step, as fraction of starting balance
- Breach penalty subtracted if FTMO limit breached
- NO alpha term, NO accuracy term, NO signal term

### 6.2 Known Issues

| Issue | Finding | Source | Status |
|-------|---------|--------|--------|
| Reward scale | Default `position_size=100000.0` is realistic | `src/env/trading_env.py` signature | **RESOLVED** |
| Breach penalty dominance | breach_penalty=1.0 dominates per-step reward (~±0.004 at 100k notional) | `docs/READINESS_AUDIT.md` | Intentional |

**Historical note:** The original audit (READINESS_AUDIT.md) flagged `position_size=1.0` as producing near-zero reward signal. This was corrected before this audit — the default is now `100000.0`.

### 6.3 Not Verified (Security Restriction)

- Exact `breach_penalty` default value
- Exact `position_size` default value
- Exact equity calculation implementation

### 6.4 Status

**Reward is NOT finalized.** Requires:
1. Direct inspection of env reward code path
2. Decision on position_size default
3. Decision on breach_penalty value
4. Your approval before any training run

---

## 7. CURRENT PHASE: SINGLE-SYMBOL BASELINE

**Scope:** Train and validate on one symbol at a time (default: EURUSD).

**Rationale:** Establish baseline performance before multi-symbol portfolio training.

**Pipeline:**
1. Build cache for single symbol (`src/data/cache_builder.py`)
2. Train PPO on single-symbol env (`src/training/trainer.py`)
3. Evaluate via walk-forward validation (`src/training/walk_forward.py`)
4. Diagnose via Policy Doctor and Barbershop (`src/barbershop/`)

**Future phases:** Multi-symbol portfolio, multi-timeframe specialization, live MT5 integration.

---

## 8. WHAT IS ALREADY STRONG

| Component | Status | Evidence |
|-----------|--------|----------|
| Leak-free multi-TF cache | Verified | `tests/test_cache_no_leakage.py` |
| Fixed 64-slot alpha registry | Verified | `src/strategies/registry.py` |
| Percentage-based account features | Verified | `src/account/win_loss_features.py` |
| Runtime-editable FTMO rules | Verified | `config/ftmo_config.py`, `update_risk_settings()` |
| Alpha vs action separation | Verified | `CLAUDE.md`, `docs/OBSERVATION_CONTRACT.md` |
| Signal accuracy (leak-free) | Verified | `src/signals/signal_accuracy.py` |
| Policy Doctor diagnostics | Verified | `src/barbershop/policy_doctor.py` |
| Walk-forward validation | Verified | `src/training/walk_forward.py` |

---

## 9. DOCUMENT SCOPE

This spec captures:
- The FTMO-pass mission
- The 367 observation contract (sacred, but with unresolved constants bug documented)
- Alpha philosophy (suggestion, not command)
- Gravity alpha (slot 0, all other 63 empty)
- 5m open gate (OR logic on CCI)
- Reward status (NOT finalized, approval-gated)
- Current phase (single-symbol baseline)

**Unresolved issues are documented as audit findings, not baked in as final truth.**
