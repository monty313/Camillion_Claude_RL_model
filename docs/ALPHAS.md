# ALPHAS.md — the complete alpha (strategy) catalogue

> **What this file is.** A single, plain-English reference for **every alpha in
> the repo** and **exactly how each one works** — the indicators it reads, the
> precise buy / sell / inactive conditions, the thresholds, and where it sits in
> the policy's view. Written for Monty (not a programmer): the *idea* is in
> prose, the *exact rule* is in the code box next to it.
>
> Source of truth = the code in [src/strategies/](../src/strategies/). If the
> code and this doc ever disagree, **the code wins** — re-read the file and fix
> this doc.

---

## 0. The 30-second version

- An **alpha** = a tiny "signal generator". Every bar it answers one question:
  *"Is there a setup right now?"* and outputs exactly one of three numbers:
  - **`+1`** = active **BUY** setup (go long)
  - **`-1`** = active **SELL** setup (go short)
  - **`0`** = **inactive** — *this alpha sees no setup right now*
- There are **16 production alphas** (canonical slots **0–15**) plus **3 example /
  teaching strategies** (not wired into training). They fall into **6 families**:

| Family | Slots | Count | One-line idea |
|---|---|---|---|
| **Gravity** | 0 | 1 | Many detectors vote on "distance from equilibrium"; fire only if **30m AND 4h agree** |
| **Regime Pulse** | 1–4 | 4 | Higher-TF regime (vs SMA20 & SMA200) + lower-TF trigger; **trend** = aligned, **pullback** = buy-the-dip |
| **CCI Surge Sentinel** | 5–8 | 4 | Raw CCI(30) + CCI(100) on two TFs; **trend** = all same sign, **pullback** = fast CCI dips against the trend |
| **SMA Stack Prophet** | 9–12 | 4 | Close vs a lagged SMA-of-high / SMA-of-low "envelope" on two TFs; **trend** vs **pullback** |
| **SMA Reversion Rally** | 13–14 | 2 | Higher-TF SMA30>SMA50 regime + lower-TF close **re-crossing** its SMA30 (rejoin the trend) |
| **ORB NY Breakout** | 15 | 1 | **Indices only** — break of the New-York opening range, filtered by the 30m SMA200; stateful per day |
| *Examples (not registered)* | — | 3 | SMA50/200 cross, RSI14 reversion, Bollinger breakout — teaching templates |

> **Crucial distinction — `alpha = 0` is NOT "hold".** Alpha `0` means *"this
> strategy has no setup"* (it lives in **alpha-space**). The RL policy's own
> `HOLD` action (also the integer `0`) lives in a **separate action-space**
> `{HOLD, BUY, SELL, CLOSE}`. They share the digit `0` but mean different things.
> A **third** state — an **empty slot** (no alpha assigned) — also reads `0`, and
> is told apart only by the **occupancy mask**. Keep the three distinct.

---

## 1. What an alpha is (the contract)

Every alpha is a subclass of `BaseStrategy` ([src/strategies/base.py](../src/strategies/base.py))
that implements **one** method:

```python
def compute_signal(self, ctx) -> int:   # must return +1, -1, or 0
```

- The public, **validated** entry point is `signal(ctx)`. It calls
  `compute_signal`, coerces the result to `int`, and **raises** if the value is
  not strictly in `(-1, 0, +1)`. The registry always calls `signal()` (never
  `compute_signal` directly), so a buggy alpha can never leak a `2` or a `0.5`
  into the system.
- `reset()` is an optional per-episode hook (default: does nothing). Only
  **stateful** alphas (currently just ORB) override it.
- The three legal outputs are named constants in
  [config/constants.py](../config/constants.py):
  `ALPHA_BUY = 1`, `ALPHA_SELL = -1`, `ALPHA_INACTIVE = 0`.

### What an alpha reads — the `MarketContext`

Each bar the environment hands every alpha a **read-only snapshot**
(`MarketContext`, [src/strategies/context.py](../src/strategies/context.py)):

| Field | Meaning |
|---|---|
| `close` | the current bar's close price (float; `NaN` if unknown) |
| `indicators` | a flat dict of cached indicator values, keyed `"{tf}__{column}"` |
| `bar_index` | index of the bar in the episode |
| `symbol` | the instrument name (e.g. `"US30"`, `"EURUSD"`) |
| `minute_of_day` | UTC minute-of-day of the bar close, `0..1439` (`-1` = unknown) |

Alphas never touch the dict directly. They call:

```python
ctx.ind("cci30_raw", "5m")   # -> the 5m raw CCI(30); returns NaN if missing
```

`ind(col, tf)` builds the key `f"{tf}__{col}"` and returns `NaN` for any missing
key. There is also a guard helper `MarketContext.is_valid(*vals)` that returns
`True` only if **every** value is finite — alphas use it (or a private `_v`
helper that maps `None`/`NaN` to "abstain") to **return `0` during warm-up**
instead of emitting a bogus signal.

> **Speed rule (CLAUDE.md #3).** No TA-Lib / MT5 / pandas runs inside the hot
> loop. The `MarketContext` only ever carries **already-cached float32 values**;
> alphas just compare numbers. That is why every alpha here is a handful of
> arithmetic/sign comparisons.

### How an alpha gets a slot — the registry

The `AlphaRegistry` ([src/strategies/registry.py](../src/strategies/registry.py))
is a **fixed-length list of 64 slots** (`MAX_STRATEGIES = 64`), each holding a
strategy or `None`. **The length never changes.** Registering a strategy just
flips one slot from `None` to a strategy.

- `register(strategy, slot=None)` → puts the strategy in the given slot, or the
  **first free** slot if `slot is None`, and returns the slot index.
- `collect_alphas(ctx)` → a length-64 float32 array of every slot's `+1/-1/0`
  output (empty slots read `0.0`).
- `occupancy_mask()` → a length-64 float32 array, `1.0` where a slot is filled,
  `0.0` where empty. **This is what tells "inactive" (assigned but `0`) apart
  from "empty" (no alpha).**

**Why 64 fixed slots and not a growing list?** Because the observation shape is
locked (CLAUDE.md rule #1). The design is deliberately **per-slot**: each alpha
owns one slot, which maps to **3 inputs** the policy sees (its value, its mask
bit, its signal-streak). So the policy can learn an **individual weight per
alpha**. Filling more slots only changes those slots' *values* — never the
*number* of inputs — so adding alphas up to 64 is free and a trained policy keeps
working. Empty slots cost only memory (a constant-`0` input gets no gradient).
Raising `MAX_STRATEGIES` past 64 is a deliberate **observation-contract bump**
(it resizes 3 obs blocks) and must follow the contract protocol.

### Canonical slot order — the alpha pack

[src/strategies/alpha_pack.py](../src/strategies/alpha_pack.py) defines
`register_all(registry)`, which calls each alpha's tiny `register(...)` helper in
a **fixed order**. Because each helper grabs "the next free slot", the **call
order = the slot order**:

| Slot | Alpha (`.name`) | Family | TF pair (gravity → trigger) |
|---:|---|---|---|
| 0 | `gravity_30m_4h_agree` | Gravity | 30m **&** 4h must agree |
| 1 | `regime_pulse_trend_5m_30m` | Regime Pulse | 30m → 5m |
| 2 | `regime_pulse_pullback_5m_30m` | Regime Pulse | 30m → 5m |
| 3 | `regime_pulse_trend_30m_4h` | Regime Pulse | 4h → 30m |
| 4 | `regime_pulse_pullback_30m_4h` | Regime Pulse | 4h → 30m |
| 5 | `cci_surge_trend_5m_30m` | CCI Surge | 30m → 5m |
| 6 | `cci_surge_pullback_5m_30m` | CCI Surge | 30m → 5m |
| 7 | `cci_surge_trend_30m_4h` | CCI Surge | 4h → 30m |
| 8 | `cci_surge_pullback_30m_4h` | CCI Surge | 4h → 30m |
| 9 | `sma_stack_trend_5m_30m` | SMA Stack | 30m → 5m |
| 10 | `sma_stack_pullback_5m_30m` | SMA Stack | 30m → 5m |
| 11 | `sma_stack_trend_30m_4h` | SMA Stack | 4h → 30m |
| 12 | `sma_stack_pullback_30m_4h` | SMA Stack | 4h → 30m |
| 13 | `sma_reversion_rally_5m_30m` | SMA Reversion | 30m → 5m |
| 14 | `sma_reversion_rally_30m_4h` | SMA Reversion | 4h → 30m |
| 15 | `orb_ny_breakout_indices` | ORB | indices only |
| 16–63 | *(empty)* | — | headroom |

> The `alpha_pack.py` docstring still says *"gravity + the 14-alpha pack
> (slots 1–14)"* — that wording predates the ORB alpha. The live count is **16**
> filled slots (0–15).

### How the policy actually sees the alphas

Each bar:

1. The env builds the `MarketContext`, then calls `collect_alphas(ctx)` (64
   values) and `occupancy_mask()` (64 bits).
2. [src/signals/signal_summary.py](../src/signals/signal_summary.py) collapses the
   64 slots into **4 scale-stable percentages** —
   `[buy_pct, sell_pct, active_pct, net_signal_pct]` (buy/sell/net are over the
   *firing* alphas; active is over *assigned* slots; net ∈ `[-1, +1]`).
3. [src/observation/builder.py](../src/observation/builder.py) drops these into
   named observation blocks: `alpha_values` (64), `alpha_mask` (64),
   `alpha_summary` (4), plus an `alpha_streak` block (64) of per-alpha signal
   streaks. `nan_to_num` guarantees the observation is never non-finite.

So the policy sees, per alpha: **its value, its mask bit, and its streak** — plus
4 aggregate percentages. It **never** sees a strategy's internal logic. Using
**percentages** (not raw counts) keeps the inputs stable as the alpha count grows.

---

## 2. The indicator vocabulary (how to read a column name)

Alphas read cached indicators. The cache holds **44 indicator columns per
timeframe × 5 timeframes = 220 raw values** (the first block of the observation).
Column keys are `"{tf}__{column}"` — note the **double** underscore between the
timeframe and the column (single underscores live *inside* a column name).

### Timeframes (order is part of the contract)

`TIMEFRAMES = ("1m", "5m", "30m", "4h", "1d")`

| TF | Role |
|---|---|
| `1m` | base / fastest bar |
| `5m` | short-term micro-trend (lower-TF *trigger* in the 5m/30m families) |
| `30m` | intraday swing (trigger in 30m/4h families; gravity in 5m/30m families) |
| `4h` | higher-TF trend / confluence (gravity in 30m/4h families) |
| `1d` | slowest macro / regime |

### Indicator families and naming

| Column pattern | What it is | Notes for reading it |
|---|---|---|
| `sma_p{P}_s{S}` | **Simple Moving Average**, period `P`, shifted `S` bars back | `sma_p1_s0` = SMA(1) of close = **just the last close** (used as "price"). `sma_p1_s1` = the **previous** close. `sma_p50_s0`, `sma_p200_s0`, `sma_p30_s0` = classic trend means. |
| `sma_p1_s0 … sma_p4_s3` | the **SMA "fan"** (lagged closes) | `sma_p1_s0`=now, `sma_p2_s1`≈1 bar ago, `sma_p3_s2`≈2 ago, `sma_p4_s3`≈3 ago. Compare newest vs the older ones to read very-short-term momentum. |
| `sma4_sh4_{high\|low}` | **SMA(4) of the bar HIGH / LOW, shifted 4 bars** | A lagged high/low "envelope". Used by the SMA Stack family. |
| `cci{P}_raw` | **raw Commodity Channel Index**, period `P` (30, 100) | Signed "distance from equilibrium": `>0` momentum up, `<0` down; `±100` = classically "stretched". |
| `cci{P}_sma2sh4` | smoothed+lagged copy of the CCI | Lets an alpha read CCI's slope (not used by the current alphas). |
| `rsi{P}_raw` | **raw RSI** (0–100), period `P` (4, 14) | `>50` bullish bias, `<50` bearish; `>70` overbought, `<30` oversold. |
| `rsi{P}_sma2sh2` | smoothed+lagged RSI | momentum direction (not used by current alphas). |
| `bb{P}_dev{X}_{upper\|middle\|lower}` | **Bollinger Bands**, period `P` (20, 200), `X` std-devs (0.5/1/2/4) | **`bb{P}_dev{X}_middle == SMA(P)`** — the centre line is just the moving average (same for every dev). Several alphas use the *middle* band purely as an SMA20 / SMA200. |
| `atr{P}_raw`, `atr{P}_sma2sh4` | **Average True Range** (volatility), period 14 | absolute bar range; for sizing / staying clear of FTMO walls. *Not* consumed by any current alpha, but part of the 44-column vocabulary. |

The `_raw` suffix marks the indicator's instantaneous value at the current bar
(the paired `_smaNshM` column is the same indicator smoothed and lagged). All
values are **raw TA-Lib units, never normalized**. Every column emits `NaN`
during its warm-up window (e.g. BB(200) until 200 bars), which is why alphas
guard with `is_valid` / `_v` and fall back to `0`.

> **Per-TF make-up of the 44:** 6 SMA + 4 CCI + 4 RSI + 2 ATR + 24 BB
> (2 periods × 4 devs × 3 bands) + 4 extras (`sma_p30_s0`, `sma_p1_s1`,
> `sma4_sh4_high`, `sma4_sh4_low`) = **44**. (CCI/RSI/BB/ATR auto-use TA-Lib if
> installed, else equivalent pandas math, so it runs on plain Colab.)

---

## 3. The 16 production alphas

Each writeup gives: the **idea** (prose), the **exact** BUY / SELL / INACTIVE
rule (code box), the indicators it reads, and the key thresholds. A recurring
naming convention across the multi-TF families:

- **`g…` = gravity-TF values** (the *higher* timeframe — the context / regime).
- **`s…` = signal-TF values** (the *lower* timeframe — the entry trigger).
- A `_v(ctx, col, tf)` helper reads a column and returns `None` for missing/`NaN`;
  if **any** required value is missing the alpha returns `0` (inactive).

---

### Slot 0 — `gravity_30m_4h_agree` (Gravity)

📄 [src/strategies/gravity_30m_4h_alpha.py](../src/strategies/gravity_30m_4h_alpha.py) · class `Gravity30m4hAlpha`

**Idea.** The flagship confluence alpha. On **each** of the 30m and 4h
timeframes it runs a panel of detectors that each vote on whether price is
"pulled" up or down from equilibrium — **raw CCI(30) & CCI(100)**, **raw RSI(4) &
RSI(14)**, **all 8 Bollinger configs** (periods 20 & 200 × devs 0.5/1/2/4), and
the **SMA fan**. It takes a per-timeframe **majority** vote, and fires **only if
30m and 4h agree** on the same non-zero direction.

**Detector rules (each emits +1 / −1 / 0, missing data = abstain):**

- **CCI:** `+1` if `cci > +25`, `−1` if `cci < −25`, else `0`. (`[−25, +25]` dead zone)
- **RSI:** `+1` if `rsi > 55`, `−1` if `rsi < 45`, else `0`. (`[45, 55]` dead zone)
- **Bollinger (per config):** position within the band
  `rel = (close − middle) / ((upper − lower) / 2)` — `0` at the middle, `±1` at
  the bands. `+1` if `rel > +0.25`, `−1` if `rel < −0.25`, else `0`. (price = `sma_p1_s0`)
- **SMA fan:** `rel = (s0 − mean(s1, s2, s3)) / |mean(...)|` over the fan
  `sma_p1_s0 … sma_p4_s3`. **No dead zone** → votes purely by the sign of `rel`.

**Combining the votes — `VOTE_MODE`:**

- **`"family"` (default, balanced):** CCI / RSI / BB / SMA each cast **one** vote
  (the majority within that group), then the four family-votes are
  majority-voted. No single family can outvote the other three.
- **`"flat"` (debug/ablation):** every one of the 13 detectors votes directly, so
  the 8 BB detectors dominate (8 of 13).

```text
a = tf_vote("30m");  b = tf_vote("4h")
BUY      (+1):  a == +1 AND b == +1      # both timeframes vote up
SELL     (-1):  a == -1 AND b == -1      # both timeframes vote down
INACTIVE  (0):  a == 0, OR b == 0, OR a != b   # flat on a TF, or the two disagree
```

| Threshold | Value | Meaning |
|---|---|---|
| CCI dead zone | `±25` | inside → CCI abstains |
| RSI dead zone | `[45, 55]` | inside → RSI abstains |
| BB dead zone | `±0.25` | band position inside → BB abstains |
| SMA dead zone | `None` | SMA fan votes purely by sign |
| `VOTE_MODE` | `"family"` | balanced 1-vote-per-family (vs `"flat"`) |

Stateless · all assets · registered. *(A `debug_votes()` method dumps every
detector's vote plus both flat & family results for inspection.)*

---

### Slots 1–4 — Regime Pulse (`RegimePulse…`)

📄 `regime_pulse_{trend,pullback}_{5m_30m,30m_4h}_alpha.py`

**Idea.** A higher-TF **regime** filter + a lower-TF **trigger**, both expressed
through moving averages. On each timeframe it reads the **close** (`sma_p1_s0`),
the **SMA20** (`bb20_dev1.0_middle`) and the **SMA200** (`bb200_dev1.0_middle`).
The **shared regime gate** is: the gravity TF's close is stacked on the **same
side of BOTH its SMA20 and SMA200** (above both = bull regime; below both = bear),
**and** the trigger's close agrees with the big picture via its own SMA200. The
two axes:

- **trend vs pullback** is decided **only** by the trigger vs its **own SMA20**:
  - **trend** → trigger also on the trend side of its SMA20 (momentum-aligned
    *continuation* entry).
  - **pullback** → trigger on the **opposite** side of its SMA20 while still on
    the trend side of its SMA200 (a *buy-the-dip / sell-the-rip* entry into the
    higher-TF regime).
- **TF pair** → 5m/30m (slots 1–2: 30m gravity, 5m trigger) or 30m/4h (slots 3–4:
  4h gravity, 30m trigger).

`gc/g20/g200` = gravity close/SMA20/SMA200; `sc/s20/s200` = trigger
close/SMA20/SMA200.

**Trend (slots 1 & 3):**
```text
BUY  (+1):  gc > g200 AND gc > g20 AND sc > s200 AND sc > s20
SELL (-1):  gc < g200 AND gc < g20 AND sc < s200 AND sc < s20
```
**Pullback (slots 2 & 4)** — *only the trigger-vs-SMA20 term flips:*
```text
BUY  (+1):  gc > g200 AND gc > g20 AND sc > s200 AND sc < s20   # trigger DIPPED below its SMA20
SELL (-1):  gc < g200 AND gc < g20 AND sc < s200 AND sc > s20   # trigger POPPED above its SMA20
INACTIVE (0): any value missing/NaN, or neither bull nor bear fully holds
```

| Slot | Name | Gravity TF | Trigger TF | Variant |
|---:|---|---|---|---|
| 1 | `regime_pulse_trend_5m_30m` | 30m | 5m | trend |
| 2 | `regime_pulse_pullback_5m_30m` | 30m | 5m | pullback |
| 3 | `regime_pulse_trend_30m_4h` | 4h | 30m | trend |
| 4 | `regime_pulse_pullback_30m_4h` | 4h | 30m | pullback |

Indicators: `sma_p1_s0`, `bb20_dev1.0_middle`, `bb200_dev1.0_middle` on both TFs.
No dead-band (strict `>`/`<`; exact equality → `0`). Stateless · all assets · registered.

---

### Slots 5–8 — CCI Surge Sentinel (`CciSurge…`)

📄 `cci_surge_{trend,pullback}_{5m_30m,30m_4h}_alpha.py`

**Idea.** Pure **sign agreement of raw CCI** across two periods on two
timeframes. It reads `cci30_raw` (fast) and `cci100_raw` (slow) on both the
gravity TF and the trigger TF. The gravity pair must agree (both `>0` or both
`<0`). Then:

- **trend** → the trigger's **fast** CCI agrees with gravity (all four same sign).
- **pullback** → the trigger's **fast** CCI (`s30`) goes the **opposite** way
  while the trigger's **slow** CCI (`s100`) still agrees — a short-term
  counter-move within the higher-TF trend (buy-the-dip / fade).

`g30/g100` = gravity fast/slow CCI; `s30/s100` = trigger fast/slow CCI.

**Trend (slots 5 & 7):**
```text
BUY  (+1):  g30 > 0 AND g100 > 0 AND s30 > 0 AND s100 > 0   # all four CCIs positive
SELL (-1):  g30 < 0 AND g100 < 0 AND s30 < 0 AND s100 < 0   # all four CCIs negative
```
**Pullback (slots 6 & 8)** — *only the fast-trigger sign `s30` flips:*
```text
BUY  (+1):  g30 > 0 AND g100 > 0 AND s30 < 0 AND s100 > 0   # gravity up, slow trigger up, fast trigger DIPPED negative
SELL (-1):  g30 < 0 AND g100 < 0 AND s30 > 0 AND s100 < 0   # gravity down, slow trigger down, fast trigger POPPED positive
INACTIVE (0): any value missing, mixed signs, or any CCI exactly 0
```

| Slot | Name | Gravity TF | Trigger TF | Variant |
|---:|---|---|---|---|
| 5 | `cci_surge_trend_5m_30m` | 30m | 5m | trend |
| 6 | `cci_surge_pullback_5m_30m` | 30m | 5m | pullback |
| 7 | `cci_surge_trend_30m_4h` | 4h | 30m | trend |
| 8 | `cci_surge_pullback_30m_4h` | 4h | 30m | pullback |

> **Only the *sign* of CCI is used.** A code comment mentions the `±100` "strong
> surge" band as inspectable colour, but the boolean logic **never** checks
> magnitude. Stateless · all assets · registered.

---

### Slots 9–12 — SMA Stack Prophet (`SmaStack…`)

📄 `sma_stack_{trend,pullback}_{5m_30m,30m_4h}_alpha.py`

**Idea.** On each timeframe, compare that TF's **close** (`sma_p1_s0`) to a
**lagged high/low envelope**: `sma4_sh4_high` (SMA(4) of the bar HIGH, shifted 4
bars) and `sma4_sh4_low` (SMA(4) of the bar LOW, shifted 4 bars). A TF is
**"stacked bullish"** when the close is **above BOTH** envelope lines, **"stacked
bearish"** when **below BOTH**, otherwise it's inside the envelope (no stack).

- **trend** → gravity and trigger stacked the **same** way.
- **pullback** → gravity stacked one way while the trigger has poked through to
  the **opposite** side of its envelope (counter-stack dip inside the context).

`gc/ghi/glo` = gravity close/high-env/low-env; `sc/shi/slo` = trigger
close/high-env/low-env.

**Trend (slots 9 & 11):**
```text
BUY  (+1):  gc > ghi AND gc > glo AND sc > shi AND sc > slo   # both TFs stacked bullish
SELL (-1):  gc < ghi AND gc < glo AND sc < shi AND sc < slo   # both TFs stacked bearish
```
**Pullback (slots 10 & 12)** — *only the trigger-stack direction flips:*
```text
BUY  (+1):  gc > ghi AND gc > glo AND sc < shi AND sc < slo   # gravity bullish, trigger DIPPED below its envelope
SELL (-1):  gc < ghi AND gc < glo AND sc > shi AND sc > slo   # gravity bearish, trigger POPPED above its envelope
INACTIVE (0): any value missing, or neither full pattern holds (e.g. a TF is inside its envelope)
```

| Slot | Name | Gravity TF | Trigger TF | Variant |
|---:|---|---|---|---|
| 9 | `sma_stack_trend_5m_30m` | 30m | 5m | trend |
| 10 | `sma_stack_pullback_5m_30m` | 30m | 5m | pullback |
| 11 | `sma_stack_trend_30m_4h` | 4h | 30m | trend |
| 12 | `sma_stack_pullback_30m_4h` | 4h | 30m | pullback |

Envelope params: `SMA_HL_PERIOD = 4`, `SMA_HL_SHIFT = 4`. Stateless · all assets · registered.

---

### Slots 13–14 — SMA Reversion Rally (`SmaReversionRally…`)

📄 `sma_reversion_rally_{5m_30m,30m_4h}_alpha.py`

**Idea.** Mean-reversion **into** a trend. The gravity TF sets the regime by the
**SMA30-vs-SMA50** spread (`sma_p30_s0` vs `sma_p50_s0`). The trigger TF watches
its **close re-cross its own SMA30** — price had pulled away from the mean and is
now *rallying back through it* to rejoin the prevailing trend. The cross is
measured over exactly **one closed bar** using the current close (`sma_p1_s0`)
and the previous close (`sma_p1_s1`). The two members differ **only** in the TF
pair — there is no separate trend/pullback split.

`g30/g50` = gravity SMA30/SMA50; `sc` = trigger close now; `s30` = trigger SMA30;
`sprev` = trigger close one bar ago.

```text
BUY  (+1):  g30 > g50 AND sprev <= s30 AND sc > s30   # up-regime + close crossed UP through its SMA30
SELL (-1):  g30 < g50 AND sprev >= s30 AND sc < s30   # down-regime + close crossed DOWN through its SMA30
INACTIVE (0): any value missing, or regime & cross don't both line up, or no fresh cross this bar
```

| Slot | Name | Gravity TF | Trigger TF |
|---:|---|---|---|
| 13 | `sma_reversion_rally_5m_30m` | 30m | 5m |
| 14 | `sma_reversion_rally_30m_4h` | 4h | 30m |

Note the boundary is **inclusive on the prior bar** (`<=` / `>=`) and **strict
this bar** (`>` / `<`), so it captures the bar where price *crosses* the SMA30.
Stateless · all assets · registered.

---

### Slot 15 — `orb_ny_breakout_indices` (ORB NY Breakout)

📄 [src/strategies/orb_ny_breakout_indices_alpha.py](../src/strategies/orb_ny_breakout_indices_alpha.py) · class `OrbNyBreakoutIndicesAlpha`

**Idea.** An **Opening-Range Breakout** around the **New-York open**, **indices
only**, and the one **stateful** alpha. During the 4 hours **before** the NY open
it accumulates an opening **range** (high/low, approximated by the close because
the env only carries close). During the 2-hour entry window after the open it
fires when price **breaks** that range **in agreement with the 30m SMA200 trend**.

**Timeline (UTC minute-of-day):**

| Window | Minutes | What happens |
|---|---|---|
| Opening range | `09:30–13:30` (`570 ≤ mod < 810`) | expand `_orb_high`/`_orb_low` from each close; output `0` |
| Breakout entry | `13:30–15:30` (`810 ≤ mod < 930`) | break of the frozen range, trend-filtered → `+1/-1` |
| Otherwise | — | output `0` |

The trend filter is `sma200 = ctx.ind("bb200_dev1.0_middle", "30m")` (the 30m
BB200 **middle** band = the 30m SMA200 — the repo has no 15m TF, so the operator's
15m trend was adapted to 30m).

```text
# (indices only: A.asset_class(symbol) == "index"; needs a clock & a finite close)
BUY  (+1):  in entry window AND range built AND close > _orb_high AND close > sma200
SELL (-1):  in entry window AND range built AND close < _orb_low  AND close < sma200
INACTIVE (0): no clock; non-index symbol; non-finite close; during range-building;
              entry window but no range / sma200 not finite; price inside the range;
              break against the 200-trend; or any time outside both windows
```

**State (resets each UTC day):** `_orb_high`, `_orb_low` (running max/min of close
over today's opening range; `None` until the first in-window bar) and `_prev_mod`
(previous bar's minute-of-day, to detect the midnight wrap). Two resets: (1)
automatic — when `mod < _prev_mod` (UTC midnight crossed) the range is cleared;
(2) `reset()` — called from `__init__` and per-episode, clears all three.

**Leak-free:** the range is built only from bars **before** the entry window, so
no future information enters the breakout decision. Because the signal is
recomputed every bar from the live close vs the frozen range + trend, it
**persists** while price stays beyond the broken level and the trend holds, and
falls back to `0` the moment price closes back inside the range.

| Threshold | Value |
|---|---|
| `_PRE_START` / `_PRE_END` | `570` (09:30) / `810` (13:30) |
| `_SESS_START` / `_SESS_END` | `810` (13:30) / `930` (15:30) |
| trend filter | 30m `bb200_dev1.0_middle` (= SMA200) |
| break test | strict `>` / `<` (no buffer) |
| asset filter | **indices only** |

Stateful · indices only · registered.

---

## 4. Example / teaching strategies (NOT registered)

📁 [src/strategies/examples/](../src/strategies/examples/) — three minimal,
single-timeframe templates that demonstrate the `BaseStrategy` contract. **They
are not in the alpha pack** (no slot, not used in training); they exist to show
how to write an alpha. Each defaults to `tf="1m"` and is stateless.

| Name | File | Reads | Rule |
|---|---|---|---|
| `sma_trend_50_200` | `examples/sma_trend.py` | `sma_p50_s0`, `sma_p200_s0` | `+1` if `fast > slow·(1+eps)`, `-1` if `fast < slow·(1−eps)`, else `0` (golden/death cross; `eps` default `0.0`) |
| `rsi14_reversion` | `examples/rsi_reversion.py` | `rsi14_raw` | `+1` if `rsi < lower(30)`, `-1` if `rsi > upper(70)`, else `0` (contrarian mean-reversion) |
| `bb20_2_breakout` | `examples/bollinger_breakout.py` | `bb20_dev2.0_upper/lower` + `ctx.close` | `+1` if `close > upper`, `-1` if `close < lower`, else `0` (band breakout) |

All three guard with `MarketContext.is_valid(...)` and return `0` during warm-up.

---

## 5. How to add a new alpha (the pattern)

1. **Write the strategy** — a new file `src/strategies/<name>_alpha.py`:
   subclass `BaseStrategy`, set a unique `name`, implement
   `compute_signal(ctx) -> int` returning only `+1/-1/0`. Read indicators with
   `ctx.ind("col", "tf")` and guard missing values (`_v` helper or `is_valid`)
   so you return `0` during warm-up. **No TA-Lib/pandas in `compute_signal`** —
   only compare cached numbers.
2. **Add a register helper** — `src/strategies/register_<name>_alpha.py` with
   `def register(registry): return registry.register(<Class>())`.
3. **Wire it into the pack** — import and call it in `register_all(...)` in
   [src/strategies/alpha_pack.py](../src/strategies/alpha_pack.py). It takes the
   **next free slot** (16, 17, …). This is **free and shape-stable** up to 64
   alphas — the observation does not change.
4. Going **past 64** alphas (`MAX_STRATEGIES`) is a deliberate
   **observation-contract bump** (resizes `alpha_values`, `alpha_mask`,
   `alpha_streak`) — follow the contract protocol in
   [docs/OBSERVATION_CONTRACT.md](OBSERVATION_CONTRACT.md) and bump the version.

> **The locked invariant.** Adding alphas only ever *fills slots*. It never
> changes the 479-float observation shape, and a previously-trained policy keeps
> working (it simply starts seeing non-zero values in the new slots).

---

## Quick index

| # | Name | Family | Type | TFs | Key indicators |
|---:|---|---|---|---|---|
| 0 | gravity_30m_4h_agree | Gravity | confluence vote | 30m+4h | CCI, RSI, BB(×8), SMA-fan |
| 1 | regime_pulse_trend_5m_30m | Regime Pulse | trend | 30m→5m | close, SMA20, SMA200 |
| 2 | regime_pulse_pullback_5m_30m | Regime Pulse | pullback | 30m→5m | close, SMA20, SMA200 |
| 3 | regime_pulse_trend_30m_4h | Regime Pulse | trend | 4h→30m | close, SMA20, SMA200 |
| 4 | regime_pulse_pullback_30m_4h | Regime Pulse | pullback | 4h→30m | close, SMA20, SMA200 |
| 5 | cci_surge_trend_5m_30m | CCI Surge | trend | 30m→5m | CCI(30), CCI(100) |
| 6 | cci_surge_pullback_5m_30m | CCI Surge | pullback | 30m→5m | CCI(30), CCI(100) |
| 7 | cci_surge_trend_30m_4h | CCI Surge | trend | 4h→30m | CCI(30), CCI(100) |
| 8 | cci_surge_pullback_30m_4h | CCI Surge | pullback | 4h→30m | CCI(30), CCI(100) |
| 9 | sma_stack_trend_5m_30m | SMA Stack | trend | 30m→5m | close, SMA(4)-of-high/low |
| 10 | sma_stack_pullback_5m_30m | SMA Stack | pullback | 30m→5m | close, SMA(4)-of-high/low |
| 11 | sma_stack_trend_30m_4h | SMA Stack | trend | 4h→30m | close, SMA(4)-of-high/low |
| 12 | sma_stack_pullback_30m_4h | SMA Stack | pullback | 4h→30m | close, SMA(4)-of-high/low |
| 13 | sma_reversion_rally_5m_30m | SMA Reversion | rally | 30m→5m | SMA30, SMA50, close |
| 14 | sma_reversion_rally_30m_4h | SMA Reversion | rally | 4h→30m | SMA30, SMA50, close |
| 15 | orb_ny_breakout_indices | ORB | breakout (stateful) | 30m + 1m close | opening range, 30m SMA200 |
| — | sma_trend_50_200 | *example* | trend | 1m | SMA50, SMA200 |
| — | rsi14_reversion | *example* | reversion | 1m | RSI14 |
| — | bb20_2_breakout | *example* | breakout | 1m | BB(20, 2.0) |

---

*Generated 2026-06-28 by reading every file in `src/strategies/` (verified against
`config/constants.py` and `src/indicators/`). Keep it in sync with the code — the
code is the source of truth.*
