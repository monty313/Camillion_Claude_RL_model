# PHASE 1 REPORT — data, env, training, interpretability

## Cache: format + what is precomputed
Source = 1-minute OHLCV (MT5 live / exported CSV / parquet). Precomputed ONCE by
`src/data/cache_builder.py`, stored **float32** and memmap-loaded:
- `{symbol}_indicators.npy` — `(T, 200)` aligned multi-timeframe indicators
- `{symbol}_close.npy` — `(T,)` close · `{symbol}_time_ns.npy` — `(T,)` int64 time
**Alignment (leak-free):** each higher TF (5m/30m/4h/1d) is resampled from 1m,
indicators computed on it, then aligned to the 1m timeline by the **last
higher-TF bar that has CLOSED** (`close_time <= current 1m close`). The in-progress
higher-TF bar is never used. Proven in `tests/test_cache_no_leakage.py`.
`env.step()` reads only these arrays — no TA-Lib/MT5/pandas in the loop.

## Observation schema the trainer's policy sees — 367 float32 (contract v1.1.0)
indicators 200 · alpha_values 64 · alpha_mask 64 · alpha_summary 4 ·
signal_memory 5 · signal_accuracy 2 · account_daily 7 · account_episode 7 ·
time 6 · portfolio 8. Indicators from the cache row; alphas from the AlphaRegistry;
account/portfolio from the live FTMO account; summary/memory/accuracy precomputed
leak-free.

## Reward — plain English + formula
**Plain English:** the real money made or lost this step — the change in account
equity — as a fraction of the starting balance, minus a penalty if an FTMO limit
was breached. **No alpha, reliability, accuracy, or signal term appears in it.**
**Formula:** `r_t = (equity_{t+1} - equity_t) / starting_balance
- breach_penalty * 1[breached]`. Two-phase auto-flat on hitting the daily target
realizes PnL, which is already inside the equity term. Proven alpha-independent
by `tests/test_trading_env.py::test_reward_independent_of_alphas`.

## FTMO state, day by day
`AccountState` tracks daily + episode realized PnL, equity peaks (trailing DD),
wins/losses, loss streaks. At each midnight the env calls `reset_day()` (daily
counters reset; `day_start_balance = balance`). Every step the breach detector
checks daily DD / total DD / trailing DD (all editable, no retrain); a breach
terminates the episode. Daily +target% triggers two-phase auto-flat -> fresh trail.

## Policy Doctor in evaluation (read-only, eval-safe)
`src/training/evaluate.py` runs the policy greedily through the env, capturing
per-step introspection (action distribution, value, entropy, block-ablation
saliency) and building the Policy Doctor report (alpha-vs-policy 1/3/10 scoreboard,
explicit leader-chasing flag, best-single-alpha comparison, block importance). It
**never** modifies training; the trainer calls it read-only.

## Minimum Colab steps (end to end)
1. Open `notebooks/Camillion_One_Click_Train.ipynb`.
2. Mount Drive -> clone -> `pip install numpy pandas gymnasium stable-baselines3 torch` -> run tests.
3. Build the cache from your 1m data (or the demo synthesizer).
4. Register alphas -> `trainer.train(...)` -> `evaluate_policy(...)` (Policy Doctor) -> save/resume.

## Definitions kept explicit (as requested)
- **alpha signal vs alpha reliability** — the *signal* is a strategy's current
  output (+1/-1/0), and it IS in the observation. *Reliability* is how often that
  alpha's past signals were directionally right (rolling 1/3/10-bar accuracy): a
  DIAGNOSTIC, not in the observation and not in reward.
- **action decision vs directional grading** — an *action* is what the policy does
  (HOLD/BUY/SELL/CLOSE). *Directional grading* maps an action to {-1,0,+1} and
  scores whether it predicted the next move's sign: a measurement, not a control.
- **policy directional accuracy vs actual trading performance** — *directional
  accuracy* is a hit-rate ("did its calls predict direction"). *Trading
  performance* is realized PnL, drawdown, FTMO pass behaviour, trade quality —
  what actually matters and the ONLY thing reward is tied to. A policy can have a
  lower hit-rate but better performance; we report both and reward only the latter.

## Caveat (interpretability honesty)
Block-ablation saliency is **evidence, not ground truth** — a debugging/comparison
lens for catching shortcut learning, not a perfect read of "what the policy really
thinks." Treat it as one signal among several.
