# TRAINING — step by step (and see the DAY-BY-DAY +2.5% / trailing-DD results)

> Goal: train one policy that trades the **whole FTMO portfolio**, then read its results the way the
> challenge is judged — **day by day**: did it make **+2.5% of the initial balance**, and did it stay
> inside the **4% trailing-drawdown wall**? (2.5% is the FTMO daily target; if you really want 2.4%,
> set `FTMO_DAILY_TARGET_PCT = 2.4` in `config/variables.py` — every report below uses whatever you set.)

Do these in order. Steps 1–4 train; **step 5 is the day-by-day report you asked for.**

---

## 0. Open the environment
- **Colab (recommended):** open `notebooks/Camillion_One_Click_Train.ipynb`. Training needs
  `stable-baselines3` + `torch` + TA-Lib — the notebook installs them.
- **Locally:** `pip install -e .` then `pip install stable-baselines3 torch` (+ TA-Lib).
- Sanity check first: `python tools/run_tests.py` → all green.

## 1. Load ALL FOUR symbols from Google Drive
Train on the **whole universe — EURUSD, GBPUSD, XAUUSD, US30 — together**, so the one policy learns to
**balance the portfolio** (when to lean into the index vs the metal vs the pairs) instead of overfitting
one easy asset. In Colab, mount your Drive and point at the folder with your four 1m CSVs:
```python
from google.colab import drive; drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/Camillion_data"   # <- your folder of 4 CSVs
FILES = {"EURUSD": f"{DRIVE}/EURUSD_1m.csv", "GBPUSD": f"{DRIVE}/GBPUSD_1m.csv",
         "XAUUSD": f"{DRIVE}/XAUUSD_1m.csv", "US30":   f"{DRIVE}/US30_1m.csv"}   # adjust names
```
Any common column names work (`datetime/timestamp` + `open/high/low/close/volume`).

## 2. Build the leak-free cache for all four
```python
from src.data.cache_builder import load_ohlcv_csv, build_cache
for sym, path in FILES.items():
    build_cache(load_ohlcv_csv(path), out_dir="data_cache", symbol=sym)   # one float32 cache per symbol
```
This precomputes every indicator **once**, aligned leak-free (last-closed-bar) — the env never touches
TA-Lib/pandas in its hot loop.

## 3. Set your risk (no retrain needed later)
Edit `config/variables.py` if you want to change anything — these are **percentages**, so you can retune
them later without retraining:
- `FTMO_DAILY_TARGET_PCT = 2.5`   ← the +2.5%-of-initial daily target
- `FTMO_TRAILING_DRAWDOWN_PCT = 4.0`   ← the trailing wall
- `FTMO_TWO_PHASE_ENABLED = True`   ← hit +2.5% → bank & stop for the day

## 4. Train ONE policy across ALL FOUR symbols (learns to balance)
This is the key step: `train_multi_symbol` spreads the workers across **all four caches** and trains a
**single policy** over them, with per-asset calibrated size + the cross-asset observation features
(ATR-normalized, asset-class identity). That's how the bot learns to **balance the book** — read every
asset in common units and allocate risk across them — instead of mastering one and ignoring the rest:
```python
from src.training.trainer import train_multi_symbol
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.data.cache_builder import load_cache

symbol_data = {s: load_cache("data_cache", s) for s in ["EURUSD", "GBPUSD", "XAUUSD", "US30"]}
train_multi_symbol(symbol_data, lambda: _r(), total_timesteps=2_000_000,
                   save_path="models/camillion_ppo")

def _r():
    r = AlphaRegistry(); register_all(r); return r
```
This saves **`models/camillion_ppo`** + **`models/camillion_ppo_vecnorm.pkl`** — **keep both** (eval needs
the vecnorm stats). *(Single symbol? Use `trainer.train(*load_cache("data_cache","EURUSD"), _r, ...)`.)*

## 5. ⭐ SEE THE DAY-BY-DAY RESULTS (+2.5% target + trailing DD)
This is the output you want — one row per day, did it hit +2.5% and stay inside the 4% wall:
```bash
# per symbol (run for each; the portfolio shared-pot day-by-day is the next env build)
python -m src.training.daily_report --data data_cache --symbol EURUSD --model models/camillion_ppo
```
Example output:
```
DAY-BY-DAY FTMO REPORT  (daily target +2.5% of initial | trailing wall 4.0%)
DAY  DATE         P&L%   +TGT?  TRAIL_DD%  <WALL?  DAILY_LOSS%  BREACH    CUM%
------------------------------------------------------------------------------------
  1  2026-03-02   2.61    YES      1.30      ok       0.40        no      2.61
  2  2026-03-03   2.48    no       3.10      ok       1.10        no      5.09
  3  2026-03-04  -0.90    no       2.20      ok       1.80        no      4.19
  4  2026-03-05   3.05    YES      0.90      ok       0.20        no      7.24
------------------------------------------------------------------------------------
SUMMARY: 4 days | hit +2.5%: 2/4 | within 4.0% trailing: 4/4 | breaches: 0 | final +7.24% | CHALLENGE not yet (need +10.0% with 0 breaches)
```
- **+TGT?** = made +2.5% of initial that day. **<WALL?** = stayed inside the 4% trailing drawdown.
- **BREACH** = hit a hard line. **CHALLENGE PASSED** = reached +10% with **zero** breaches.
- (You can also call `daily_report(env, policy)` in a notebook to get the rows as data.)

## 6. The objective metric — walk-forward pass-rate
Day-by-day shows *one* run; the number to trust across many held-out windows is the **walk-forward
pass-rate** (fraction of windows that reach +10% without breaching). See `docs/TRAINING_LEDGER.md` and
`src/training/run_log.py` (`best_run(fingerprint=env_fingerprint())`).

## 7. Register the policy so JARVIS organizes it
```bash
python -m src.jarvis.policy_registry add --id portfolio-v1 --path models/camillion_ppo \
   --fingerprint <env_fingerprint()> --pass-rate 0.7 --max-dd 3.2 --largest-day 30 --universe "EURUSD,GBPUSD,XAUUSD,US30"
```
Now JARVIS ranks it by consistency; ask him *"which policy should I run?"*.

## 8. (Optional) Go live + ask JARVIS
```bash
pip install -r requirements-jarvis.txt
python go_live.py --data data_cache --symbols EURUSD,GBPUSD,XAUUSD,US30 --model models/camillion_ppo
# open http://localhost:8000/JARVIS%20Cockpit.dc.html ; ask "how did I do day by day?"
```

---
**Tips & gotchas:** eval/report **must** use the saved vecnorm (the report loader does this for you);
judge consistency on **held-out** data, not the bars you trained on; a different `env_fingerprint()`
means a different experiment (don't compare pass-rates across fingerprints). Stuck? read
`docs/TROUBLESHOOTING.md` or just **ask JARVIS**.
