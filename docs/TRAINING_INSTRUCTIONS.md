# TRAINING — the easy way (one command) + the manual steps if you want them

> Goal: train **one bot that trades your whole portfolio**, then read the results the way the challenge
> is judged — **day by day**: did it make **+2.5% of your balance**, and did it stay inside the **4%
> trailing-drawdown wall**? You do **not** need to understand trading to run this.

---

## ⭐ THE EASY WAY — one command (recommended)
1. Put your four 1-minute CSVs in one folder (names containing **EURUSD, GBPUSD, XAUUSD, US30**). On
   Colab that's your Google Drive folder.
2. Install the engine once: `pip install stable-baselines3 torch` (the Colab notebook does this for you).
3. Run **one line**:
```bash
python run_training.py --data /content/drive/MyDrive/Camillion_data
```
That's it. It finds your files, prepares the features, trains **one bot on all four from one account**,
then prints the **day-by-day** table and a plain summary:
```
[1/5] Looking for your 1-minute data ...        EURUSD ready, GBPUSD ready, XAUUSD ready, US30 ready
[3/5] Teaching ONE bot to trade all 4 together, from one account ...
[4/5] How it did, DAY BY DAY — did it make +2.5% and stay inside the 4% wall:
   DAY  DATE         P&L%   +TGT?  TRAIL_DD%  <WALL?   ...   CUM%
     1  2026-03-02   2.61    YES      1.30      ok            2.61
     ...
ALL DONE.  2/4 days hit +2.5%  |  4/4 stayed inside the 4% wall  |  final +7.24%   Challenge: not yet
```
**In Colab it's literally one cell:** `!python run_training.py --data /content/drive/MyDrive/Camillion_data`.
*(Want the daily target at 2.4% instead of 2.5%? Set `FTMO_DAILY_TARGET_PCT = 2.4` in
`config/variables.py` once — the report re-judges against it.)*

---

## The manual steps (only if you want the detail — the one command above does all of this)

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

## 4. Train ONE bot on ALL FOUR symbols from ONE shared pot (learns to balance)
This is the key step. **`train_portfolio`** trains a **single policy** on the **shared-pot
`PortfolioEnv`**: it holds simultaneous positions across **all four symbols in one account**, decides one
symbol at a time **while seeing how exposed the pot already is**, and is rewarded on the **pot's** equity.
That's how the bot learns to **balance the book** — and because decisions are per-symbol with portfolio
context, the same policy **scales to the full FTMO broker list live without changing the locked 479
observation**:
```python
from src.training.trainer import train_portfolio
from src.env.portfolio_env import align_symbol_data
from src.data.cache_builder import load_cache
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
def _r():
    r = AlphaRegistry(); register_all(r); return r

# align on shared bars (FX vs index hours differ), then train one policy on the whole pot
symbol_data = align_symbol_data({s: load_cache("data_cache", s) for s in ["EURUSD", "GBPUSD", "XAUUSD", "US30"]})
train_portfolio(symbol_data, _r, total_timesteps=2_000_000, save_path="models/camillion_portfolio_ppo")
```
Saves **`models/camillion_portfolio_ppo`** + its **`_vecnorm.pkl`** — **keep both**. *(Lighter alternative:
`train_multi_symbol` trains one policy across symbols with SEPARATE accounts — it generalises across
assets but does not learn shared-pot risk allocation. Use `train_portfolio` for the real portfolio bot.)*

## 5. ⭐ SEE THE DAY-BY-DAY RESULTS (+2.5% target + trailing DD), on the SHARED POT
One row per day for the **whole portfolio (one account)** — did it make +2.5% and stay inside the 4% wall:
```bash
python -m src.training.daily_report --data data_cache --portfolio EURUSD,GBPUSD,XAUUSD,US30 \
    --model models/camillion_portfolio_ppo
```
*(Single symbol instead: `--symbol EURUSD --model models/camillion_ppo`.)*
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
