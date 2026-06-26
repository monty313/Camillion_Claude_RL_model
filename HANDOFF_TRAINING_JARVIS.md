# HANDOFF — Camillion (training + JARVIS need a review)

**Date:** 2026-06-26 · **Branch:** `feat/jarvis-bridge` (PR #13, open) · **Latest commit:** `877fa2c`
**Owner:** Mark / monty313 — **NOT a programmer.** Keep everything simple, commented, Colab-friendly.
**Read first:** `CLAUDE.md` (the 3 hard rules) and `docs/UPDATE_LOG.md` (dated IRAC history — newest at the bottom).

> Purpose of this file: hand the next agent a clear picture so it can **look over TRAINING and JARVIS**.
> The plumbing works and is tested (suite 156/156, audit GO 38/42), but **neither has been confirmed
> end-to-end on Mark's REAL data yet** — that's the job.

---

## 0. The 3 invariants you must not break (from CLAUDE.md)
1. **Observation is LOCKED at 479 float32 (contract v1.5.0).** Never change the shape silently.
2. **Never change the FTMO numbers** (2.5% daily target, 4% trailing wall, 5% daily / 10% total hard lines).
3. **Nothing heavy (TA-Lib / MT5 / pandas) inside `env.step()`** — the hot loop only reads cached float32.

Run the tests after any change: `python tools/run_tests.py` (stdlib, no pytest needed).
Run the full health check: `python tools/run_full_audit.py` → must end **✅ GO** (writes `audit_results/*`).

---

## 1. How everything runs (the commands)

**Train (Colab, where Mark's Google-Drive data lives):**
```bash
# QUICK first run (minutes) — ALWAYS do this before the full run:
python run_training.py --data /content/drive/MyDrive/Camillion_data --from 2024-01-01 --to 2024-03-31 --steps 200000
# FULL run (hours on Colab CPU):
python run_training.py --data /content/drive/MyDrive/Camillion_data
```
It auto-finds the 4 CSVs, builds leak-free caches, trains ONE shared-pot portfolio bot, prints a
**day-by-day** report (did it make +2.5% / stay inside the 4% wall), and registers the policy.

**JARVIS (read-only cockpit + advisor):**
```bash
pip install -r requirements-jarvis.txt
python go_live.py --port 8000                                   # synthetic DEMO (no model)
python go_live.py --data data_cache --model models/camillion_portfolio_ppo --symbols EURUSD,GBPUSD,XAUUSD,US30
# open http://<host>:8000/  (root redirects to the cockpit 0_JARVIS_COCKPIT.html)
```
On **Colab**, render it INLINE (no link/DNS issues): the notebook Step 6 cell uses
`google.colab.output.serve_kernel_port_as_iframe(8000, path='/0_JARVIS_COCKPIT.html')`.
On the **Codespace**, open the **PORTS** tab → globe icon on port 8000.

**One-click notebook:** `notebooks/Camillion_One_Click_Train.ipynb` (mount Drive → clone → install →
audit → train → open JARVIS). Colab link uses branch `feat/jarvis-bridge`.

---

## 2. ⚠️ TRAINING — what to look over (the priority)

**Mark's data:** 4 MetaTrader-5 exports, 2021→2026, ~2M 1-min bars each, ~1.8M *shared* bars after
alignment. Symbols: EURUSD, GBPUSD, XAUUSD, US30.

**Status:** the pipeline RUNS (loader fixed for MT5; the OOM hang fixed). **It has NOT been confirmed
that the bot actually LEARNS / trades / passes on the real data.** Open items, most important first:

1. **Confirm a quick run end-to-end on real data** (`--from 2024-01-01 --to 2024-03-31 --steps 200000`)
   and READ the `[4/5]` day-by-day table. Does it trade at all? Make +2.5% on any day? Stay inside 4%?
2. **Entropy collapse risk — HIGH.** `PPO_HPARAMS` in `src/training/trainer.py` has **`ent_coef=0.0`**.
   With zero entropy bonus the policy can collapse to **always-HOLD** and never trade. CHECK the trained
   policy's action distribution (e.g. `src/barbershop/policy_doctor.py`, or JARVIS `policy.entropy`). If
   it only HOLDs, raise `ent_coef` to ~0.005–0.01 and retrain. (This is also why the audit's JARVIS
   knowledge has an `entropy-collapse` entry.)
3. **Speed/memory on the FULL history.** The OOM hang (SubprocVecEnv pickling ~6 GB × 8 workers) is fixed
   — `make_portfolio_vec_env` now uses **DummyVecEnv** (shared arrays, one process) with `min(4,N_ENVS)`
   envs, and `train_portfolio` prints a **heartbeat** (steps/s + ETA). BUT: building the env over 1.8M
   bars × 4 symbols × 4 envs (= 16 sub-`TradingEnv`s) may still be slow — **profile the env-build time on
   real data** and the per-step rate from the heartbeat. If impractical on Colab free, recommend training
   on a 1–2 year `--from` slice and/or lower `--steps`, or a stronger runtime.
4. **Does it generalise?** Judge on held-out **walk-forward** windows, not the bars it trained on
   (`src/training/walk_forward.py`, `run_log.best_run(fingerprint=env_fingerprint())`), NOT a single
   day-by-day run. The day-by-day report is illustrative, not the objective metric.
5. **Reward / FTMO behaviour** — verify the breach engine actually halts on the pot (daily 5% / total 10%
   / 4% trailing) during a real run, and that the two-phase +2.5% bank-and-stop fires.

**Key training files:** `run_training.py` (entry), `src/training/trainer.py` (`train_portfolio`,
`PPO_HPARAMS`, `_make_heartbeat`), `src/training/vector_env_factory.py` (`make_portfolio_vec_env` →
DummyVecEnv), `src/env/portfolio_env.py` (shared-pot env + `align_symbol_data`),
`src/data/cache_builder.py` (`load_ohlcv_csv` now handles MT5; `build_aligned_indicators` 1m→5TF leak-free),
`src/training/daily_report.py` (`run_portfolio_report`).

---

## 3. ⚠️ JARVIS — what to look over

**Status:** read-only cockpit + grounded council (OMEGA→JUSTICE→JARVIS) are built and tested; the
"open JARVIS" link bugs are fixed. **NOT confirmed against a REAL trained model** (only synthetic demo).

1. **Open it with a real model** (`go_live.py --data data_cache --model models/camillion_portfolio_ppo`)
   and verify `/health` shows `model_attached:true` and the panels reflect the trained policy (not the
   honest no-model fallback). On Colab use the inline-iframe cell; verify it actually renders.
2. **LLM path.** The council can use Anthropic (`claude-opus-4-8`) when `use_llm` is on AND
   `ANTHROPIC_API_KEY` is set; otherwise it runs a **deterministic, grounded** fallback. Verify which
   mode Mark wants and that the key is available; the deterministic mode must always work offline.
   (`src/jarvis/council.py` — `deliberate`, `answer`, `_call_llm`.)
3. **Read-only guarantee** must hold: GET routes only; any POST/order → 405
   (`tests/test_jarvis_bridge.py::test_bridge_routes_are_get_only_and_portfolio`).
4. **Cockpit reachability** is covered by `tests/test_jarvis_bridge.py` (`test_cockpit_url_is_wellformed`,
   `test_root_url_redirects_to_existing_cockpit`) and audit check `6.6`. Keep these green.
5. **Two environments** confuse Mark — be explicit: he **trains in Colab** (his data), and JARVIS in the
   **Codespace** is only a no-model DEMO. To see his real bot, JARVIS must run in Colab after training.

**Key JARVIS files:** `jarvis_bridge.py` (`create_app`, `cockpit_url`, `COCKPIT_FILE`,
root redirect), `go_live.py` (launcher; `--data`/`--model`/`--port`), `src/jarvis/{council,knowledge,
state_provider,state_contract,consistency,market_view,policy_registry}.py`, `0_JARVIS_COCKPIT.html` (HUD).
Older contract reference: `HANDOFF.md` (note: it predates some renames — real cockpit file is
`0_JARVIS_COCKPIT.html`, state builder is `src/jarvis/state_contract.py`, daily target is 2.5%).

---

## 4. Recent fixes (context for the new chat — newest last)
- One-command full-system audit `tools/run_full_audit.py` (44 checks + GO/NO-GO; runs the unit suite as
  check `0.0`). Adversarial review hardened it (dead-neuron probe, locale, HTML-escape, false-NO-GO).
- MT5 CSV loader: `load_ohlcv_csv` auto-detects tab/semicolon, strips `<...>`, combines `<DATE>`+`<TIME>`
  (keeps minute resolution), parses dotted dates, prefers tick volume.
- JARVIS link: tested `cockpit_url` helper + root redirect + **inline iframe** on Colab.
- `run_training.py`: `--from/--to` date range for a fast first run.
- **Training hang FIXED:** DummyVecEnv (no gigabyte pickling) + heartbeat (visible progress/ETA).

## 5. Verification status (at handoff)
- `python tools/run_tests.py` → **156/156** green.
- `python tools/run_full_audit.py` → **✅ GO, 38/42**, 0 critical failures. 4 honest warnings (LIVE-only):
  weekend auto-close, regime-coverage (data-dependent), checkpoint contract-version guard, reconnect layer.
- **Not yet done:** a confirmed real-data training run + JARVIS on a real model. **That is the task.**

## 6. First moves for the next agent
1. Read `CLAUDE.md` + the tail of `docs/UPDATE_LOG.md`. Run `python tools/run_tests.py` (expect 156/156).
2. Help Mark do the **quick training run** and read the day-by-day table together; check it actually
   trades (entropy/HOLD). Then decide on the full run vs a `--from` slice.
3. Open **JARVIS on the trained model** and verify `model_attached:true` + real panels.
4. Anything you change: keep the 3 invariants, add a dated `docs/UPDATE_LOG.md` entry, keep tests + audit
   green, and write for a non-programmer.
