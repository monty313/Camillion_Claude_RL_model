# JARVIS Cockpit — Live Bot Wiring — Handoff for GitHub Codespaces

**Goal:** connect the JARVIS HUD (`JARVIS Cockpit.dc.html`) to the real
`Camillion_Claude_RL_model` PPO/MLP bot so every panel and every word of JARVIS's
reasoning runs on **live data** instead of the current simulation. JARVIS stays
**read-only** — he observes, reasons, and advises; he never places orders or edits
trading code.

This repo already ships the wiring described below:
1. A tiny **bridge server** (`jarvis_bridge.py`) that exposes the bot's live state as JSON.
2. The HUD points at that bridge (`pullLive()` in the `.dc.html`).
3. A **test suite** that proves every field works (`tests/test_jarvis_bridge.py`).

---

> **BOT STATE (2026-07-01) — read this first.** The policy is now a **multi-head "super-scalper" actor**:
> observation **v1.12.0 (557 float32)**; the actor outputs direction + continuous **TP / SL / lot** (bracket
> orders, 1%-equity risk clamp), trained via a freeze/unlock **curriculum** (`ACTOR_CURRICULUM_STAGE` 1→2→3)
> and an **R:R self-discovery reward**. The bracket actor is **default-OFF** (`bracket_enabled=0`) so the
> discrete bot + this cockpit wiring are unchanged. **Super-scalper Stages 1–4 are complete + CPU↔JAX
> parity-verified**; the **ONNX export** now emits `direction_logits[4], tp_pct, sl_pct, lot_mult`. Deferred:
> the **MT5 EA (`.mq5`, external) must read the 4 ONNX outputs** before live deployment. Details:
> `TRAINING_TASKS.md`, `README.md` QUICKSTART, `docs/UPDATE_LOG.md`.

---

## 0. The big picture

```
  Camillion_Claude_RL_model (Python)
        │  env.step() / AccountState / PolicyIntrospector
        ▼
  src/jarvis/state_provider.py  → snapshot
        ▼
  src/jarvis/bridge_state.py    → build_state(snapshot) → the /state dict (PURE, testable)
        ▼
  jarvis_bridge.py  ── serves ──►  GET /state  (JSON, ~1/second)
        ▲                                   │
        └───────────  JARVIS Cockpit.dc.html (pullLive() fetches /state)
```

The HUD has a fixed internal "shape" it reasons over (`snapshot()` in the file). The
bridge's only job is to make `/state` return that shape filled with real numbers.
**Do not change the HUD's UI** — just feed it.

---

## 1. The data contract (SOURCE OF TRUTH)

`GET /state` returns exactly this JSON. Field names and types matter.

```jsonc
{
  "account": {
    "balance": 100000, "equity": 106240.12, "day_start_equity": 106900,
    "episode_start_equity": 104100, "peak_equity": 107010
  },
  "ftmo": {
    "daily_loss_limit_pct": 5, "max_drawdown_limit_pct": 10,
    "profit_target_pct": 10, "daily_target_pct": 1.5
  },
  "position": {
    "dir": "SHORT", "symbol": "EUR/USD", "lots": 0.80,
    "entry": 1.08562, "price": 1.08410, "age_min": 47
  },
  "alphas": [ { "name": "EMA Cross", "signal": -1, "streak": 3 } ],
  "policy": {
    "action": "SELL", "prob_buy": 0.18, "prob_sell": 0.74, "prob_hold": 0.08,
    "value": 0.31, "confidence": 0.72, "entropy": 0.41, "advantage": 0.20,
    "regime": "mean-revert", "recommended_lots": 0.60,
    "expected_dd_pct": 0.80, "value_calibration_pct": 78
  },
  "perf": {
    "win_rate_pct": 58, "trades": 214, "consecutive_losses": 1,
    "day_history": [820, -410, 1180, 640, 905]
  },
  "news": [ { "start": "14:00", "end": "14:30", "currency": "USD",
             "title": "FOMC Rate Decision", "impact": "high" } ],
  "human": { "overrides": 4, "panic_closes": 2, "discipline_pct": 83 },
  "clock": "13:18:11"
}
```

### Where each value comes from in `Camillion_Claude_RL_model`

| Contract field | Source in the repo |
|---|---|
| `account.*` | `AccountState` (balance, equity, `episode_peak_equity`→peak, `day_start_balance`, `episode_start_balance`) |
| `ftmo.*` | `config/ftmo_config.py` + `config/variables.py` (the active config) |
| `position.*` | the env's live single-position state (`dir`/`symbol`/`lots`/`entry`/`mark`/`age`) |
| `alphas[]` | the **directional** alpha slots (`registry` names + `alpha_matrix` + `streak_matrix`). Non-directional gates are surfaced separately so a gate's `1` is never a bullish vote |
| `policy.action/prob_*/value/entropy` | `PolicyIntrospector.introspect(policy, obs)` (softmax of PPO logits + value head) |
| `policy.advantage/regime/recommended_lots/expected_dd_pct/value_calibration_pct` | risk model / rolling calibration if present, else a clearly-flagged safe default |
| `perf.*` | `AccountState` win/trade/consec counters + realized day history |
| `news[]` | optional Firecrawl/ForexFactory scrape (see §5) or a static calendar |
| `human.*` | manual-override / early-close counters vs the policy (start at 0) |

> If a value isn't genuinely produced by the bot, return a safe default (`0`, `"n/a"`)
> and flag it — **never fabricate**. JARVIS is instructed to say so if a figure is missing.

---

## 2. The bridge — `jarvis_bridge.py`

At the repo root. FastAPI is imported **lazily** (the repo's stdlib test runner has no
FastAPI), so importing the contract logic never requires the web stack.

```bash
pip install fastapi uvicorn
uvicorn jarvis_bridge:app --reload --port 8000
# open http://localhost:8000/JARVIS%20Cockpit.dc.html
```

The contract itself lives in `src/jarvis/bridge_state.py::build_state(snapshot)` — a **pure**
function (stdlib + numpy) that is unit-tested without a server. `src/jarvis/state_provider.py`
produces the snapshot from the env/account/policy (with a no-model net-signal fallback so the
bridge works out of the box).

---

## 3. Point the HUD at the bridge

`pullLive()` (in the `.dc.html`) fetches `http://localhost:8000/state` once per second and,
when the bridge answers, takes over from the simulation: `policySnap()` prefers the live
policy, `dataTick()` early-returns while live, and `newsSnap()` uses the live calendar. If the
bridge is down, the simulation keeps running so the HUD never goes blank.

---

## 4. JARVIS's brain (Claude) — already wired
The chat/voice calls Claude with the full `snapshot()` as context, so once `/state` is live his
reasoning is about the real bot. No change needed.

---

## 5. (Optional) Red-folder news via Firecrawl
Browsers can't scrape ForexFactory (CORS). In the bridge, scrape the calendar server-side,
keep `impact == "High"`, map to `{start,end,currency,title,impact:"high"}`, cache ~10 min, and
return it in `state()["news"]`.

---

## 6. Tests

`tests/test_jarvis_bridge.py` runs under the repo's stdlib runner (`python tools/run_tests.py`)
and proves the contract: every key present, types/ranges correct, `prob_*` sum ≈ 1, each alpha
`signal ∈ {-1,0,1}`, `peak_equity ≥ equity`, a non-directional gate never counts as a bullish
vote, and **no mutation endpoint exists** (read-only guarantee). A server smoke test runs too
when FastAPI is installed and is skipped otherwise.

---

## 7. Acceptance checklist
- [ ] `GET /state` returns the §1 schema, 200, ~1/s, no errors
- [ ] HUD shows live equity / P&L / position matching the bot
- [ ] PPO·MLP LINK action + value + confidence match `PolicyIntrospector`
- [ ] AGENT PSYCHOLOGY moves with real losses/size
- [ ] P(PASS) gauge updates from live data
- [ ] RED FOLDER WATCH fires from `news[]`
- [ ] JARVIS chat cites real numbers; refuses to trade (read-only)
- [ ] EXPORT BRIEF downloads a populated report
- [ ] `python tools/run_tests.py` (and `pytest tests/`) green
- [ ] No endpoint can place/modify a trade (read-only guarantee)
