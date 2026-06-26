# JARVIS GUIDE — the cockpit, the council, and how to ask JARVIS for fixes

> JARVIS is your **read-only** co-pilot: he watches the real bot, reasons about it with two other
> agents, and coaches you toward passing the FTMO challenge **consistently** — but he can **never**
> place or modify a trade. This explains every piece, every endpoint, and exactly how to ask him
> "how do I fix X?". For the rest of the repo see `docs/REPO_GUIDE.md`.

---

## 1. What JARVIS is (three things)
1. **A live, honest data feed** (`/state`) — the real bot's account, position, alphas, policy, and
   FTMO headroom, served as JSON. It **never fabricates**: anything the bot doesn't produce is a
   safe default listed in `gaps`.
2. **A reasoning council** (`/council`) — **OMEGA → JUSTICE → JARVIS** talk to each other over the
   live system + your chat, grounded in the system's own numbers, and **always** end on the next
   improvement toward a consistent pass.
3. **A knowledge brain** (`/knowledge`, `/ask`) — JARVIS carries a map of how the whole bot works
   plus ~30 grounded fixes for training/trading problems, so you can ask him directly and get a
   system-correct answer.

All of it is **structurally read-only**: the bridge has **GET routes only**. `POST /order` → **405**.

## 2. Architecture
```
  JARVIS Cockpit.dc.html (the HUD)
        │  pullLive() → GET /state        councilLive() → GET /council        ask → GET /ask
        ▼
  jarvis_bridge.py  (FastAPI, lazy import, GET-only, read-only)
        │
        ├─ src/jarvis/state_provider.py  → snapshot the REAL env+account+policy (or no-model fallback)
        ├─ src/jarvis/state_contract.py  → build_state(snapshot): the exact /state JSON (pure, honest)
        ├─ src/jarvis/consistency.py     → analyze_consistency(state): the system-logic read
        ├─ src/jarvis/council.py         → OMEGA→JUSTICE→JARVIS deliberation (+ optional LLM)
        └─ src/jarvis/knowledge.py       → how-it-works summary + the troubleshooting fixes
```
The contract + council are **pure Python (stdlib+numpy)** and fully tested **without** FastAPI; only
`jarvis_bridge.py` imports FastAPI, and lazily — so the test runner needs no web stack.

## 3. The modules, in detail

### `state_provider.py` — the live snapshot (read-only)
`StateProvider(env, registry, policy=None)` holds a real `TradingEnv`, the real alpha registry, and
an **optional** policy. `snapshot()` reads the live account/position/alphas/policy and returns raw
primitives. Key behaviors:
- **No-model fallback** — with `policy=None`, the action is an honest **alpha-consensus lean at
  confidence 0** (it never pretends the model decided).
- **Directional mask** — gates are excluded from the net signal (uses `registry.directional_mask()`
  if present, else a per-slot `DIRECTIONAL` fallback). The net signal is divided by the **count of
  directional alphas**, never a hardcoded 15.
- **Tracks** position age and a per-day realized-P&L history itself (the env doesn't store them).
- **Constructors:** `from_synthetic(...)` (runs the real env on placeholder data, for demo/tests)
  and **`from_cache(data_dir, symbol, policy=None)`** (the **go-live** path: real cached market data
  + the real 16 alphas + your optional trained policy).

### `state_contract.py` — `build_state(snapshot) -> dict` (pure, honest)
Shapes the snapshot into the exact `/state` JSON. It **folds the 4-action policy** `{HOLD,BUY,SELL,
CLOSE}` into the HUD's 3-way `{BUY,SELL,HOLD}` (CLOSE→HOLD), computes `confidence` from entropy,
exposes a **directional-only `net_signal` + `net_signal_basis`**, and lists every defaulted field in
**`gaps[]`** so JARVIS can say "not yet measured." It never crashes on missing/None inputs.

### `consistency.py` — `analyze_consistency(state) -> dict` (the brain)
The single system-logic read the agents cite: progress to target, **daily- and max-DD headroom**,
the **binding constraint**, pace vs the +2.5%/day plan, day-to-day concentration, decisiveness, a
**p(pass)** estimate, the **top risk to consistency**, and — always — a **progressive next step** +
a **posture** (`STAND DOWN / BANK & STOP / DEFENSIVE / STEADY / SELECTIVE / PRESS-STEADY`).

### `council.py` — OMEGA → JUSTICE → JARVIS
`deliberate(state, chat_history, use_llm)` runs the three agents **in order**, each seeing the full
grounded context + the chat + **the prior speakers** (they reason *together*):
- **OMEGA** (telemetry analyst) surfaces the inefficiency that matters, citing the number.
- **JUSTICE** (risk arbiter) weighs OMEGA against the FTMO walls + the consistency goal and names the
  binding constraint.
- **JARVIS** (IRAC arbiter) rules: **Issue / Rule / Application / Conclusion**, ending on the one
  next improvement.
A **deterministic, system-grounded core** always works and is fully tested; an **optional Anthropic
LLM** (`claude-opus-4-8`) activates only when `ANTHROPIC_API_KEY` is set + `anthropic` is installed,
and degrades cleanly to deterministic. Every reply is **grounded** (cites real numbers) and
**progressive** (never "nothing to do"). `answer(question, state)` is the "how do I fix X" path.

### `knowledge.py` — what JARVIS always knows
- `SYSTEM_SUMMARY` — a compact map of how the whole bot works (always in JARVIS's context).
- `TROUBLESHOOTING` — ~30 grounded fixes (`{area, symptom, cause, fix, refs}`) for **training,
  trading, data, and bridge** problems.
- `search(q)` ranks the right fix for a question; `as_context(q)` builds the LLM string;
  `render_markdown()` generates `docs/TROUBLESHOOTING.md` (single source of truth).

### `jarvis_bridge.py` — the server (read-only)
`create_app(provider)` wires the GET routes; `main()` runs uvicorn. Serves the HUD + `support.js`
statically from the repo root. **No POST/PUT/PATCH/DELETE exist** — that's the read-only guarantee.

## 4. The `/state` contract (every field → where it comes from)
| Field | Source |
|---|---|
| `account.{balance,equity,day_start_equity,episode_start_equity,peak_equity}` | `AccountState` |
| `ftmo.{daily_loss_limit_pct,max_drawdown_limit_pct,profit_target_pct,daily_target_pct}` | active `FTMOConfig` |
| `position.{dir,symbol,lots,entry,price,age_min}` | the env's live position (lots/age derived) |
| `alphas[].{name,signal,streak,directional}` | the registry + `alpha_matrix`/`streak_matrix` |
| `policy.{action,prob_buy,prob_sell,prob_hold,value,confidence,entropy}` | `PolicyIntrospector.introspect` |
| `policy.{advantage,regime,recommended_lots,expected_dd_pct,value_calibration_pct}` | **gaps** (safe defaults, flagged) |
| `perf.{win_rate_pct,trades,consecutive_losses,day_history}` | `AccountState` + provider ledger |
| `net_signal`, `net_signal_basis` | directional-only consensus + count (never 15) |
| `news`, `human.*` | **gaps** (no scraper / no human layer yet) |
| `clock` | the cached bar timestamp |
| `gaps[]` | every field that fell back to a default |

## 5. The endpoints (all GET, all read-only)
| Endpoint | What it returns |
|---|---|
| `GET /state` | the contract above **+ portfolio fields**: `universe`, `positions[]` (per symbol), `portfolio`, `heatmap` (advances the whole portfolio one bar per poll) |
| `GET /heatmap` | the **full-FTMO-universe buy/sell map** (its own cockpit tab) + portfolio summary |
| `GET /council?use_llm=auto&chat=<json>` | OMEGA→JUSTICE→JARVIS transcript + a grounded, progressive ruling (sees the market + policies + your chat) |
| `GET /policies` | the **policy roster JARVIS organizes**, ranked by FTMO consistency, + the champion |
| `GET /knowledge?q=<text>` | the system summary + the most relevant troubleshooting fixes |
| `GET /ask?q=<text>` | **ask JARVIS how to fix X / which policy to run** — grounded answer + fixes + live posture |
| `GET /health` | `{ok, model_attached, symbols}` |

### Portfolio + policies (the bot trades everything at once)
The bridge is built on a **`MarketView`** — one read-only provider **per FTMO symbol** — so `/state`
and `/heatmap` show the **whole book**, and the council reasons at the **portfolio** level. *(Honest
note: each symbol is its own env today; a true single shared-pot portfolio env is the next env build.)*
**Policies** live in `src/jarvis/policy_registry.py` (persisted JSON). Add one with
`python -m src.jarvis.policy_registry add --id v2 --path models/... --pass-rate 0.88 --fingerprint <fp>`;
JARVIS ranks them by a **consistency score** and `champion()` is the one to run (only same-fingerprint
policies are comparable). Ask him *"which policy should I run?"* and he answers from the roster.

## 6. How to run it
```bash
pip install -r requirements-jarvis.txt          # fastapi + uvicorn (+ optional anthropic)
uvicorn jarvis_bridge:app --port 8000
# then, in another shell or a browser:
curl http://localhost:8000/state
curl "http://localhost:8000/ask?q=I+keep+breaching+the+daily+drawdown"
```
The contract + council run **without** these deps too (`python tools/run_tests.py` proves it). To
enable the conversational LLM layer, set `ANTHROPIC_API_KEY` and `pip install anthropic`.

## 7. Wiring the HUD (the cockpit screen)
The bridge serves whatever HUD file is in the repo root. Drop your **`JARVIS Cockpit.dc.html`** +
**`support.js`** there, then apply the paste-ready patch in **`docs/JARVIS_LIVE_WIRING.md`** (four
methods: `pullLive` → `/state`, `councilLive` → `/council`, plus the net-signal/gate fix so the HUD
uses `net_signal_basis` instead of a hardcoded 15). Open
`http://localhost:8000/JARVIS%20Cockpit.dc.html`.

## 8. Going fully live (two plug-ins)
Today the bridge runs the **real env + real 16 alphas** on **synthetic data with no trained model**
— and says so (`model_attached: false`, fields in `gaps`). To go 100% live:
1. **Real data** — `StateProvider.from_cache(your_cache_dir, "EURUSD")` (build the cache first with
   `cache_builder.build_cache`).
2. **A trained model** — none exists until a training run; attach it:
   `policy = sb3_policy_fn(*load_for_eval(...))` then `StateProvider.from_cache(dir, sym, policy=policy)`.

## 9. Ask JARVIS for fixes (the whole point)
JARVIS **always** has the system map + the fixes in context, so you can ask him directly:
- In the cockpit chat: *"I'm breaching the daily drawdown — how do I fix it?"*
- Or over HTTP: `GET /ask?q=...`

He answers **grounded in the real system** (cites the file to edit, e.g. `config/variables.py`'s
trailing settings) and ends on the next step toward a consistent pass. The same knowledge is in
**`docs/TROUBLESHOOTING.md`** if you'd rather read it. Examples he can fix: *model not learning,
breaching drawdown, not hitting +2.5%/day, gave back the daily target, trades not opening, position
size wrong per asset, model_attached false, cockpit blank, training slow on Colab, eval worse than
training, which model to trust, adding an alpha without retraining.*

## 10. The read-only guarantee (why it's safe)
- The bridge has **GET routes only** — there is no order/place/execute/close function anywhere in
  the JARVIS modules (a test scans for them).
- A real `POST /order` returns **405** (verified live).
- The provider's `step()` only advances a **simulation**; it never sends a broker order.
JARVIS observes, reasons, and advises. He is structurally incapable of trading.

## 11. How to extend
- **Add a troubleshooting fix:** append an entry to `TROUBLESHOOTING` in `src/jarvis/knowledge.py`
  (`{id, area, symptom, cause, fix, refs}`), then regenerate the doc:
  `python -c "from src.jarvis import knowledge as KB; open('docs/TROUBLESHOOTING.md','w').write(KB.render_markdown())"`.
  JARVIS picks it up automatically.
- **Add a `/state` field:** add it to the provider snapshot + `build_state` (default + flag if the
  bot can't produce it). Keep it additive; the HUD ignores unknown keys.
- **Add a council rule/posture:** extend `analyze_consistency()` (the agents will cite it
  automatically — they read from that one analysis).
- **Tune JARVIS's voice:** the personas + the progressive directive live at the top of `council.py`.
