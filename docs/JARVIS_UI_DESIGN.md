# JARVIS // FTMO QUANT COMMAND — UI design

Streamlit, clickable, modular. Phase-0 ships an importable shell
(`src/jarvis/app.py`); Phase 2 builds the full interactive dashboard.

## Top bar
Equity · Day P/L · Daily Target progress · Trailing DD used · Server clock ·
**RUN DIAGNOSTIC** · **VOICE** · **HELP**.

## Center — clickable workflow graph (`src/jarvis/workflow_map.py`)
`01 Data` · `02 Universe` · `03 Alphas` · `04 Alpha Combination` · `R Risk Model` ·
`05 Portfolio Construction` · `O Objective` · `C Constraints` · `06 Trading`.
Clicking a module shows: what it does, inputs, outputs, files used, dependent files,
current status, related tests (all already encoded in `workflow_map.MODULES`).

## Right panel — AGENT COMMS (IRAC LOOP)  (`src/jarvis/agent_bus.py`)
- **OMEGA** — watches pipeline I/O, finds bugs, checks consistency
  (field of view: 01–05; uses Feature Doctor).
- **JUSTICE** — checks risk, FTMO constraints, pass/fail
  (field of view: R, O, C, 06; uses the breach detector).
- **JARVIS** — final judge; synthesizes OMEGA + JUSTICE into one verdict.
Buttons: **CLEAR**, **NEW IRAC CYCLE**. Local mock bus — no paid API keys.

Run it: `streamlit run src/jarvis/app.py`.
