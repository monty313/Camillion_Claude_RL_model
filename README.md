# Camillion_Claude_RL_model

## Project mission
Develop and operate a trading/market intelligence system focused on Camillion Claude RL model.

## What this repository currently contains
Top-level structure:
- `.gitignore` (file)
- `0_JARVIS_COCKPIT.html` (file)
- `audit_results` (dir)
- `CLAUDE.md` (file)
- `config` (dir)
- `data` (dir)
- `docs` (dir)
- `go_live.py` (file)
- `HANDOFF.md` (file)
- `HANDOFF_IRAC.md` (file)
- `HANDOFF_TRAINING_JARVIS.md` (file)
- `jarvis_bridge.py` (file)
- `jax_tpu` (dir)
- `notebooks` (dir)
- `pyproject.toml` (file)
- `README.md` (file)
- `records` (dir)
- `requirements-jarvis.txt` (file)
- `run_training.py` (file)
- `src` (dir)

Primary languages/tooling detected: Python (206), Markdown (36), Jupyter Notebook (4)

Likely key implementation files:
- `README.md`
- `src/__init__.py`
- `src/account/__init__.py`
- `src/account/account_state.py`
- `src/account/trade_history.py`
- `src/account/win_loss_features.py`
- `src/barbershop/__init__.py`
- `src/barbershop/alpha_edge.py`
- `src/barbershop/day_replay.py`
- `src/barbershop/feature_doctor.py`

## Quickstart (operator + LLM friendly)
Use this sequence first, then adapt to repo-specific details.

1. Clone and enter repo:
   ```bash
   git clone https://github.com/monty313/Camillion_Claude_RL_model.git
   cd Camillion_Claude_RL_model
   ```
2. Setup environment/dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. Run locally:
   ```bash
   python main.py  # adjust to project entrypoint
   ```
4. Validate changes:
   ```bash
   pytest -q
   ```

## Operating intent (CEO-mode guide for future LLMs)
When continuing this project, optimize for business outcome first:

- Preserve or improve strategy quality, reliability, and reproducibility.
- Prefer small, testable, reversible changes over broad rewrites.
- Keep docs synchronized with code behavior.
- Add explicit assumptions, constraints, and success criteria in every major change.

## Delivery checklist
- [ ] Environment setup is deterministic.
- [ ] Primary run path is documented and works.
- [ ] Validation/test path is documented and works.
- [ ] Key risks and TODOs are tracked in the handoff file.

## Branch and release notes
- Default branch: `main`
- Keep PRs focused and attach proof-of-run or proof-of-test for behavior changes.
