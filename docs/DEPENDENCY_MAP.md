# DEPENDENCY MAP

Generated/maintained from the file headers (`DEPENDS_ON` / `USED_BY`); a builder lives
in `src/utils/dependency_utils.py`.

## Layers (bottom → top)
```
config/constants.py            (frozen: contract, indicator specs, slot count)
        │
config/variables.py  ──► config/ftmo_config.py            config/training_speed_config.py
        │                         │
src/indicators/{sma,cci,rsi,bollinger}.py ─► src/indicators/base.py   (190 columns)
        │
src/strategies/{base,registry}.py  ─►  src/signals/{summary,memory,accuracy}.py
        │                                          │
src/account/{account_state,trade_history,win_loss_features}.py        src/risk/{ftmo_rules,free_mode_rules,breach_detector}.py
        └──────────────┬───────────────────────────┘
                       ▼
        src/observation/{observation_contract,builder}.py   ──►  the 357 vector
                       │
        src/env/trading_env.py  (Phase 1)  ─►  src/training/*  (Phase 1)
                       │
        src/barbershop/*   and   src/jarvis/*   (read-only views)
```

## Key contracts
- **Observation shape** is owned by `config/constants.py` + `observation_contract.py`.
- **FTMO/FREE limits** are owned by `config/variables.py` (+ `ftmo_config.py`).
- Indicators feed strategies; strategies + indicators + account feed the observation.
