# FTMO & FREE MODE

Two modes, selected by `config/variables.py: MODE` (`"FTMO"` or `"FREE"`).

## FTMO mode (defaults preserved from Quantra)
- **+2.5% / day** target
- **5%** daily loss limit, **10%** total loss limit
- **4% trailing wall** (trailing drawdown), can be toggled
- **Two-phase episode:** hit **+2.5% → auto-flat all → fresh 1% trailing**
- Pass/fail + breach detection (`src/risk/breach_detector.py`)

## FREE mode
You set everything: target, daily/total drawdown, trailing amount, trailing on/off.

## Editable at RUNTIME — no retrain (important)
Target, trailing-DD amount, and trailing on/off are editable in **both** modes via
`config/variables.py` or live:

```python
from config.ftmo_config import update_risk_settings
update_risk_settings(daily_target_pct=3.0, trailing_pct=6.0, trailing_enabled=False)  # FTMO
update_risk_settings(mode="FREE", daily_target_pct=1.5, trailing_pct=2.0)              # FREE
```

**Why no retrain is needed:** the observation exposes these only as *percentages*
(progress-to-target, % of drawdown budget used/remaining). Those fractions keep the
same meaning when the absolute numbers change, so the same trained policy still reads
them correctly. These knobs live in `variables.py` (runtime) — never in
`constants.py` (which would change the observation contract and require a retrain).

## Where it lives
`config/ftmo_config.py` (configs + `update_risk_settings`), `config/variables.py`
(editable knobs), `src/risk/{ftmo_rules,free_mode_rules,breach_detector}.py` (checks),
`src/account/win_loss_features.py` (fraction features read from the active config).
