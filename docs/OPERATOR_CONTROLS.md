# OPERATOR CONTROLS

Runtime-editable settings. No retrain required — observation uses percentages.

---

## 1. Starting Account Size

**File:** `config/variables.py`

```python
from config import variables as V
V.STARTING_BALANCE = 50_000.0  # $50k challenge
```

---

## 2. Daily Target

**File:** `config/variables.py` or via `update_risk_settings()`

```python
from config.ftmo_config import update_risk_settings

# FTMO mode: change to +3%/day
cfg = update_risk_settings(daily_target_pct=3.0)

# FREE mode: custom target
cfg = update_risk_settings(mode="FREE", daily_target_pct=1.5)
```

**Default:** 2.5%/day

---

## 3. Daily Risk / Drawdown

**File:** `config/variables.py` or via `update_risk_settings()`

```python
from config.ftmo_config import update_risk_settings

# Change daily drawdown limit
cfg = update_risk_settings(daily_drawdown_pct=3.0)

# Disable trailing drawdown
cfg = update_risk_settings(trailing_enabled=False)

# Change trailing wall
cfg = update_risk_settings(trailing_pct=6.0)
```

**Defaults (FTMO mode):**
- Daily drawdown: 5%
- Total drawdown: 10%
- Trailing drawdown: 4%
- Trailing enabled: True

---

## 4. Lot Sizing

### Current State: Phase 1 — Fixed Lots

| Aspect | Status |
|--------|--------|
| Action space | 4 discrete: {HOLD, BUY, SELL, CLOSE} |
| Size | Fixed `position_size=100000.0` (default) |
| Policy control | None — set externally |

**Where to set:**
- `TradingEnv(position_size=...)`
- Notebook: `POSITION_SIZE = 100000.0`

### Future Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Fixed lot sizing | ✅ Current |
| Phase 2 | Bucketed lot sizing (SMALL/MEDIUM/LARGE) | Future, approval-gated |
| Phase 3 | Learned/free lot sizing | Future, approval-gated |

**Phase 2/3 require:** Action space expansion, policy architecture changes, possible reward redesign. Not for baseline.

---

## Quick Reference

| Setting | Where | How |
|---------|-------|-----|
| Starting balance | `config/variables.py` | `V.STARTING_BALANCE = ...` |
| Daily target | `config/ftmo_config.py` | `update_risk_settings(daily_target_pct=...)` |
| Daily drawdown | `config/ftmo_config.py` | `update_risk_settings(daily_drawdown_pct=...)` |
| Trailing DD | `config/ftmo_config.py` | `update_risk_settings(trailing_pct=..., trailing_enabled=...)` |
| Mode | `config/ftmo_config.py` | `update_risk_settings(mode="FTMO" or "FREE")` |
| Position size | `TradingEnv` or notebook | `position_size=...` |

---

**All settings are runtime-editable. No retrain required.**
