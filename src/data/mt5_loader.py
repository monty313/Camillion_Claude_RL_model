# WHEN 2026-06-21 (Phase 0 STUB) | WHO Claude for Monty
# WHY  Interface for pulling MT5 bars (live or exported). NEVER called inside
#      the training hot loop (training-speed rule).
# WHERE src/data/mt5_loader.py | HOW Phase-1 wires MetaTrader5 / file readers.
# DEPENDS_ON src/data/data_contracts.py | USED_BY cache_builder.py (Phase 1).
"""MT5 data loader interface (Phase-0 placeholder)."""
from __future__ import annotations
from src.data.data_contracts import BarRequest

def load_bars(req: BarRequest):
    """Return an OHLCV table for req. PHASE-0 STUB."""
    raise NotImplementedError("Phase 1: wire MetaTrader5 / CSV / parquet readers.")
