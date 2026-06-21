# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  Define the OHLCV data shape every loader must return, so live MT5,
#      exported CSV/parquet, and backtest data are interchangeable.
# WHERE src/data/data_contracts.py | HOW a dataclass + required columns.
# DEPENDS_ON dataclasses | USED_BY mt5_loader.py, cache_builder.py (Phase 1).
"""Canonical OHLCV data contract shared by every data source."""
from __future__ import annotations
from dataclasses import dataclass

OHLCV_COLUMNS = ("time", "open", "high", "low", "close", "volume")

@dataclass(frozen=True)
class BarRequest:
    symbol: str
    timeframe: str          # one of config.constants.TIMEFRAMES
    start: str | None = None
    end: str | None = None
    count: int | None = None
