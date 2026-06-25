# Registration helper for CciSurgeTrend5m30mAlpha (auto-assigns the next free slot).
from src.strategies.cci_surge_trend_5m_30m_alpha import CciSurgeTrend5m30mAlpha


def register(registry) -> int:
    return registry.register(CciSurgeTrend5m30mAlpha())
