# Registration helper for CciSurgeTrend30m4hAlpha (auto-assigns the next free slot).
from src.strategies.cci_surge_trend_30m_4h_alpha import CciSurgeTrend30m4hAlpha


def register(registry) -> int:
    return registry.register(CciSurgeTrend30m4hAlpha())
