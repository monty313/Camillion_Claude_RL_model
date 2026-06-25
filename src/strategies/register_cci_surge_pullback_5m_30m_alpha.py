# Registration helper for CciSurgePullback5m30mAlpha (auto-assigns the next free slot).
from src.strategies.cci_surge_pullback_5m_30m_alpha import CciSurgePullback5m30mAlpha


def register(registry) -> int:
    return registry.register(CciSurgePullback5m30mAlpha())
