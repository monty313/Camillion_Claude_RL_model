# Registration helper for CciSurgePullback30m4hAlpha (auto-assigns the next free slot).
from src.strategies.cci_surge_pullback_30m_4h_alpha import CciSurgePullback30m4hAlpha


def register(registry) -> int:
    return registry.register(CciSurgePullback30m4hAlpha())
