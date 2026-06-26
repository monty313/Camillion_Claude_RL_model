# Registration helper for DualMovementFilter5m30mAlpha (auto-assigns the next free slot).
from src.strategies.dual_movement_filter_5m_30m_alpha import DualMovementFilter5m30mAlpha


def register(registry) -> int:
    return registry.register(DualMovementFilter5m30mAlpha())
