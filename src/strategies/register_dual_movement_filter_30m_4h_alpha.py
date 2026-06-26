# Registration helper for DualMovementFilter30m4hAlpha (auto-assigns the next free slot).
from src.strategies.dual_movement_filter_30m_4h_alpha import DualMovementFilter30m4hAlpha


def register(registry) -> int:
    return registry.register(DualMovementFilter30m4hAlpha())
