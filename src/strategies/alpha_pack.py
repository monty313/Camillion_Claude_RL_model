"""Registers gravity (0) + pack (1-14) + ORB (15) + 2 movement filters (16-17), canonical order."""
from src.strategies.gravity_30m_4h_alpha import register_gravity_alpha
from src.strategies.register_regime_pulse_trend_5m_30m_alpha import register as _r_regime_pulse_trend_5m_30m_alpha
from src.strategies.register_regime_pulse_pullback_5m_30m_alpha import register as _r_regime_pulse_pullback_5m_30m_alpha
from src.strategies.register_regime_pulse_trend_30m_4h_alpha import register as _r_regime_pulse_trend_30m_4h_alpha
from src.strategies.register_regime_pulse_pullback_30m_4h_alpha import register as _r_regime_pulse_pullback_30m_4h_alpha
from src.strategies.register_cci_surge_trend_5m_30m_alpha import register as _r_cci_surge_trend_5m_30m_alpha
from src.strategies.register_cci_surge_pullback_5m_30m_alpha import register as _r_cci_surge_pullback_5m_30m_alpha
from src.strategies.register_cci_surge_trend_30m_4h_alpha import register as _r_cci_surge_trend_30m_4h_alpha
from src.strategies.register_cci_surge_pullback_30m_4h_alpha import register as _r_cci_surge_pullback_30m_4h_alpha
from src.strategies.register_sma_stack_trend_5m_30m_alpha import register as _r_sma_stack_trend_5m_30m_alpha
from src.strategies.register_sma_stack_pullback_5m_30m_alpha import register as _r_sma_stack_pullback_5m_30m_alpha
from src.strategies.register_sma_stack_trend_30m_4h_alpha import register as _r_sma_stack_trend_30m_4h_alpha
from src.strategies.register_sma_stack_pullback_30m_4h_alpha import register as _r_sma_stack_pullback_30m_4h_alpha
from src.strategies.register_sma_reversion_rally_5m_30m_alpha import register as _r_sma_reversion_rally_5m_30m_alpha
from src.strategies.register_sma_reversion_rally_30m_4h_alpha import register as _r_sma_reversion_rally_30m_4h_alpha
from src.strategies.register_orb_ny_breakout_indices_alpha import register as _r_orb_ny_breakout_indices_alpha
from src.strategies.register_dual_movement_filter_5m_30m_alpha import register as _r_dual_movement_filter_5m_30m_alpha
from src.strategies.register_dual_movement_filter_30m_4h_alpha import register as _r_dual_movement_filter_30m_4h_alpha


def register_all(registry):
    register_gravity_alpha(registry)            # slot 0
    _r_regime_pulse_trend_5m_30m_alpha(registry)
    _r_regime_pulse_pullback_5m_30m_alpha(registry)
    _r_regime_pulse_trend_30m_4h_alpha(registry)
    _r_regime_pulse_pullback_30m_4h_alpha(registry)
    _r_cci_surge_trend_5m_30m_alpha(registry)
    _r_cci_surge_pullback_5m_30m_alpha(registry)
    _r_cci_surge_trend_30m_4h_alpha(registry)
    _r_cci_surge_pullback_30m_4h_alpha(registry)
    _r_sma_stack_trend_5m_30m_alpha(registry)
    _r_sma_stack_pullback_5m_30m_alpha(registry)
    _r_sma_stack_trend_30m_4h_alpha(registry)
    _r_sma_stack_pullback_30m_4h_alpha(registry)
    _r_sma_reversion_rally_5m_30m_alpha(registry)
    _r_sma_reversion_rally_30m_4h_alpha(registry)
    _r_orb_ny_breakout_indices_alpha(registry)   # slot 15: ORB NY-open breakout (INDICES only)
    _r_dual_movement_filter_5m_30m_alpha(registry)   # slot 16: movement filter (5m/30m), 1/0
    _r_dual_movement_filter_30m_4h_alpha(registry)   # slot 17: movement filter (30m/4h), 1/0
    return registry
