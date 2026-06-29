"""Registers the canonical alpha pack: slots 0-17 (gravity, regime-pulse, cci-surge, sma-stack,
sma-reversion, ORB, and the two ADX-DI alignment alphas) in fixed order."""
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
from src.strategies.register_adx_di_align_5m_30m_alpha import register as _r_adx_di_align_5m_30m_alpha
from src.strategies.register_adx_di_align_30m_4h_alpha import register as _r_adx_di_align_30m_4h_alpha
# operator 2026-06-29: 3 strong-setup alphas (slots 18/19/20) — CCI |>160| on 5m+30m, BB200&BB20(dev1)
# double-breakout on any TF, forward-displaced SMA(4) alignment on 5m+30m. Fill free slots (no obs change).
from src.strategies.register_cci_x160_align_5m_30m_alpha import register as _r_cci_x160_align_5m_30m_alpha
from src.strategies.register_bb_double_breakout_anytf_alpha import register as _r_bb_double_breakout_anytf_alpha
from src.strategies.register_fwd_sma4_align_5m_30m_alpha import register as _r_fwd_sma4_align_5m_30m_alpha


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
    _r_adx_di_align_5m_30m_alpha(registry)        # slot 16: ADX-DI alignment (5m & 30m, periods 14&45)
    _r_adx_di_align_30m_4h_alpha(registry)        # slot 17: ADX-DI alignment (30m & 4h, periods 14&45)
    _r_cci_x160_align_5m_30m_alpha(registry)      # slot 18: CCI |>160| on 5m AND 30m (operator 2026-06-29)
    _r_bb_double_breakout_anytf_alpha(registry)   # slot 19: price beyond BB200 & BB20 (dev1) on ANY TF
    _r_fwd_sma4_align_5m_30m_alpha(registry)      # slot 20: price vs forward-SMA(4) on 5m AND 30m
    return registry


# --- v1.7.0 CONVICTION bonus support (operator 2026-06-29) -------------------------------------------------
# The 3 "strong-setup" alphas above. The PortfolioEnv conviction_bonus pays (PnL-capped) when >=2 of these
# CONFIRMED the trade's direction AT ENTRY and the trade closes in profit (day net up). Both the CPU env and
# the JAX env read these EXACT slots from the per-symbol alpha matrix (no new precompute). CONVICTION_SLOTS is
# the canonical register order; a test asserts it equals the resolved slots so a future reorder can't drift.
CONVICTION_ALPHA_NAMES = ("cci_x160_align_5m_30m", "bb_double_breakout_anytf", "fwd_sma4_align_5m_30m")
CONVICTION_SLOTS = (18, 19, 20)
# SELECTIVITY (operator 2026-06-29): the conviction reward scales with the NUMBER of firing alphas pointing
# the SAME way as the trade (the "greatest amount of signals in one direction"), gated on the bot trading WITH
# the consensus (entry agreed with the majority). Saturates at this many aligned signals (so a strong,
# lopsided stack pays the full bonus; a thin one pays a fraction). Read by BOTH the CPU and JAX envs.
CONVICTION_ALIGN_CAP = 8.0


def slot_of(name: str, registry=None) -> int:
    """Slot index of an alpha by name (canonical pack if no registry given); -1 if absent."""
    from src.strategies.registry import AlphaRegistry
    if registry is None:
        registry = AlphaRegistry(); register_all(registry)
    for i, s in enumerate(registry._slots):
        if s is not None and getattr(s, "name", None) == name:
            return i
    return -1


def conviction_slots(registry=None) -> tuple:
    """The canonical slot indices of the 3 conviction alphas (resolved by name)."""
    return tuple(slot_of(n, registry) for n in CONVICTION_ALPHA_NAMES)
