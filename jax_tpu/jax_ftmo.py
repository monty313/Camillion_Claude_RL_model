# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  The FTMO breach + two-phase-banking rules rewritten as BRANCHLESS jnp math
#      so thousands of envs evaluate them in lockstep on a TPU. This is a 1:1
#      port of src/risk/ftmo_rules.py (the CPU reference) — same thresholds, same
#      comparisons, same "any breach = daily OR total OR trailing".
# WHERE jax_tpu/jax_ftmo.py
# HOW   Pure functions over jnp scalars/arrays (broadcast -> vmap-safe). Inputs are
#       FRACTIONS (0.025, 0.04, ...) not percents; the caller converts cfg pct/100.
#       Booleans are returned as 0.0/1.0 float32 (TPU has no bool branching).
# DEPENDS_ON: jax
# USED_BY: jax_tpu/jax_env.py, jax_tpu/tests/test_jax_parity.py
# CHANGE_NOTES(IRAC): I: branchy Python if/else can't run on a TPU in lockstep.
#   R: blueprint Rule 3 (branchless lax.select) + CLAUDE.md "never change FTMO
#   numbers". A: express each ftmo_rules.py check as a jnp comparison; keep the
#   exact thresholds/bases. C: identical breach verdicts to the CPU env, vectorized.
# =====================================================================
"""Branchless FTMO rules (breach + two-phase banking) in jnp — 1:1 with src/risk/ftmo_rules.py."""
from __future__ import annotations
import jax.numpy as jnp


def daily_target_hit(equity, day_start_balance, starting_balance, daily_target_frac):
    """1.0 if the DAY's gain on EQUITY >= daily_target_frac of the INITIAL balance.
    CPU ref (ftmo_rules.daily_target_hit): (equity - day0) >= starting * pct/100."""
    return (jnp.asarray(equity - day_start_balance)
            >= starting_balance * daily_target_frac).astype(jnp.float32)


def daily_drawdown_breached(equity, day_start_balance, daily_dd_frac):
    """CPU ref: (day_start - equity) >= day_start * daily_dd_pct/100."""
    return (jnp.asarray(day_start_balance - equity)
            >= day_start_balance * daily_dd_frac).astype(jnp.float32)


def total_drawdown_breached(equity, starting_balance, total_dd_frac):
    """CPU ref: (starting - equity) >= starting * max_total_drawdown_pct/100."""
    return (jnp.asarray(starting_balance - equity)
            >= starting_balance * total_dd_frac).astype(jnp.float32)


def trailing_breached(equity, peak_equity, trailing_dd_frac, trailing_enabled):
    """CPU ref: trailing_enabled AND (peak - equity) >= peak * trailing_dd_pct/100.
    `trailing_enabled` is a 0.0/1.0 float (or array) so this stays branchless."""
    hit = (jnp.asarray(peak_equity - equity) >= peak_equity * trailing_dd_frac).astype(jnp.float32)
    return hit * jnp.asarray(trailing_enabled, dtype=jnp.float32)


def breach(equity, day_start_balance, starting_balance, peak_equity,
           daily_dd_frac, total_dd_frac, trailing_dd_frac, trailing_enabled):
    """Return dict of 0/1 floats: each breach kind + `any_breach` (= daily OR total OR trailing).
    Mirrors breach_detector.detect()/ftmo_rules.reasons(): ANY non-empty reason => breached."""
    d = daily_drawdown_breached(equity, day_start_balance, daily_dd_frac)
    tot = total_drawdown_breached(equity, starting_balance, total_dd_frac)
    tr = trailing_breached(equity, peak_equity, trailing_dd_frac, trailing_enabled)
    # OR in 0/1 space = max
    any_b = jnp.maximum(jnp.maximum(d, tot), tr)
    return {"daily": d, "total": tot, "trailing": tr, "any_breach": any_b}


def should_auto_flat(equity, day_start_balance, starting_balance,
                     daily_target_frac, two_phase_enabled):
    """CPU ref (ftmo_rules.should_auto_flat): two_phase_enabled AND daily_target_hit.
    `two_phase_enabled` is a 0.0/1.0 float so this stays branchless."""
    return daily_target_hit(equity, day_start_balance, starting_balance, daily_target_frac) \
        * jnp.asarray(two_phase_enabled, dtype=jnp.float32)
