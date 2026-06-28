# =====================================================================
# WHEN 2026-06-26 (Phase 1/2) | WHO Claude for Monty
# WHY  Show training/eval results the way Monty judges the FTMO challenge: DAY BY DAY,
#      did the bot make the +2.5%-of-INITIAL daily target, and did it stay inside the
#      4% trailing-drawdown wall (and the 5% daily / 10% total hard lines)? One table
#      per run + a pass/fail summary, so you SEE consistency, not just a final number.
# WHERE src/training/daily_report.py
# HOW  Step an env with a policy (or HOLD if no model), snapshot per-day P&L %, target
#      hit, peak-to-trough trailing DD used, daily-loss used, and breaches. Read-only.
# DEPENDS_ON: src/env/trading_env.py, src/risk/breach_detector.py, src/interpret/policy_introspector.py
# USED_BY: notebooks (post-train report), `python -m src.training.daily_report`, eval
# CHANGE_NOTES(IRAC): I: a final equity number hides whether the bot passes CONSISTENTLY.
#   R: operator 2026-06-26 -- "see the results relative to passing +2.5% of initial and
#   avoiding the trailing DD, day by day". A: per-day table (target hit + trailing DD used)
#   + a summary (days passed, days within trailing, breaches, final %, challenge pass). C:
#   you read consistency at a glance and can trust which policy to run.
# =====================================================================
"""Day-by-day FTMO report: per-day +2.5% target + 4% trailing-DD status for a run."""
from __future__ import annotations
import numpy as np


def _datestr(d) -> str:
    try:
        return str(np.datetime_as_string(d, unit="D"))
    except Exception:
        return str(d)[:10]


def _greedy_action(env, policy) -> int:
    if policy is None:
        from config import constants as C
        return C.ACTION_HOLD
    from src.interpret.policy_introspector import introspect
    return int(introspect(policy, env._obs(), ablate=False, bar_index=int(env.ptr)).chosen_action)


def daily_report(env, policy=None, max_days: int | None = None):
    """Run env (with `policy`, else HOLD) and return (rows, summary).

    Each row = one trading day: P&L % of INITIAL, whether it hit the +daily-target%, the MAX TRAILING
    drawdown reached that day (chronological drawdown from that day's running peak), the daily-loss used,
    any breach, and cumulative %. Read-only: it never sends a real order.

    DIAGNOSTIC walk-forward: a breach does NOT stop the report -- it keeps walking so you SEE every day and
    can watch the policy improve across checkpoints (only a +10% PASS, end-of-data, or max_days stops it).
    `summary` also carries `action_mix` (chosen-action tally) and `passed_plus10`.
    """
    env.reset()
    cfg = env.cfg
    init = float(cfg.starting_balance)
    target = float(cfg.daily_target_pct)                                  # +2.5% of initial
    trail = float(getattr(cfg, "trailing_drawdown_pct", 4.0))             # 4% trailing wall
    rows: list[dict] = []
    day_start_eq = float(env.acc.equity)
    run_peak = float(env.acc.equity)        # EPISODE running peak (persists across days) -> matches the engine
    day_max_trail = 0.0                       # worst trailing drawdown % (from the running peak) seen THIS day
    day_max_loss = 0.0                        # worst daily-loss % (from this day's start) seen THIS day
    breached_day = False

    def _record(end_eq, date):
        day_pnl_pct = (end_eq - day_start_eq) / init * 100.0
        rows.append({
            "day": len(rows) + 1, "date": _datestr(date),
            "day_pnl_pct": round(day_pnl_pct, 2),
            "passed_target": bool(day_pnl_pct >= target),                 # made +2.5% of initial?
            "trailing_dd_pct": round(day_max_trail, 2),                   # chronological DD from the running peak
            "within_trailing": bool(day_max_trail < trail),              # stayed inside the 4% wall?
            "daily_loss_pct": round(day_max_loss, 2),
            "breached": bool(breached_day),
            "cum_pnl_pct": round((end_eq - init) / init * 100.0, 2),
        })

    # The PortfolioEnv decides ONE symbol per step, so it takes len(symbols) steps to advance ONE bar.
    # The guard below must count those sub-steps or it trips ~1/len(symbols) of the way in (it once cut
    # the portfolio report off at ~30% of the data, even showing 0 days). TradingEnv -> steps_per_bar=1.
    steps_per_bar = max(1, len(getattr(env, "symbols", [None])))
    guard = 0
    action_mix: dict = {}                     # diagnostic: tally the chosen actions (HOLD/BUY/SELL/CLOSE)
    passed = False                            # +10% challenge PASS reached (a WIN -> stop the walk)
    while True:
        guard += 1
        prev_date = env._dates[env.ptr]
        eq_before = float(env.acc.equity)
        # TRAILING drawdown from the RUNNING peak, chronologically. DIAGNOSTIC report: the peak is reset at the
        # start of each NEW day (below), so each day's trailing-DD is measured FRESH from that day's start --
        # this is what lets the walk CONTINUE past a breach and show every day. (The real cross-day trailing
        # wall is still enforced inside env.step(); that is what a live challenge / training uses.)
        run_peak = max(run_peak, eq_before)
        if run_peak > 0:
            day_max_trail = max(day_max_trail, (run_peak - eq_before) / run_peak * 100.0)
        if day_start_eq > 0:
            day_max_loss = max(day_max_loss, (day_start_eq - eq_before) / day_start_eq * 100.0)
        a = _greedy_action(env, policy)
        action_mix[a] = action_mix.get(a, 0) + 1
        env.step(a)
        if day_max_trail >= trail:                                       # this DAY crossed the 4% wall
            breached_day = True
        passed = passed or bool(env.acc.episode_passed)   # tracked for the summary; no longer stops the walk
        at_end = env.ptr >= env.T - 1
        crossed = env._dates[env.ptr] != prev_date
        if crossed or at_end:
            _record(eq_before if crossed else float(env.acc.equity), prev_date)
            # KEEP WALKING through a breach AND past a +10% pass: a single bad day (or a single win) no longer
            # ENDS the whole report, so you SEE every day and the LONGEST run of winning days. Only end-of-data
            # or max_days stops the walk (operator 2026-06-28). On a new day, reset the per-day peak/loss fresh.
            if at_end or (max_days and len(rows) >= max_days):
                break
            # FAIL -> START OVER next day: if the day just BREACHED, reset to a FRESH challenge attempt (don't
            # carry a dead/breached account into the next day). Matches training + the won-day-streak eval.
            if breached_day and hasattr(env, "restart_account"):
                env.restart_account()
            day_start_eq = float(env.acc.equity)                         # start a fresh day (loss resets)
            run_peak = float(env.acc.equity)                             # fresh per-day trailing peak (diagnostic)
            day_max_trail = 0.0
            day_max_loss = 0.0
            breached_day = False
        if guard > env.T * steps_per_bar + 16:                            # safety net (scaled by sub-steps)
            break
    summary = _summarize(rows, cfg)
    summary["action_mix"] = action_mix                                   # diagnostic: trading, or HOLD-collapse?
    summary["passed_plus10"] = bool(passed)
    return rows, summary


def running_drawdown_pct(equities) -> float:
    """Max trailing drawdown % from the RUNNING peak over a chronological equity path. This is the correct
    'trailing drawdown' (drawdown from the highest equity SO FAR) -- NOT (max - min), which pairs a later
    peak with an earlier trough and overstates the drawdown."""
    peak = None
    mdd = 0.0
    for e in equities:
        e = float(e)
        peak = e if peak is None else max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak * 100.0)
    return mdd


def _summarize(rows, cfg) -> dict:
    n = len(rows)
    passed = sum(r["passed_target"] for r in rows)
    within = sum(r["within_trailing"] for r in rows)
    breaches = sum(r["breached"] for r in rows)
    final = rows[-1]["cum_pnl_pct"] if rows else 0.0
    pass_target = float(getattr(cfg, "profit_target_total_pct", 10.0))
    return {
        "days": n, "days_passed_target": passed, "days_within_trailing": within,
        "breaches": breaches, "final_cum_pct": round(final, 2),
        "daily_target_pct": float(cfg.daily_target_pct),
        "trailing_pct": float(getattr(cfg, "trailing_drawdown_pct", 4.0)),
        "challenge_target_pct": pass_target,
        "challenge_passed": bool(final >= pass_target and breaches == 0),
    }


def format_action_mix(action_mix) -> str:
    """Format a chosen-action tally as 'HOLD 70% · BUY 12% · SELL 10% · CLOSE 8%' (HOLD-collapse check)."""
    if not action_mix:
        return ""
    from config import constants as C
    names = {C.ACTION_HOLD: "HOLD", C.ACTION_BUY: "BUY", C.ACTION_SELL: "SELL", C.ACTION_CLOSE: "CLOSE"}
    tot = max(1, sum(action_mix.values()))
    return " · ".join(f"{names.get(a, a)} {100.0 * n / tot:.0f}%" for a, n in sorted(action_mix.items()))


def format_daily_report(rows, summary) -> str:
    """A plain, aligned table + the FTMO pass/fail summary."""
    out = [f"DAY-BY-DAY FTMO REPORT  (daily target +{summary['daily_target_pct']:.1f}% of initial | "
           f"trailing wall {summary['trailing_pct']:.1f}%)",
           f"{'DAY':>3}  {'DATE':<10}  {'P&L%':>7}  {'+TGT?':>6}  {'TRAIL_DD%':>9}  {'<WALL?':>6}  "
           f"{'DAILY_LOSS%':>11}  {'BREACH':>6}  {'CUM%':>7}",
           "-" * 84]
    for r in rows:
        out.append(f"{r['day']:>3}  {r['date']:<10}  {r['day_pnl_pct']:>7.2f}  "
                   f"{('YES' if r['passed_target'] else 'no'):>6}  {r['trailing_dd_pct']:>9.2f}  "
                   f"{('ok' if r['within_trailing'] else 'BREACH'):>6}  {r['daily_loss_pct']:>11.2f}  "
                   f"{('YES' if r['breached'] else 'no'):>6}  {r['cum_pnl_pct']:>7.2f}")
    out += ["-" * 84,
            f"SUMMARY: {summary['days']} days | hit +{summary['daily_target_pct']:.1f}%: "
            f"{summary['days_passed_target']}/{summary['days']} | within {summary['trailing_pct']:.1f}% trailing: "
            f"{summary['days_within_trailing']}/{summary['days']} | breaches: {summary['breaches']} | "
            f"final {summary['final_cum_pct']:+.2f}% | CHALLENGE "
            f"{'PASSED' if summary['challenge_passed'] else 'not yet (need +' + str(summary['challenge_target_pct']) + '% with 0 breaches)'}"]
    mix = format_action_mix(summary.get("action_mix"))
    if mix:
        out.append(f"ACTION-MIX: {mix}")
    return "\n".join(out)


def run_daily_report(data_dir: str | None = None, symbol: str = "EURUSD",
                     model_path: str | None = None, n_synth: int = 4320, max_days: int | None = None):
    """Build an env (real cache if data_dir, else synthetic ~3 days) + optional model, print the report."""
    max_days = 30 if max_days is None else max_days   # report now WALKS THROUGH breaches -> cap the printout
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    from src.env.trading_env import TradingEnv
    reg = AlphaRegistry(); register_all(reg)
    if data_dir:
        from src.data.cache_builder import load_cache
        ind, close, time_ns = load_cache(data_dir, symbol)
        env = TradingEnv(np.asarray(ind), np.asarray(close), np.asarray(time_ns), reg, symbol=symbol)
    else:
        import pandas as pd
        rng = np.random.default_rng(0)
        close = (100.0 + np.cumsum(rng.standard_normal(n_synth) * 0.04)).astype(np.float32)
        ind = np.zeros((n_synth, __import__("config.constants", fromlist=["N_INDICATORS_TOTAL"]).N_INDICATORS_TOTAL), np.float32)
        idx = pd.date_range("2026-03-02 00:00", periods=n_synth, freq="1min")
        env = TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64), reg, symbol=symbol)
    policy = None
    if model_path:
        try:    # best-effort: needs SB3 + the saved model + its vecnorm
            from src.data.cache_builder import load_cache
            from src.training.trainer import load_for_eval, sb3_policy_fn
            ind, close, time_ns = load_cache(data_dir, symbol)
            model, venv = load_for_eval(model_path, ind, close, time_ns,
                                        lambda: _fresh_reg(), symbol=symbol)
            policy = sb3_policy_fn(model, venv)
        except Exception as e:  # pragma: no cover
            print(f"[daily_report] no model ({e}); reporting the HOLD baseline (real per-day comes with a trained policy).")
    rows, summary = daily_report(env, policy=policy, max_days=max_days)
    print(format_daily_report(rows, summary))
    return rows, summary


def run_portfolio_report(data_dir: str, symbols, model_path: str | None = None, max_days: int | None = None):
    """Day-by-day on the SHARED-POT PortfolioEnv: ALL symbols, ONE account (the true portfolio view).

    Requires the per-symbol caches to be time-aligned (same bars)."""
    max_days = 30 if max_days is None else max_days   # report now WALKS THROUGH breaches -> cap the printout
    from src.data.cache_builder import load_cache
    from src.env.portfolio_env import PortfolioEnv, align_symbol_data
    syms = list(symbols)
    sd = align_symbol_data({s: load_cache(data_dir, s) for s in syms})   # keep only shared bars
    env = PortfolioEnv(sd, _fresh_reg)
    policy = None
    if model_path:
        try:  # pragma: no cover - needs SB3 + a saved portfolio model
            from src.training.trainer import load_for_eval, sb3_policy_fn
            ind, close, time_ns = load_cache(data_dir, syms[0])     # obs is identical across symbols
            model, venv = load_for_eval(model_path, ind, close, time_ns, _fresh_reg, symbol=syms[0])
            policy = sb3_policy_fn(model, venv)
        except Exception as e:
            print(f"[daily_report] no portfolio model ({e}); HOLD baseline.")
    rows, summary = daily_report(env, policy=policy, max_days=max_days)
    print(format_daily_report(rows, summary))
    return rows, summary


def _fresh_reg():
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    r = AlphaRegistry(); register_all(r); return r


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Day-by-day FTMO report (+2.5% target + trailing DD).")
    ap.add_argument("--data", default=None)
    ap.add_argument("--symbol", default="EURUSD", help="single-symbol report")
    ap.add_argument("--portfolio", default=None, help="comma-separated symbols -> ONE shared-pot report")
    ap.add_argument("--model", default=None); ap.add_argument("--max-days", type=int, default=None)
    a = ap.parse_args()
    if a.portfolio:
        run_portfolio_report(a.data, [s.strip() for s in a.portfolio.split(",") if s.strip()],
                             a.model, max_days=a.max_days)
    else:
        run_daily_report(a.data, a.symbol, a.model, max_days=a.max_days)
