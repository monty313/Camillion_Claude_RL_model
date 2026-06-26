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

    Each row = one trading day: P&L % of INITIAL, whether it hit the +daily-target%, the
    peak-to-trough TRAILING DD used (wall = trailing_drawdown_pct), the daily-loss used, any
    breach, and cumulative %. Read-only: it never sends a real order.
    """
    env.reset()
    cfg = env.cfg
    init = float(cfg.starting_balance)
    target = float(cfg.daily_target_pct)                                  # +2.5% of initial
    trail = float(getattr(cfg, "trailing_drawdown_pct", 4.0))             # 4% trailing wall
    rows: list[dict] = []
    day_start_eq = peak = trough = float(env.acc.equity)
    breached_day = False

    def _record(end_eq, date):
        day_pnl_pct = (end_eq - day_start_eq) / init * 100.0
        trailing_used = (peak - trough) / init * 100.0                    # peak->trough that day
        daily_loss_used = max(0.0, day_start_eq - trough) / init * 100.0
        rows.append({
            "day": len(rows) + 1, "date": _datestr(date),
            "day_pnl_pct": round(day_pnl_pct, 2),
            "passed_target": bool(day_pnl_pct >= target),                 # made +2.5% of initial?
            "trailing_dd_pct": round(trailing_used, 2),
            "within_trailing": bool(trailing_used < trail),               # stayed inside the 4% wall?
            "daily_loss_pct": round(daily_loss_used, 2),
            "breached": bool(breached_day),
            "cum_pnl_pct": round((end_eq - init) / init * 100.0, 2),
        })

    guard = 0
    while True:
        guard += 1
        prev_date = env._dates[env.ptr]
        eq_before = float(env.acc.equity)
        peak, trough = max(peak, eq_before), min(trough, eq_before)
        env.step(_greedy_action(env, policy))
        if env.acc.episode_breached:
            breached_day = True
        terminated = bool(env.acc.episode_breached or env.acc.episode_passed)
        at_end = env.ptr >= env.T - 1
        crossed = env._dates[env.ptr] != prev_date
        if crossed or terminated or at_end:
            _record(eq_before if crossed else float(env.acc.equity), prev_date)
            if terminated or at_end or (max_days and len(rows) >= max_days):
                break
            day_start_eq = peak = trough = float(env.acc.equity)         # start a fresh day
            breached_day = False
        if guard > env.T + 5:
            break
    return rows, _summarize(rows, cfg)


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
    return "\n".join(out)


def run_daily_report(data_dir: str | None = None, symbol: str = "EURUSD",
                     model_path: str | None = None, n_synth: int = 4320, max_days: int | None = None):
    """Build an env (real cache if data_dir, else synthetic ~3 days) + optional model, print the report."""
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
