#!/usr/bin/env python
# =====================================================================
# WHEN 2026-06-26 (Phase 2) | WHO Claude for Monty
# WHY  Train the Camillion PORTFOLIO bot in ONE command -- for someone who does NOT
#      trade and does not want a pile of steps. Point it at your data folder; it finds
#      your files, prepares the market features, trains ONE bot on the whole book from
#      one shared pot, then shows the DAY-BY-DAY results (+2.5% target + 4% trailing DD)
#      and registers the policy so JARVIS can organize it. Plain-language output, no jargon.
# WHERE run_training.py (repo root)
# HOW  python run_training.py --data /content/drive/MyDrive/Camillion_data
# DEPENDS_ON: src/data/cache_builder, src/training/{trainer,daily_report}, src/env/portfolio_env,
#             src/strategies/*, src/jarvis/policy_registry  (+ stable-baselines3 ONLY to actually train)
# USED_BY: the operator (Monty); the one-click Colab notebook calls this in a single cell.
# CHANGE_NOTES(IRAC): I: training was several manual steps -> easy to get lost. R: operator
#   2026-06-26 -- make it easy + intuitive for a non-trader, one flow, no confusion. A: a single
#   entry that auto-finds the data, builds caches, trains the shared-pot portfolio, prints the
#   day-by-day pass/DD table, and registers the policy -- with friendly errors. C: one command,
#   clear results, toward a consistent portfolio pass.
# =====================================================================
"""Train the Camillion portfolio bot in ONE command (data folder in -> day-by-day results out)."""
from __future__ import annotations
import argparse
import glob
import os
import sys

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "US30"]


def _find_csv(folder: str, symbol: str):
    """First CSV in `folder` whose filename contains the symbol (case-insensitive)."""
    for f in sorted(glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.CSV"))):
        if symbol.lower() in os.path.basename(f).lower():
            return f
    return None


def prepare_caches(data_dir: str, symbols, cache_dir: str = "data_cache", date_from=None, date_to=None):
    """Find each symbol's CSV and build its market-feature cache. Optionally restrict to a date range
    (date_from/date_to, e.g. '2024-01-01') for a fast first run. Returns the symbols it prepared."""
    import pandas as _pd
    from src.data.cache_builder import load_ohlcv_csv, build_cache
    found = []
    for s in symbols:
        f = _find_csv(data_dir, s)
        if not f:
            print(f"      (skipped {s} — no CSV with that name in {data_dir})")
            continue
        df = load_ohlcv_csv(f)
        if date_from:
            df = df[df.index >= _pd.Timestamp(date_from)]
        if date_to:
            df = df[df.index < _pd.Timestamp(date_to) + _pd.Timedelta(days=1)]   # inclusive of the end day
        if len(df) == 0:
            print(f"      (skipped {s} — no rows in the chosen date range)")
            continue
        build_cache(df, out_dir=cache_dir, symbol=s)
        print(f"      {s} ready  ({os.path.basename(f)}, {len(df):,} bars)")
        found.append(s)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Camillion portfolio bot in one command.")
    ap.add_argument("--data", required=True, help="folder with your 1-minute CSVs (one per symbol)")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated (default: the 4 majors)")
    ap.add_argument("--steps", type=int, default=2_000_000, help="how long to train (more = better but slower)")
    ap.add_argument("--out", default="models/camillion_portfolio_ppo", help="where to save the trained bot")
    ap.add_argument("--cache", default="data_cache", help="where to store the prepared features")
    ap.add_argument("--from", dest="date_from", default=None,
                    help="only use data on/after this date, e.g. 2024-01-01 (use for a FAST first run)")
    ap.add_argument("--to", dest="date_to", default=None, help="only use data up to this date, e.g. 2024-06-30")
    ap.add_argument("--feature-cache", dest="feature_cache", default=None,
                    help="where to SAVE the prepared features so re-runs skip the slow build (default: auto -> "
                         "Google Drive MyDrive/Camillion/feature_cache on Colab, else a local folder). "
                         "Use 'off' to disable.")
    a = ap.parse_args(argv)
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]

    line = "=" * 72
    print(f"{line}\n  CAMILLION — training ONE bot to trade your whole portfolio\n{line}")

    # 1) find the data
    print(f"\n[1/5] Looking for your 1-minute data in: {a.data}")
    if not os.path.isdir(a.data):
        sys.exit(f"\n  That folder doesn't exist: {a.data}\n  On Colab, mount Drive first and use a path like "
                 f"/content/drive/MyDrive/Camillion_data")

    # 2) prepare the market features
    rng = ""
    if a.date_from or a.date_to:
        rng = f"  (date range: {a.date_from or 'start'} -> {a.date_to or 'end'})"
    print(f"\n[2/5] Preparing market features (one-time)...{rng}")
    found = prepare_caches(a.data, symbols, a.cache, a.date_from, a.date_to)
    if not found:
        sys.exit(f"\n  No matching CSVs in {a.data}. Put files there whose names contain the symbol "
                 f"(e.g. EURUSD_1m.csv, US30.csv).")

    # 3) train ONE bot on all of them (shared pot)
    print(f"\n[3/5] Teaching ONE bot to trade all {len(found)} together, from one account...")
    try:
        import stable_baselines3  # noqa: F401
    except Exception:
        sys.exit("\n  The training engine isn't installed yet. Run this ONE line, then run me again:\n"
                 "      pip install stable-baselines3 torch\n")
    from src.training.trainer import train_portfolio
    from src.data.cache_builder import load_cache
    from src.env.portfolio_env import align_symbol_data
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all

    def _reg():
        r = AlphaRegistry(); register_all(r); return r

    sd = align_symbol_data({s: load_cache(a.cache, s) for s in found})
    bars = len(next(iter(sd.values()))[1])
    # Where to SAVE the prepared features (so re-runs skip the slow build). Default: auto (Google Drive
    # on Colab, else local). 'off' disables. The cache is fingerprinted so it never loads stale features.
    from src.data.feature_cache import default_cache_dir
    feat_cache = None if (a.feature_cache or "").lower() == "off" else (a.feature_cache or default_cache_dir())
    print(f"      {len(found)} symbols, {bars:,} shared bars — training for {a.steps:,} steps "
          f"(grab a coffee; this is the slow part)...")
    if feat_cache:
        print(f"      prepared features will be saved/reused at: {feat_cache}")
    train_portfolio(sd, _reg, total_timesteps=a.steps, save_path=a.out, feature_cache_dir=feat_cache,
                    data_cache_dir=a.cache, symbols=found)   # data_cache_dir+symbols enable multi-core workers
    print(f"      trained + saved -> {a.out} (+ its _vecnorm.pkl)")

    # 4) the day-by-day results (the part you care about)
    print("\n[4/5] How it did, DAY BY DAY — did it make +2.5% of your balance and stay inside the 4% wall:\n")
    from src.training.daily_report import run_portfolio_report
    rows, summary = run_portfolio_report(a.cache, found, model_path=a.out)

    # 5) save it to the roster so JARVIS can organize it
    print("\n[5/5] Filing this bot in the roster (so you can ask JARVIS which to run)...")
    try:
        from src.jarvis import policy_registry as PR
        from src.training.env_fingerprint import env_fingerprint
        pr = summary["days_passed_target"] / max(1, summary["days"])
        PR.add_policy(id=os.path.basename(a.out), path=a.out, fingerprint=env_fingerprint(),
                      universe=",".join(found), walk_forward_pass_rate=round(pr, 3),
                      notes="trained by run_training.py")
        print("      filed. In the cockpit, ask JARVIS: \"which policy should I run?\"")
    except Exception as e:
        print(f"      (couldn't file it automatically: {e})")

    p, w, n = summary["days_passed_target"], summary["days_within_trailing"], summary["days"]
    print(f"\n{line}\n  ALL DONE.  {p}/{n} days hit +2.5%   |   {w}/{n} stayed inside the 4% wall   |   "
          f"final {summary['final_cum_pct']:+.2f}%\n  Challenge: "
          f"{'PASSED ✅' if summary['challenge_passed'] else 'not yet — keep training / tuning'}\n{line}")
    return summary


if __name__ == "__main__":   # pragma: no cover
    main()
