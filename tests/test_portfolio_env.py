# ONE bot, ONE pot, all symbols: the shared-pot PortfolioEnv. Obs stays 479; positions are
# simultaneous in one account; FTMO breach is on the pot; it scales to any universe size.
import numpy as np
import pandas as pd
from config import constants as C
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _data(symbols, n=3000, seed=0):
    time_ns = pd.date_range("2026-03-02 00:00", periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    rng = np.random.default_rng(seed)
    base = {"EURUSD": 1.10, "GBPUSD": 1.27, "XAUUSD": 2000.0, "US30": 38000.0}
    scl = {"EURUSD": 3e-4, "GBPUSD": 3e-4, "XAUUSD": 0.5, "US30": 2.0}
    sd = {}
    for s in symbols:
        close = (base.get(s, 100.0) + np.cumsum(rng.standard_normal(n) * scl.get(s, 0.05))).astype(np.float32)
        sd[s] = (np.zeros((n, C.N_INDICATORS_TOTAL), np.float32), close, time_ns)
    return sd


def test_portfolio_obs_is_still_479():
    env = PortfolioEnv(_data(["EURUSD", "US30"]), _reg)
    obs, _ = env.reset()
    assert obs.shape == (479,) and np.all(np.isfinite(obs))
    assert C.OBS_TOTAL_SIZE == 479                       # the locked obs is unchanged for the portfolio


def test_portfolio_scales_to_any_universe_without_changing_obs():
    for syms in (["EURUSD"], ["EURUSD", "US30"], ["EURUSD", "GBPUSD", "XAUUSD", "US30"]):
        env = PortfolioEnv(_data(syms), _reg)
        obs, _ = env.reset()
        assert obs.shape == (479,)                       # identical regardless of how many symbols
        assert env.symbols == syms


def test_portfolio_simultaneous_positions_in_one_pot():
    env = PortfolioEnv(_data(["EURUSD", "XAUUSD", "US30"]), _reg)
    env.reset()
    env.step(C.ACTION_BUY)                               # EURUSD long
    env.step(C.ACTION_SELL)                              # XAUUSD short
    env.step(C.ACTION_BUY)                               # US30 long  (bar advances)
    assert sum(1 for p in env.position.values() if p != 0) == 3      # three open AT ONCE
    assert env.acc.open_positions == 3
    assert env.position["XAUUSD"] == -1 and env.position["EURUSD"] == 1
    # one shared account: a single equity/balance pot for all of them
    assert hasattr(env, "acc") and env.acc.open_positions == 3


def test_portfolio_breach_is_on_the_shared_pot():
    env = PortfolioEnv(_data(["EURUSD", "XAUUSD", "US30"], n=4320, seed=5), _reg)
    env.reset()
    done, steps = False, 0
    while not done and steps < 20000:
        # dumb always-buy -> stress the pot to the breach OR the end of data. Breach/pass = terminated;
        # reaching the end of data (a time limit) = truncated. Either ends the episode.
        _, _, term, trunc, _ = env.step(C.ACTION_BUY)
        done = bool(term or trunc)
        steps += 1
    assert done                                          # episode ends (breach/pass = terminated, end = truncated)
    assert isinstance(env.acc.episode_breached, bool)


def test_portfolio_day_by_day_report_runs_on_the_pot():
    from src.training.daily_report import daily_report
    env = PortfolioEnv(_data(["EURUSD", "US30"], n=3000), _reg)
    rows, summary = daily_report(env, policy=None)       # HOLD on the whole pot
    assert rows and summary["daily_target_pct"] == 2.5 and summary["trailing_pct"] == 4.0
    assert summary["days_passed_target"] == 0 and summary["breaches"] == 0   # flat baseline


def test_portfolio_daily_report_covers_all_days_not_just_a_quarter():
    # Regression: the report guard once counted per-symbol SUB-steps as if they were bars, so it cut the
    # portfolio run off ~1/len(symbols) of the way in (3 symbols -> ~30% of the data, sometimes 0 rows).
    # It must now traverse the WHOLE range and report every day.
    from src.training.daily_report import daily_report
    env = PortfolioEnv(_data(["EURUSD", "XAUUSD", "US30"], n=4320), _reg)    # ~3 calendar days, 3 symbols
    rows, summary = daily_report(env, policy=None)
    assert summary["days"] >= 3                           # all ~3 days reported, not ~1
    assert env.ptr >= env.T - 2                           # traversed essentially all bars (not ~30%)


def test_portfolio_random_window_gives_diverse_starts():
    # With random_window each worker starts at a RANDOM bar; different seeds -> different start sequences,
    # so vectorised training explores DIFFERENT stretches instead of replaying one identical trajectory.
    sd = _data(["EURUSD", "US30"], n=6000)
    e1 = PortfolioEnv(sd, _reg, random_window=True, window=1000, seed=1)
    e2 = PortfolioEnv(sd, _reg, random_window=True, window=1000, seed=2)
    starts1, starts2 = [], []
    for _ in range(6):
        e1.reset(); starts1.append(e1.t)
        e2.reset(); starts2.append(e2.t)
    assert len(set(starts1)) > 1                          # starts actually vary (not pinned to warmup)
    assert starts1 != starts2                             # the two seeds diverge
    e3 = PortfolioEnv(sd, _reg)                           # default: random_window off (eval / report path)
    e3.reset()
    assert e3.t == e3.warmup                              # deterministic full walk-forward


def test_portfolio_random_window_diversifies_even_on_short_history():
    # Regression for the silent collapse: with window >= the data, every worker used to pin to `warmup`
    # (identical copies again). The window is now clamped to the usable span, so starts still vary on a
    # SHORT (trimmed --from/--to) history -- the exact fast-first-run case an operator hits first.
    sd = _data(["EURUSD", "US30"], n=3000)
    starts = {PortfolioEnv(sd, _reg, random_window=True, window=5000, seed=s).t for s in range(8)}
    assert len(starts) > 1                                # workers diversify even though window (5000) > T (3000)
