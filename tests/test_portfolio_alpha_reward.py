# Alpha-shaping (2026-06-27, operator decision, ON by default): the PortfolioEnv reward now adds small
# alpha-conditioned terms -- USE the alphas (bonus for a profitable close that agreed with >=50% of firing
# alphas), BEAT the alphas (bonus for out-earning a follow-the-consensus position), and a penalty for OPENING
# against >=50%. Every bonus is CAPPED at the trade's own PnL and only pays when the day is net up. These
# tests force a known consensus and check each term, the cap, and that the OFF path stays alpha-INDEPENDENT.
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import FTMOConfig
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.portfolio_env import PortfolioEnv


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _env(prices, *, alpha_on, consensus_dir, n_slots=10, **cfgkw):
    """1-symbol pot with a FORCED alpha consensus (all n_slots fire `consensus_dir` every bar)."""
    n = len(prices)
    idx = pd.date_range("2026-03-02 09:00", periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    sd = {"TESTPAIR": (ind, np.asarray(prices, dtype=np.float64), idx)}
    cfg = FTMOConfig(alpha_reward_enabled=alpha_on, phase2_continue=False, **cfgkw)
    env = PortfolioEnv(sd, _reg, warmup=0, cfg=cfg)
    sub = env.subs["TESTPAIR"]
    occ = np.zeros(C.MAX_STRATEGIES, dtype=bool); occ[:n_slots] = True
    sub.occupancy = occ
    am = np.zeros((sub.T, C.MAX_STRATEGIES), dtype=np.float32)
    am[:, :n_slots] = float(consensus_dir)                 # every firing slot points `consensus_dir`
    sub.alpha_matrix = am
    env.reset()
    return env


def _run(env, actions):
    out = []
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        out.append(r)
        if term or trunc:
            break
    return out


# prices: +1.5% then flat -> profitable + day-up, but BELOW the +2.5% auto-bank (so OUR close fires, not the engine's)
PR = [100.0, 100.015, 100.015]
SEQ = [C.ACTION_BUY, C.ACTION_CLOSE, C.ACTION_HOLD]


def test_off_path_is_alpha_independent():
    # With the toggle OFF, different alpha consensus must NOT change the reward (invariant preserved).
    off_buy = _run(_env(PR, alpha_on=False, consensus_dir=1), SEQ)
    off_sell = _run(_env(PR, alpha_on=False, consensus_dir=-1), SEQ)
    assert np.allclose(off_buy, off_sell), "reward depends on alphas while OFF -> invariant broken"


def test_agree_bonus_fires_on_profitable_consensus_close():
    off = _run(_env(PR, alpha_on=False, consensus_dir=1), SEQ)
    on = _run(_env(PR, alpha_on=True, consensus_dir=1), SEQ)        # BUY agrees with the +1 consensus
    delta_close = on[1] - off[1]
    assert abs(delta_close - 0.001) < 3e-4, f"agree bonus ~0.001 expected, got {delta_close}"


def test_penalty_for_opening_against_the_consensus():
    off = _run(_env(PR, alpha_on=False, consensus_dir=-1), SEQ)
    on = _run(_env(PR, alpha_on=True, consensus_dir=-1), SEQ)       # BUY against the -1 (SELL) consensus
    delta_open = on[0] - off[0]
    assert delta_open < -0.0005, f"counter-consensus open should be penalised, got {delta_open}"


def test_beat_bonus_when_winning_against_the_consensus():
    off = _run(_env(PR, alpha_on=False, consensus_dir=-1), SEQ)
    on = _run(_env(PR, alpha_on=True, consensus_dir=-1), SEQ)       # alphas said SELL; BUY won -> beat them
    assert (on[0] - off[0]) < -0.0005          # entry: penalised for fighting the consensus
    assert (on[1] - off[1]) > 0.0015           # close: BEAT bonus is 2x (~0.002) so it isn't cancelled by the penalty


def test_bonus_is_capped_at_the_trade_pnl():
    # Huge coef but a modest ~+1.5% trade -> the bonus is CAPPED at the trade's PnL, NOT the 0.5 coef.
    off = _run(_env(PR, alpha_on=False, consensus_dir=1, alpha_agree_bonus=0.5, alpha_beat_bonus=0.0), SEQ)
    on = _run(_env(PR, alpha_on=True, consensus_dir=1, alpha_agree_bonus=0.5, alpha_beat_bonus=0.0), SEQ)
    delta_close = on[1] - off[1]
    assert delta_close < 0.05, f"bonus not capped at PnL (got {delta_close}, coef was 0.5)"
    assert 0.008 < delta_close < 0.016, f"capped bonus should ~= the trade PnL (~0.0115), got {delta_close}"
