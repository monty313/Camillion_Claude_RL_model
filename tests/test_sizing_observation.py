# v1.7.0 SIZING observation block (10 floats, ALL fractions of INITIAL balance): a 0.01..4-lot
# what-if ladder (size -> account-% of a typical move) + how much is still needed today +
# drawdown room + the active size. OBSERVATION ONLY (sizing is not an action yet).
import numpy as np
import pandas as pd
from config import constants as C
from config.ftmo_config import load_ftmo_config
from config import asset_specs as A
from src.account.account_state import AccountState
from src.account import win_loss_features as WL
from src.observation import observation_contract as OC
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry


def test_sizing_block_is_in_contract():
    assert C.OBS_BLOCK_SIZING == 10 and C.OBSERVATION_CONTRACT_VERSION == "v1.12.0"
    sl = OC.BLOCK_SLICES["sizing"]
    assert sl.stop - sl.start == 10
    assert OC.FEATURE_NAMES[sl.start] == "size_move_pct_lot0_01"
    assert OC.FEATURE_NAMES[sl.stop - 1] == "active_move_value_pct"


def test_ladder_is_monotone_and_bounded():
    f = WL.sizing_features(AccountState(starting_balance=100_000.0), load_ftmo_config(),
                           value_per_point=100_000.0, ref_move=0.0025, position_size=312_500.0)
    ladder = f[:6]                                  # lots 0.01,0.1,0.5,1,2,4
    assert np.all(np.diff(ladder) >= 0)             # bigger size -> bigger % of account
    assert np.all((ladder >= 0) & (ladder <= 1))


def test_target_remaining_shrinks_as_you_make_money():
    cfg = load_ftmo_config()
    acc = AccountState(starting_balance=100_000.0)
    f0 = WL.sizing_features(acc, cfg, value_per_point=100_000.0, ref_move=0.002, position_size=100_000.0)
    assert abs(f0[6] - cfg.daily_target_pct / 100.0) < 1e-6   # fresh day -> full 2.5% remaining
    acc.mark_equity(101_250.0)                                # +1.25% on the day
    f1 = WL.sizing_features(acc, cfg, value_per_point=100_000.0, ref_move=0.002, position_size=100_000.0)
    assert f1[6] < f0[6]


def test_dd_room_shrinks_after_drawdown():
    cfg = load_ftmo_config()
    acc = AccountState(starting_balance=100_000.0); acc.mark_equity(100_000.0); acc.mark_equity(98_000.0)
    f = WL.sizing_features(acc, cfg, value_per_point=100_000.0, ref_move=0.002, position_size=100_000.0)
    assert 0.0 <= f[7] < 1.0


def test_active_lots_norm_reflects_size():
    # position_size = 2 lots * 100,000 contract -> active_lots = 2 -> /4 (ladder max) = 0.5
    f = WL.sizing_features(AccountState(starting_balance=100_000.0), load_ftmo_config(),
                           value_per_point=100_000.0, ref_move=0.002, position_size=200_000.0)
    assert abs(f[8] - 0.5) < 1e-6


def test_env_resolves_per_asset_value_per_point_and_shape():
    n = 400
    idx = pd.date_range("2026-03-02 09:00", periods=n, freq="1min")
    close = (1.10 + np.cumsum(np.random.default_rng(0).standard_normal(n)) * 0.0002).astype(np.float32)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    env = TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64), AlphaRegistry(),
                     warmup=250, symbol="EURUSD", position_size=A.calibrated_position_size("EURUSD"))
    assert env.value_per_point == 100_000.0          # from the asset spec, not the position_size
    obs, _ = env.reset()
    assert obs.shape == (557,) and np.all(np.isfinite(obs))
    block = obs[OC.BLOCK_SLICES["sizing"]]
    assert block.shape == (10,)
    assert 0.6 < block[8] < 0.95                      # active_lots ~3.12 -> /4 ~0.78
