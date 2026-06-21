# Test 8: the observation shape contract holds (357 float32 finite) and stays
# constant as strategies are added.
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC
from src.observation import builder as B
from src.strategies.base import BaseStrategy
from src.strategies.registry import StrategyRegistry
from src.account.account_state import AccountState


class _Buy(BaseStrategy):
    def compute_signal(self, ctx):
        return 1


def test_total_size_357():
    assert C.OBS_TOTAL_SIZE == 357 and C.OBS_SHAPE == (357,)
    assert len(OC.FEATURE_NAMES) == 357


def test_block_sizes_sum_to_total():
    assert sum(s.stop - s.start for s in OC.BLOCK_SLICES.values()) == 357


def test_zeros_observation_valid():
    z = B.zeros()
    assert z.shape == (357,) and z.dtype == np.float32 and np.all(np.isfinite(z))


def test_full_build_finite_and_shaped():
    reg = StrategyRegistry()
    reg.register(_Buy("a"))
    obs = B.build(indicators=np.full(190, np.nan, np.float32),   # stub NaN -> sanitised
                  alpha_values=reg.collect_signals(None),
                  occupancy_mask=reg.occupancy_mask(),
                  account=AccountState(100000.0))
    assert obs.shape == (357,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))


def test_validate_rejects_wrong_shape():
    raised = False
    try:
        OC.validate(np.zeros(10, np.float32))
    except ValueError:
        raised = True
    assert raised


def test_shape_constant_as_strategies_added():
    reg = StrategyRegistry()
    for i in range(25):
        reg.register(_Buy(f"s{i}"))
        obs = B.build(alpha_values=reg.collect_signals(None),
                      occupancy_mask=reg.occupancy_mask())
        assert obs.shape == (357,)
