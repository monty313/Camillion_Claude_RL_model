# Test 8: the observation shape contract holds (513 float32 finite) and stays
# constant as strategies are added.
import numpy as np
from config import constants as C
from src.observation import observation_contract as OC
from src.observation import builder as B
from src.strategies.base import BaseStrategy
from src.strategies.registry import AlphaRegistry
from src.account.account_state import AccountState


class _Buy(BaseStrategy):
    def compute_signal(self, ctx):
        return 1


def test_total_size_479():
    assert C.OBS_TOTAL_SIZE == 513 and C.OBS_SHAPE == (513,)
    assert len(OC.FEATURE_NAMES) == 513


def test_block_sizes_sum_to_total():
    assert sum(s.stop - s.start for s in OC.BLOCK_SLICES.values()) == 513


def test_zeros_observation_valid():
    z = B.zeros()
    assert z.shape == (513,) and z.dtype == np.float32 and np.all(np.isfinite(z))


def test_full_build_finite_and_shaped():
    reg = AlphaRegistry()
    reg.register(_Buy("a"))
    obs = B.build(indicators=np.full(220, np.nan, np.float32),   # stub NaN -> sanitised
                  alpha_values=reg.collect_alphas(None),
                  occupancy_mask=reg.occupancy_mask(),
                  account=AccountState(100000.0))
    assert obs.shape == (513,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))


def test_validate_rejects_wrong_shape():
    raised = False
    try:
        OC.validate(np.zeros(10, np.float32))
    except ValueError:
        raised = True
    assert raised


def test_shape_constant_as_strategies_added():
    reg = AlphaRegistry()
    for i in range(25):
        reg.register(_Buy(f"s{i}"))
        obs = B.build(alpha_values=reg.collect_alphas(None),
                      occupancy_mask=reg.occupancy_mask())
        assert obs.shape == (513,)
