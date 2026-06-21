# Tests 1 & 2: registry shape constant as strategies grow; unused slots == 0;
# empty slot is distinguishable from assigned-but-inactive.
import numpy as np
from config import constants as C
from src.strategies.base import BaseStrategy
from src.strategies.registry import StrategyRegistry


class _S(BaseStrategy):
    def __init__(self, v, name):
        super().__init__(name)
        self.v = v
    def compute_signal(self, ctx):
        return self.v


def test_shape_constant_as_strategies_added():
    reg = StrategyRegistry()
    for i in range(20):
        reg.register(_S(1 if i % 2 else -1, f"s{i}"))
        assert reg.collect_signals(None).shape == (C.MAX_STRATEGIES,)
        assert reg.occupancy_mask().shape == (C.MAX_STRATEGIES,)


def test_unused_slots_zero():
    reg = StrategyRegistry()
    reg.register(_S(1, "a"))
    sig = reg.collect_signals(None)
    assert sig[0] == 1.0 and np.all(sig[1:] == 0.0)


def test_empty_vs_inactive_distinguished():
    reg = StrategyRegistry()
    reg.register(_S(0, "inactive"))           # assigned but no setup
    sig, mask = reg.collect_signals(None), reg.occupancy_mask()
    assert sig[0] == 0.0 and mask[0] == 1.0   # inactive: value 0, occupied
    assert sig[1] == 0.0 and mask[1] == 0.0   # empty: value 0, not occupied


def test_invalid_signal_rejected():
    class Bad(BaseStrategy):
        def compute_signal(self, ctx):
            return 2
    reg = StrategyRegistry()
    reg.register(Bad("bad"))
    raised = False
    try:
        reg.collect_signals(None)
    except ValueError:
        raised = True
    assert raised
