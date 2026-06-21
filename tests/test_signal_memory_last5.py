# Test 4: last-5-bar signal memory always has exactly 5 values.
import numpy as np
from src.signals.signal_memory import SignalMemory, last5_from_series


def test_memory_exactly_five():
    m = SignalMemory()
    assert len(m.as_vector()) == 5
    for x in [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]:
        m.push(x)
    v = m.as_vector()
    assert len(v) == 5 and v[0] == np.float32(0.6)   # newest at lag0


def test_series_slice_length_five():
    net = np.array([0.1, 0.2, 0.3])
    v = last5_from_series(net, t=2)
    assert len(v) == 5
    assert v[0] == np.float32(0.3) and v[3] == 0.0 and v[4] == 0.0
