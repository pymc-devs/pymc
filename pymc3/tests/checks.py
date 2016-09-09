import numpy as np


def close_to(x, v, bound, name="value"):
    assert np.all(np.logical_or(
        np.abs(x - v) < bound,
        x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)
