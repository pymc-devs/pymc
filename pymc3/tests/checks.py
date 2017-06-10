import numpy as np


def close_to(x, v, bound, name="value"):
    if v is True:
        assert np.all(np.logical_or(
                np.abs(np.bitwise_xor(x, v)) < bound,
                x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)
    else:
        assert np.all(np.logical_or(
                np.abs(x - v) < bound,
                x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)
