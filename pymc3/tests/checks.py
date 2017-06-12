import numpy as np


def close_to(x, v, bound, name="value"):
    assert np.all(np.logical_or(
            np.abs(x - v) < bound,
            x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)


def close_to_logical(x, v, bound, name="value"):    
    assert np.all(np.logical_or(
            np.abs(np.bitwise_xor(x, v)) < bound,
            x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)
