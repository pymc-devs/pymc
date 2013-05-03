import pymc as pm
import numpy as np

from pymc import sample, psample 

from numpy.testing import assert_almost_equal
    
def close_to(x, v, bound, name = "value"):
    assert np.all(np.logical_or(
        np.abs(x - v) < bound,
        x == v)), name + " out of bounds : " + repr(x) + ", " + repr(v) + ", " + repr(bound)

