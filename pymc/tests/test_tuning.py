import numpy as np
from numpy import inf
from pymc.tuning import scaling
import models


a = np.array([-10, -.01, 0, 10, 1e300, -inf, inf])
def test_adjust_precision():
    a1 = scaling.adjust_precision(a)

    assert all((a1 > 0) & (a1 < 1e200))

def test_guess_scaling():

    start, model,_ = models.non_normal(n = 5)
    with model:
        a1 = scaling.guess_scaling(start)

    assert all((a1 > 0) & (a1 < 1e200))
