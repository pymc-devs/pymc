import pymc
from pymc.examples import gelman_bioassay
import numpy as np
from numpy.testing import *

# Easy dice example (analytically solvable)


def dice(data=None):

    if data is None:
        x = [pymc.rbernoulli(1.0 / 6.0) for i in range(0, 100)]
    else:
        x = data

    prob = pymc.Uniform('prob', lower=0, upper=1)

    d = pymc.Bernoulli('d', p=prob, value=x, observed=True)

    return locals()


class TestSlice(TestCase):

    def test_dice_sample(self):
        M = pymc.MCMC(dice())
        M.use_step_method(pymc.Slicer, M.prob, w=.1)
        M.sample(iter=1000, tune_interval=1)

    def test_bioassay_sample(self):
        M = pymc.MCMC(gelman_bioassay)
        for stoch in M.stochastics:
            M.use_step_method(pymc.Slicer, stoch, w=.1)
        M.sample(iter=1000, tune_interval=1)
