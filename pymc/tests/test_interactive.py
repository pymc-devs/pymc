"""
Test interactive sampler
"""

# TODO: Make real test case.

from pymc import MCMC
from pymc.examples import disaster_model
import os
import nose


def test_interactive():
    S = MCMC(disaster_model)
    S.isample(
        200,
        100,
        2,
        out=open(
            'testresults/interactive.log',
            'w'),
        progress_bar=0)


if __name__ == '__main__':
    nose.runmodule()
