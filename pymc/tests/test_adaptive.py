from __future__ import with_statement

"""Test AdaptiveMetropolis."""
import pymc
import numpy as np
from numpy.testing import *
import nose
import warnings


def test_square():
    iw = pymc.Wishart("A", 2, np.eye(2))
    mnc = pymc.MvNormal(
        "v",
        np.zeros(2),
        iw,
        value=np.zeros(2),
        observed=True)

    M = pymc.MCMC([iw, mnc])
    M.sample(8, progress_bar=0)


if __name__ == '__main__':

    original_filters = warnings.filters[:]
    warnings.simplefilter("ignore")
    try:
        nose.runmodule()
    finally:
        warnings.filters = original_filters

    # TODO: Restore this implementation in 2.2
    # with warnings.catch_warnings():
    #         warnings.simplefilter('ignore',  FutureWarning)
    #         nose.runmodule()
