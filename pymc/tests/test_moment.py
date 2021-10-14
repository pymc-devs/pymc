import aesara
import numpy as np
import pytest

from aesara import tensor as at

import pymc as pm

from pymc.distributions.distribution import get_moment
from pymc.distributions.shape_utils import to_tuple


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_moment_scalar(size):
    def moment(rv, size, mu):
        return (at.ones(size) * mu).astype(rv.dtype)

    mu_val = np.array(np.random.normal(loc=2, scale=1)).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu")
        a = pm.DensityDist("a", mu, get_moment=moment, size=size)
    evaled_moment = get_moment(a).eval({mu: mu_val})
    assert evaled_moment.shape == to_tuple(size)
    assert np.all(evaled_moment == mu_val)


@pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
def test_density_dist_moment_multivariate(size):
    def moment(rv, size, mu):
        return (at.ones(size)[..., None] * mu).astype(rv.dtype)

    mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
    with pm.Model():
        mu = pm.Normal("mu", size=5)
        a = pm.DensityDist("a", mu, get_moment=moment, ndims_params=[1], ndim_supp=1, size=size)
    evaled_moment = get_moment(a).eval({mu: mu_val})
    assert evaled_moment.shape == to_tuple(size) + (5,)
    assert np.all(evaled_moment == mu_val)
