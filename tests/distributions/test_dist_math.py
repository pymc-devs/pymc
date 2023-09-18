#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.special

from pytensor import config, function
from pytensor.tensor.random.basic import multinomial
from scipy import interpolate, stats

import pymc as pm

from pymc.distributions import Discrete
from pymc.distributions.dist_math import (
    SplineWrapper,
    check_parameters,
    clipped_beta_rvs,
    factln,
    i0e,
    incomplete_beta,
    multigammaln,
)
from pymc.logprob.utils import ParameterValueError
from pymc.pytensorf import floatX
from tests.checks import close_to
from tests.helpers import verify_grad


@pytest.mark.parametrize(
    "conditions, succeeds",
    [
        ([], True),
        ([True], True),
        ([pt.ones(10)], True),
        ([pt.ones(10), 5 * pt.ones(101)], True),
        ([np.ones(10), 5 * pt.ones(101)], True),
        ([np.ones(10), True, 5 * pt.ones(101)], True),
        ([np.array([1, 2, 3]), True, 5 * pt.ones(101)], True),
        ([False], False),
        ([pt.zeros(10)], False),
        ([True, False], False),
        ([np.array([0, -1]), pt.ones(60)], False),
        ([np.ones(10), False, 5 * pt.ones(101)], False),
    ],
)
def test_check_parameters(conditions, succeeds):
    ret = check_parameters(1, *conditions, msg="parameter check msg")
    if succeeds:
        assert ret.eval()
    else:
        with pytest.raises(ParameterValueError, match="^parameter check msg$"):
            ret.eval()


def test_check_parameters_shape():
    conditions = [True, pt.ones(10), pt.ones(5)]
    assert check_parameters(1, *conditions).eval().shape == ()


class MultinomialA(Discrete):
    rv_op = multinomial

    @classmethod
    def dist(cls, n, p, *args, **kwargs):
        return super().dist([n, p], **kwargs)

    def logp(value, n, p):
        return check_parameters(
            factln(n) - factln(value).sum() + (value * pt.log(p)).sum(),
            value >= 0,
            0 <= p,
            p <= 1,
            pt.isclose(p.sum(), 1),
        )


class MultinomialB(Discrete):
    rv_op = multinomial

    @classmethod
    def dist(cls, n, p, *args, **kwargs):
        return super().dist([n, p], **kwargs)

    def logp(value, n, p):
        return check_parameters(
            factln(n) - factln(value).sum() + (value * pt.log(p)).sum(),
            pt.all(value >= 0),
            pt.all(0 <= p),
            pt.all(p <= 1),
            pt.isclose(p.sum(), 1),
        )


def test_multinomial_check_parameters():
    x = np.array([1, 5])
    n = x.sum()

    with pm.Model() as modelA:
        p_a = pm.Dirichlet("p", floatX(np.ones(2)))
        MultinomialA("x", n, p_a, observed=x)

    with pm.Model() as modelB:
        p_b = pm.Dirichlet("p", floatX(np.ones(2)))
        MultinomialB("x", n, p_b, observed=x)

    assert np.isclose(
        modelA.compile_logp()({"p_simplex__": [0]}), modelB.compile_logp()({"p_simplex__": [0]})
    )


class TestSplineWrapper:
    @pytensor.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        verify_grad(spline, [0.5])

    @pytensor.config.change_flags(compute_test_value="ignore")
    def test_hessian(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        x_var = pt.dscalar("x")
        (g_x,) = pt.grad(spline(x_var), [x_var])
        with pytest.raises(NotImplementedError):
            pt.grad(g_x, [x_var])


class TestI0e:
    @pytensor.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        verify_grad(i0e, [0.5])
        verify_grad(i0e, [-2.0])
        verify_grad(i0e, [[0.5, -2.0]])
        verify_grad(i0e, [[[0.5, -2.0]]])


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_clipped_beta_rvs(dtype):
    # Verify that the samples drawn from the beta distribution are never
    # equal to zero or one (issue #3898)
    values = clipped_beta_rvs(0.01, 0.01, size=1000000, dtype=dtype)
    assert not (np.any(values == 0) or np.any(values == 1))


def check_vals(fn1, fn2, *args):
    v = fn1(*args)
    close_to(v, fn2(*args), 1e-6 if v.dtype == np.float64 else 1e-4)


def test_multigamma():
    x = pt.vector("x", shape=(1,))
    p = pt.scalar("p")

    xvals = [np.array([v], dtype=config.floatX) for v in [0.1, 2, 5, 10, 50, 100]]

    multigammaln_ = function([x, p], multigammaln(x, p), mode="FAST_COMPILE")

    def ref_multigammaln(a, b):
        return np.array(scipy.special.multigammaln(a[0], b), config.floatX)

    for p in [0, 1, 2, 3, 4, 100]:
        for x in xvals:
            if np.all(x > 0.5 * (p - 1)):
                check_vals(multigammaln_, ref_multigammaln, x, p)


def test_incomplete_beta_deprecation():
    with pytest.warns(FutureWarning, match="incomplete_beta has been deprecated"):
        res = incomplete_beta(3, 5, 0.5).eval()
    assert np.isclose(res, pt.betainc(3, 5, 0.5).eval())
