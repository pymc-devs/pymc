#   Copyright 2020 The PyMC Developers
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
import sys

import numpy as np
import numpy.testing as npt
import pytest
import theano
import theano.tensor as tt

from scipy import interpolate, stats

import pymc3 as pm

from pymc3.distributions import Discrete
from pymc3.distributions.dist_math import (
    MvNormalLogp,
    SplineWrapper,
    alltrue_scalar,
    bound,
    clipped_beta_rvs,
    factln,
    i0e,
)
from pymc3.tests.helpers import verify_grad
from pymc3.theanof import floatX


def test_bound():
    logp = tt.ones((10, 10))
    cond = tt.ones((10, 10))
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = tt.ones((10, 10))
    cond = tt.zeros((10, 10))
    assert np.all(bound(logp, cond).eval() == (-np.inf * logp).eval())

    logp = tt.ones((10, 10))
    cond = True
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = tt.ones(3)
    cond = np.array([1, 0, 1])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf

    logp = tt.ones((2, 3))
    cond = np.array([[1, 1, 1], [1, 0, 1]])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf


def test_check_bounds_false():
    with pm.Model(check_bounds=False):
        logp = tt.ones(3)
        cond = np.array([1, 0, 1])
        assert np.all(bound(logp, cond).eval() == logp.eval())


def test_alltrue_scalar():
    assert alltrue_scalar([]).eval()
    assert alltrue_scalar([True]).eval()
    assert alltrue_scalar([tt.ones(10)]).eval()
    assert alltrue_scalar([tt.ones(10), 5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10), 5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10), True, 5 * tt.ones(101)]).eval()
    assert alltrue_scalar([np.array([1, 2, 3]), True, 5 * tt.ones(101)]).eval()

    assert not alltrue_scalar([False]).eval()
    assert not alltrue_scalar([tt.zeros(10)]).eval()
    assert not alltrue_scalar([True, False]).eval()
    assert not alltrue_scalar([np.array([0, -1]), tt.ones(60)]).eval()
    assert not alltrue_scalar([np.ones(10), False, 5 * tt.ones(101)]).eval()


def test_alltrue_shape():
    vals = [True, tt.ones(10), tt.zeros(5)]

    assert alltrue_scalar(vals).eval().shape == ()


class MultinomialA(Discrete):
    def __init__(self, n, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = n
        self.p = p

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(
            factln(n) - factln(value).sum() + (value * tt.log(p)).sum(),
            value >= 0,
            0 <= p,
            p <= 1,
            tt.isclose(p.sum(), 1),
            broadcast_conditions=False,
        )


class MultinomialB(Discrete):
    def __init__(self, n, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = n
        self.p = p

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(
            factln(n) - factln(value).sum() + (value * tt.log(p)).sum(),
            tt.all(value >= 0),
            tt.all(0 <= p),
            tt.all(p <= 1),
            tt.isclose(p.sum(), 1),
            broadcast_conditions=False,
        )


def test_multinomial_bound():

    x = np.array([1, 5])
    n = x.sum()

    with pm.Model() as modelA:
        p_a = pm.Dirichlet("p", floatX(np.ones(2)), shape=(2,))
        MultinomialA("x", n, p_a, observed=x)

    with pm.Model() as modelB:
        p_b = pm.Dirichlet("p", floatX(np.ones(2)), shape=(2,))
        MultinomialB("x", n, p_b, observed=x)

    assert np.isclose(
        modelA.logp({"p_stickbreaking__": [0]}), modelB.logp({"p_stickbreaking__": [0]})
    )


class TestMvNormalLogp:
    def test_logp(self):
        np.random.seed(42)

        chol_val = floatX(np.array([[1, 0.9], [0, 2]]))
        cov_val = floatX(np.dot(chol_val, chol_val.T))
        cov = tt.matrix("cov")
        cov.tag.test_value = cov_val
        delta_val = floatX(np.random.randn(5, 2))
        delta = tt.matrix("delta")
        delta.tag.test_value = delta_val
        expect = stats.multivariate_normal(mean=np.zeros(2), cov=cov_val)
        expect = expect.logpdf(delta_val).sum()
        logp = MvNormalLogp()(cov, delta)
        logp_f = theano.function([cov, delta], logp)
        logp = logp_f(cov_val, delta_val)
        npt.assert_allclose(logp, expect)

    @theano.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        np.random.seed(42)

        def func(chol_vec, delta):
            chol = tt.stack(
                [
                    tt.stack([tt.exp(0.1 * chol_vec[0]), 0]),
                    tt.stack([chol_vec[1], 2 * tt.exp(chol_vec[2])]),
                ]
            )
            cov = tt.dot(chol, chol.T)
            return MvNormalLogp()(cov, delta)

        chol_vec_val = floatX(np.array([0.5, 1.0, -0.1]))

        delta_val = floatX(np.random.randn(1, 2))
        verify_grad(func, [chol_vec_val, delta_val])

        delta_val = floatX(np.random.randn(5, 2))
        verify_grad(func, [chol_vec_val, delta_val])

    @pytest.mark.skip(reason="Fix in theano not released yet: Theano#5908")
    @theano.config.change_flags(compute_test_value="ignore")
    def test_hessian(self):
        chol_vec = tt.vector("chol_vec")
        chol_vec.tag.test_value = np.array([0.1, 2, 3])
        chol = tt.stack(
            [
                tt.stack([tt.exp(0.1 * chol_vec[0]), 0]),
                tt.stack([chol_vec[1], 2 * tt.exp(chol_vec[2])]),
            ]
        )
        cov = tt.dot(chol, chol.T)
        delta = tt.matrix("delta")
        delta.tag.test_value = np.ones((5, 2))
        logp = MvNormalLogp()(cov, delta)
        g_cov, g_delta = tt.grad(logp, [cov, delta])
        tt.grad(g_delta.sum() + g_cov.sum(), [delta, cov])


class TestSplineWrapper:
    @theano.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        verify_grad(spline, [0.5])

    @theano.config.change_flags(compute_test_value="ignore")
    def test_hessian(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        x_var = tt.dscalar("x")
        (g_x,) = tt.grad(spline(x_var), [x_var])
        with pytest.raises(NotImplementedError):
            tt.grad(g_x, [x_var])


class TestI0e:
    @theano.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        verify_grad(i0e, [0.5])
        verify_grad(i0e, [-2.0])
        verify_grad(i0e, [[0.5, -2.0]])
        verify_grad(i0e, [[[0.5, -2.0]]])


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "float128"]
    if sys.platform != "win32"
    else ["float16", "float32", "float64"],
)
def test_clipped_beta_rvs(dtype):
    # Verify that the samples drawn from the beta distribution are never
    # equal to zero or one (issue #3898)
    values = clipped_beta_rvs(0.01, 0.01, size=1000000, dtype=dtype)
    assert not (np.any(values == 0) or np.any(values == 1))
