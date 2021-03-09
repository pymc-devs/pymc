#   Copyright 2021 The PyMC Developers
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
import aesara
import aesara.tensor as at
import numpy as np
import numpy.testing as npt
import pytest

from aesara.tensor.random.basic import multinomial
from scipy import interpolate, stats

import pymc3 as pm

from pymc3.aesaraf import floatX
from pymc3.distributions import Discrete
from pymc3.distributions.dist_math import (
    MvNormalLogp,
    SplineWrapper,
    alltrue_scalar,
    betainc,
    bound,
    clipped_beta_rvs,
    factln,
    i0e,
)
from pymc3.tests.helpers import select_by_precision, verify_grad


def test_bound():
    logp = at.ones((10, 10))
    cond = at.ones((10, 10))
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = at.ones((10, 10))
    cond = at.zeros((10, 10))
    assert np.all(bound(logp, cond).eval() == (-np.inf * logp).eval())

    logp = at.ones((10, 10))
    cond = True
    assert np.all(bound(logp, cond).eval() == logp.eval())

    logp = at.ones(3)
    cond = np.array([1, 0, 1])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf

    logp = at.ones((2, 3))
    cond = np.array([[1, 1, 1], [1, 0, 1]])
    assert not np.all(bound(logp, cond).eval() == 1)
    assert np.prod(bound(logp, cond).eval()) == -np.inf


def test_check_bounds_false():
    with pm.Model(check_bounds=False):
        logp = at.ones(3)
        cond = np.array([1, 0, 1])
        assert np.all(bound(logp, cond).eval() == logp.eval())


def test_alltrue_scalar():
    assert alltrue_scalar([]).eval()
    assert alltrue_scalar([True]).eval()
    assert alltrue_scalar([at.ones(10)]).eval()
    assert alltrue_scalar([at.ones(10), 5 * at.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10), 5 * at.ones(101)]).eval()
    assert alltrue_scalar([np.ones(10), True, 5 * at.ones(101)]).eval()
    assert alltrue_scalar([np.array([1, 2, 3]), True, 5 * at.ones(101)]).eval()

    assert not alltrue_scalar([False]).eval()
    assert not alltrue_scalar([at.zeros(10)]).eval()
    assert not alltrue_scalar([True, False]).eval()
    assert not alltrue_scalar([np.array([0, -1]), at.ones(60)]).eval()
    assert not alltrue_scalar([np.ones(10), False, 5 * at.ones(101)]).eval()


def test_alltrue_shape():
    vals = [True, at.ones(10), at.zeros(5)]

    assert alltrue_scalar(vals).eval().shape == ()


class MultinomialA(Discrete):
    rv_op = multinomial

    @classmethod
    def dist(cls, n, p, *args, **kwargs):
        return super().dist([n, p], **kwargs)

    def logp(value, n, p):
        return bound(
            factln(n) - factln(value).sum() + (value * at.log(p)).sum(),
            value >= 0,
            0 <= p,
            p <= 1,
            at.isclose(p.sum(), 1),
            broadcast_conditions=False,
        )


class MultinomialB(Discrete):
    rv_op = multinomial

    @classmethod
    def dist(cls, n, p, *args, **kwargs):
        return super().dist([n, p], **kwargs)

    def logp(value, n, p):
        return bound(
            factln(n) - factln(value).sum() + (value * at.log(p)).sum(),
            at.all(value >= 0),
            at.all(0 <= p),
            at.all(p <= 1),
            at.isclose(p.sum(), 1),
            broadcast_conditions=False,
        )


def test_multinomial_bound():

    x = np.array([1, 5])
    n = x.sum()

    with pm.Model() as modelA:
        p_a = pm.Dirichlet("p", floatX(np.ones(2)))
        MultinomialA("x", n, p_a, observed=x)

    with pm.Model() as modelB:
        p_b = pm.Dirichlet("p", floatX(np.ones(2)))
        MultinomialB("x", n, p_b, observed=x)

    assert np.isclose(
        modelA.logp({"p_stickbreaking__": [0]}), modelB.logp({"p_stickbreaking__": [0]})
    )


class TestMvNormalLogp:
    def test_logp(self):
        np.random.seed(42)

        chol_val = floatX(np.array([[1, 0.9], [0, 2]]))
        cov_val = floatX(np.dot(chol_val, chol_val.T))
        cov = at.matrix("cov")
        cov.tag.test_value = cov_val
        delta_val = floatX(np.random.randn(5, 2))
        delta = at.matrix("delta")
        delta.tag.test_value = delta_val
        expect = stats.multivariate_normal(mean=np.zeros(2), cov=cov_val)
        expect = expect.logpdf(delta_val).sum()
        logp = MvNormalLogp()(cov, delta)
        logp_f = aesara.function([cov, delta], logp)
        logp = logp_f(cov_val, delta_val)
        npt.assert_allclose(logp, expect)

    @aesara.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        np.random.seed(42)

        def func(chol_vec, delta):
            chol = at.stack(
                [
                    at.stack([at.exp(0.1 * chol_vec[0]), 0]),
                    at.stack([chol_vec[1], 2 * at.exp(chol_vec[2])]),
                ]
            )
            cov = at.dot(chol, chol.T)
            return MvNormalLogp()(cov, delta)

        chol_vec_val = floatX(np.array([0.5, 1.0, -0.1]))

        delta_val = floatX(np.random.randn(1, 2))
        verify_grad(func, [chol_vec_val, delta_val])

        delta_val = floatX(np.random.randn(5, 2))
        verify_grad(func, [chol_vec_val, delta_val])

    @aesara.config.change_flags(compute_test_value="ignore")
    def test_hessian(self):
        chol_vec = at.vector("chol_vec")
        chol_vec.tag.test_value = floatX(np.array([0.1, 2, 3]))
        chol = at.stack(
            [
                at.stack([at.exp(0.1 * chol_vec[0]), 0]),
                at.stack([chol_vec[1], 2 * at.exp(chol_vec[2])]),
            ]
        )
        cov = at.dot(chol, chol.T)
        delta = at.matrix("delta")
        delta.tag.test_value = floatX(np.ones((5, 2)))
        logp = MvNormalLogp()(cov, delta)
        g_cov, g_delta = at.grad(logp, [cov, delta])
        # TODO: What's the test?  Something needs to be asserted.
        at.grad(g_delta.sum() + g_cov.sum(), [delta, cov])


class TestSplineWrapper:
    @aesara.config.change_flags(compute_test_value="ignore")
    def test_grad(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        verify_grad(spline, [0.5])

    @aesara.config.change_flags(compute_test_value="ignore")
    def test_hessian(self):
        x = np.linspace(0, 1, 100)
        y = x * x
        spline = SplineWrapper(interpolate.InterpolatedUnivariateSpline(x, y, k=1))
        x_var = at.dscalar("x")
        (g_x,) = at.grad(spline(x_var), [x_var])
        with pytest.raises(NotImplementedError):
            at.grad(g_x, [x_var])


class TestI0e:
    @aesara.config.change_flags(compute_test_value="ignore")
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


class TestBetaIncGrad:

    # This test replicates the one used by STAN in here:
    # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/grad_reg_inc_beta_test.cpp
    @aesara.config.change_flags(compute_test_value="ignore")
    @pytest.mark.parametrize(
        "test_a, test_b, test_z, expected_dda, expected_ddb",
        [
            (1.0, 1.0, 1.0, 0, np.nan),
            (1.0, 1.0, 0.4, -0.36651629, 0.30649537),
        ],
    )
    def test_stan_grad_combined(self, test_a, test_b, test_z, expected_dda, expected_ddb):
        a, b, z = at.scalars("a", "b", "z")
        betainc_out = betainc(a, b, z)
        betainc_grad = at.grad(betainc_out, [a, b])
        f_grad = aesara.function([a, b, z], betainc_grad)

        npt.assert_allclose(f_grad(test_a, test_b, test_z), [expected_dda, expected_ddb])

    # This test combines the following STAN tests:
    # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_dda_test.cpp
    # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddb_test.cpp
    # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddz_test.cpp
    @aesara.config.change_flags(compute_test_value="ignore")
    @pytest.mark.parametrize(
        "test_a, test_b, test_z, expected_dda, expected_ddb, expected_ddz",
        [
            (1.5, 1.25, 0.001, -0.00028665637, 4.41357328e-05, 0.063300692),
            (1.5, 1.25, 0.5, -0.26038693947, 0.29301795, 1.1905416),
            (1.5, 1.25, 0.6, -0.23806757, 0.32279575, 1.23341068),
            (1.5, 1.25, 0.999, -0.00022264493, 0.0018969609, 0.35587692),
            (15000, 1.25, 0.001, 0, 0, 0),
            (15000, 1.25, 0.5, 0, 0, 0),
            (15000, 1.25, 0.6, 0, 0, 0),
            (15000, 1.25, 0.999, -6.59543226e-10, 2.00849793e-06, 0.009898182),
            (1.5, 12500, 0.001, -3.93756641e-05, 1.47821755e-09, 0.1848717),
            (1.5, 12500, 0.5, 0, 0, 0),
            (1.5, 12500, 0.6, 0, 0, 0),
            (1.5, 12500, 0.999, 0, 0, 0),
            (15000, 12500, 0.001, 0, 0, 0),
            (15000, 12500, 0.5, -8.72102443e-53, 9.55282792e-53, 5.01131256e-48),
            (15000, 12500, 0.6, -4.085621e-14, -5.5067062e-14, 1.15135267e-71),
            (15000, 12500, 0.999, 0, 0, 0),
        ],
    )
    def test_stan_grad_partials(
        self, test_a, test_b, test_z, expected_dda, expected_ddb, expected_ddz
    ):
        a, b, z = at.scalars("a", "b", "z")
        betainc_out = betainc(a, b, z)
        betainc_grad = at.grad(betainc_out, [a, b, z])
        f_grad = aesara.function([a, b, z], betainc_grad)

        npt.assert_almost_equal(
            f_grad(test_a, test_b, test_z),
            [expected_dda, expected_ddb, expected_ddz],
            select_by_precision(float64=7, float32=3),
        )

    # This test compares against the tabulated values in:
    # Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
    # Journal of Statistical Software, 3(1), 1-20.
    @aesara.config.change_flags(compute_test_value="ignore")
    @pytest.mark.parametrize(
        "test_a, test_b, test_z, expected_dda, expected_ddb",
        [
            (1.5, 11.0, 0.001, -4.5720356e-03, 1.1845673e-04),
            (1.5, 11.0, 0.5, -2.5501997e-03, 9.0824388e-04),
            (1000.0, 1000.0, 0.5, -8.9224793e-03, 8.9224793e-03),
            (1000.0, 1000.0, 0.55, -3.6713108e-07, 4.0584118e-07),
        ],
    )
    def test_boik_robison_cox(self, test_a, test_b, test_z, expected_dda, expected_ddb):
        a, b, z = at.scalars("a", "b", "z")
        betainc_out = betainc(a, b, z)
        betainc_grad = at.grad(betainc_out, [a, b])
        f_grad = aesara.function([a, b, z], betainc_grad)
        npt.assert_almost_equal(
            f_grad(test_a, test_b, test_z),
            [expected_dda, expected_ddb],
        )

    @aesara.config.change_flags(compute_test_value="ignore")
    @pytest.mark.parametrize("test_a", [0.1, 3.0, 1000.0])
    @pytest.mark.parametrize("test_b", [0.1, 1.0, 30.0, 70.0])
    @pytest.mark.parametrize("test_z", [0.01, 0.1, 0.5, 0.7, 0.99])
    def test_aesara_grad(self, test_a, test_b, test_z):
        verify_grad(betainc, [test_a, test_b, test_z])
