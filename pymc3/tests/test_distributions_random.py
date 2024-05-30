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

import itertools
import sys

from contextlib import ExitStack as does_not_raise

import numpy as np
import numpy.random as nr
import numpy.testing as npt
import pytest
import scipy.stats as st
import theano

from scipy import linalg
from scipy.special import expit

import pymc3 as pm

from pymc3.distributions.dist_math import clipped_beta_rvs
from pymc3.distributions.distribution import (
    _DrawValuesContext,
    _DrawValuesContextBlocker,
    draw_values,
    to_tuple,
)
from pymc3.exceptions import ShapeError
from pymc3.tests.helpers import SeededTest
from pymc3.tests.test_distributions import (
    Domain,
    I,
    Nat,
    NatSmall,
    PdMatrix,
    PdMatrixChol,
    PdMatrixCholUpper,
    R,
    RandomPdMatrix,
    RealMatrix,
    Rplus,
    Rplusbig,
    Rplusdunif,
    Runif,
    Simplex,
    Unit,
    Vector,
    build_model,
    product,
)


def pymc3_random(
    dist,
    paramdomains,
    ref_rand,
    valuedomain=Domain([0]),
    size=10000,
    alpha=0.05,
    fails=10,
    extra_args=None,
    model_args=None,
):
    if model_args is None:
        model_args = {}
    model = build_model(dist, valuedomain, paramdomains, extra_args)
    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        pt.update(model_args)
        p = alpha
        # Allow KS test to fail (i.e., the samples be different)
        # a certain number of times. Crude, but necessary.
        f = fails
        while p <= alpha and f > 0:
            s0 = model.named_vars["value"].random(size=size, point=pt)
            s1 = ref_rand(size=size, **pt)
            _, p = st.ks_2samp(np.atleast_1d(s0).flatten(), np.atleast_1d(s1).flatten())
            f -= 1
        assert p > alpha, str(pt)


def pymc3_random_discrete(
    dist, paramdomains, valuedomain=Domain([0]), ref_rand=None, size=100000, alpha=0.05, fails=20
):
    model = build_model(dist, valuedomain, paramdomains)
    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        p = alpha
        # Allow Chisq test to fail (i.e., the samples be different)
        # a certain number of times.
        f = fails
        while p <= alpha and f > 0:
            o = model.named_vars["value"].random(size=size, point=pt)
            e = ref_rand(size=size, **pt)
            o = np.atleast_1d(o).flatten()
            e = np.atleast_1d(e).flatten()
            observed, _ = np.histogram(o, bins=min(7, len(set(o))))
            expected, _ = np.histogram(e, bins=min(7, len(set(o))))
            if np.all(observed == expected):
                p = 1.0
            else:
                _, p = st.chisquare(observed + 1, expected + 1)
            f -= 1
        assert p > alpha, str(pt)


class TestDrawValues(SeededTest):
    def test_draw_scalar_parameters(self):
        with pm.Model():
            y = pm.Normal("y1", mu=0.0, sigma=1.0)
            mu, tau = draw_values([y.distribution.mu, y.distribution.tau])
        npt.assert_almost_equal(mu, 0)
        npt.assert_almost_equal(tau, 1)

    def test_draw_dependencies(self):
        with pm.Model():
            x = pm.Normal("x", mu=0.0, sigma=1.0)
            exp_x = pm.Deterministic("exp_x", pm.math.exp(x))

        x, exp_x = draw_values([x, exp_x])
        npt.assert_almost_equal(np.exp(x), exp_x)

    def test_draw_order(self):
        with pm.Model():
            x = pm.Normal("x", mu=0.0, sigma=1.0)
            exp_x = pm.Deterministic("exp_x", pm.math.exp(x))

        # Need to draw x before drawing log_x
        exp_x, x = draw_values([exp_x, x])
        npt.assert_almost_equal(np.exp(x), exp_x)

    def test_draw_point_replacement(self):
        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, tau=1e-3)
            sigma = pm.Gamma("sigma", alpha=1.0, beta=1.0, transform=None)
            y = pm.Normal("y", mu=mu, sigma=sigma)
            mu2, tau2 = draw_values(
                [y.distribution.mu, y.distribution.tau], point={"mu": 5.0, "sigma": 2.0}
            )
        npt.assert_almost_equal(mu2, 5)
        npt.assert_almost_equal(tau2, 1 / 2.0 ** 2)

    def test_random_sample_returns_nd_array(self):
        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, tau=1e-3)
            sigma = pm.Gamma("sigma", alpha=1.0, beta=1.0, transform=None)
            y = pm.Normal("y", mu=mu, sigma=sigma)
            mu, tau = draw_values([y.distribution.mu, y.distribution.tau])
        assert isinstance(mu, np.ndarray)
        assert isinstance(tau, np.ndarray)


class TestDrawValuesContext:
    def test_normal_context(self):
        with _DrawValuesContext() as context0:
            assert context0.parent is None
            context0.drawn_vars["root_test"] = 1
            with _DrawValuesContext() as context1:
                assert id(context1.drawn_vars) == id(context0.drawn_vars)
                assert context1.parent == context0
                with _DrawValuesContext() as context2:
                    assert id(context2.drawn_vars) == id(context0.drawn_vars)
                    assert context2.parent == context1
                    context2.drawn_vars["leaf_test"] = 2
                assert context1.drawn_vars["leaf_test"] == 2
                context1.drawn_vars["root_test"] = 3
            assert context0.drawn_vars["root_test"] == 3
            assert context0.drawn_vars["leaf_test"] == 2

    def test_blocking_context(self):
        with _DrawValuesContext() as context0:
            assert context0.parent is None
            context0.drawn_vars["root_test"] = 1
            with _DrawValuesContext() as context1:
                assert id(context1.drawn_vars) == id(context0.drawn_vars)
                assert context1.parent == context0
                with _DrawValuesContextBlocker() as blocker:
                    assert id(blocker.drawn_vars) != id(context0.drawn_vars)
                    assert blocker.parent is None
                    blocker.drawn_vars["root_test"] = 2
                    with _DrawValuesContext() as context2:
                        assert id(context2.drawn_vars) == id(blocker.drawn_vars)
                        assert context2.parent == blocker
                        context2.drawn_vars["root_test"] = 3
                        context2.drawn_vars["leaf_test"] = 4
                    assert blocker.drawn_vars["root_test"] == 3
                assert "leaf_test" not in context1.drawn_vars
            assert context0.drawn_vars["root_test"] == 1


class BaseTestCases:
    class BaseTestCase(SeededTest):
        shape = 5
        # the following are the default values of the distribution that take effect
        # when the parametrized shape/size in the test case is None.
        # For every distribution that defaults to non-scalar shapes they must be
        # specified by the inheriting Test class. example: TestGaussianRandomWalk
        default_shape = ()
        default_size = ()

        def setup_method(self, *args, **kwargs):
            super().setup_method(*args, **kwargs)
            self.model = pm.Model()

        def get_random_variable(self, shape, with_vector_params=False, name=None):
            """Creates a RandomVariable of the parametrized distribution."""
            if with_vector_params:
                params = {
                    key: value * np.ones(self.shape, dtype=np.dtype(type(value)))
                    for key, value in self.params.items()
                }
            else:
                params = self.params
            if name is None:
                name = self.distribution.__name__
            with self.model:
                try:
                    if shape is None:
                        # in the test case parametrization "None" means "no specified (default)"
                        return self.distribution(name, transform=None, **params)
                    else:
                        return self.distribution(name, shape=shape, transform=None, **params)
                except TypeError:
                    if np.sum(np.atleast_1d(shape)) == 0:
                        pytest.skip("Timeseries must have positive shape")
                    raise

        @staticmethod
        def sample_random_variable(random_variable, size):
            """Draws samples from a RandomVariable using its .random() method."""
            try:
                if size is None:
                    return random_variable.random()
                else:
                    return random_variable.random(size=size)
            except AttributeError:
                if size is None:
                    return random_variable.distribution.random()
                else:
                    return random_variable.distribution.random(size=size)

        @pytest.mark.parametrize("size", [None, (), 1, (1,), 5, (4, 5)], ids=str)
        @pytest.mark.parametrize("shape", [None, ()], ids=str)
        def test_scalar_distribution_shape(self, shape, size):
            """Draws samples of different [size] from a scalar [shape] RV."""
            rv = self.get_random_variable(shape)
            exp_shape = self.default_shape if shape is None else tuple(np.atleast_1d(shape))
            exp_size = self.default_size if size is None else tuple(np.atleast_1d(size))
            expected = exp_size + exp_shape
            actual = np.shape(self.sample_random_variable(rv, size))
            assert (
                expected == actual
            ), f"Sample size {size} from {shape}-shaped RV had shape {actual}. Expected: {expected}"
            # check that negative size raises an error
            with pytest.raises(ValueError):
                self.sample_random_variable(rv, size=-2)
            with pytest.raises(ValueError):
                self.sample_random_variable(rv, size=(3, -2))

        @pytest.mark.parametrize("size", [None, ()], ids=str)
        @pytest.mark.parametrize(
            "shape", [None, (), (1,), (1, 1), (1, 2), (10, 11, 1), (9, 10, 2)], ids=str
        )
        def test_scalar_sample_shape(self, shape, size):
            """Draws samples of scalar [size] from a [shape] RV."""
            rv = self.get_random_variable(shape)
            exp_shape = self.default_shape if shape is None else tuple(np.atleast_1d(shape))
            exp_size = self.default_size if size is None else tuple(np.atleast_1d(size))
            expected = exp_size + exp_shape
            actual = np.shape(self.sample_random_variable(rv, size))
            assert (
                expected == actual
            ), f"Sample size {size} from {shape}-shaped RV had shape {actual}. Expected: {expected}"

        @pytest.mark.parametrize("size", [None, 3, (4, 5)], ids=str)
        @pytest.mark.parametrize("shape", [None, 1, (10, 11, 1)], ids=str)
        def test_vector_params(self, shape, size):
            shape = self.shape
            rv = self.get_random_variable(shape, with_vector_params=True)
            exp_shape = self.default_shape if shape is None else tuple(np.atleast_1d(shape))
            exp_size = self.default_size if size is None else tuple(np.atleast_1d(size))
            expected = exp_size + exp_shape
            actual = np.shape(self.sample_random_variable(rv, size))
            assert (
                expected == actual
            ), f"Sample size {size} from {shape}-shaped RV had shape {actual}. Expected: {expected}"

        @pytest.mark.parametrize("shape", [-2, 0, (0,), (2, 0), (5, 0, 3)])
        def test_shape_error_on_zero_shape_rv(self, shape):
            with pytest.raises(ValueError, match="not allowed"):
                self.get_random_variable(shape)


class TestGaussianRandomWalk(BaseTestCases.BaseTestCase):
    distribution = pm.GaussianRandomWalk
    params = {"mu": 1.0, "sigma": 1.0}
    default_shape = (1,)


class TestNormal(BaseTestCases.BaseTestCase):
    distribution = pm.Normal
    params = {"mu": 0.0, "tau": 1.0}


class TestTruncatedNormal(BaseTestCases.BaseTestCase):
    distribution = pm.TruncatedNormal
    params = {"mu": 0.0, "tau": 1.0, "lower": -0.5, "upper": 0.5}


class TestTruncatedNormalLower(BaseTestCases.BaseTestCase):
    distribution = pm.TruncatedNormal
    params = {"mu": 0.0, "tau": 1.0, "lower": -0.5}


class TestTruncatedNormalUpper(BaseTestCases.BaseTestCase):
    distribution = pm.TruncatedNormal
    params = {"mu": 0.0, "tau": 1.0, "upper": 0.5}


class TestSkewNormal(BaseTestCases.BaseTestCase):
    distribution = pm.SkewNormal
    params = {"mu": 0.0, "sigma": 1.0, "alpha": 5.0}


class TestHalfNormal(BaseTestCases.BaseTestCase):
    distribution = pm.HalfNormal
    params = {"tau": 1.0}


class TestUniform(BaseTestCases.BaseTestCase):
    distribution = pm.Uniform
    params = {"lower": 0.0, "upper": 1.0}


class TestTriangular(BaseTestCases.BaseTestCase):
    distribution = pm.Triangular
    params = {"c": 0.5, "lower": 0.0, "upper": 1.0}


class TestWald(BaseTestCases.BaseTestCase):
    distribution = pm.Wald
    params = {"mu": 1.0, "lam": 1.0, "alpha": 0.0}


class TestBeta(BaseTestCases.BaseTestCase):
    distribution = pm.Beta
    params = {"alpha": 1.0, "beta": 1.0}


class TestKumaraswamy(BaseTestCases.BaseTestCase):
    distribution = pm.Kumaraswamy
    params = {"a": 1.0, "b": 1.0}


class TestExponential(BaseTestCases.BaseTestCase):
    distribution = pm.Exponential
    params = {"lam": 1.0}


class TestLaplace(BaseTestCases.BaseTestCase):
    distribution = pm.Laplace
    params = {"mu": 1.0, "b": 1.0}


class TestAsymmetricLaplace(BaseTestCases.BaseTestCase):
    distribution = pm.AsymmetricLaplace
    params = {"kappa": 1.0, "b": 1.0, "mu": 0.0}


class TestLognormal(BaseTestCases.BaseTestCase):
    distribution = pm.Lognormal
    params = {"mu": 1.0, "tau": 1.0}


class TestStudentT(BaseTestCases.BaseTestCase):
    distribution = pm.StudentT
    params = {"nu": 5.0, "mu": 0.0, "lam": 1.0}


class TestPareto(BaseTestCases.BaseTestCase):
    distribution = pm.Pareto
    params = {"alpha": 0.5, "m": 1.0}


class TestCauchy(BaseTestCases.BaseTestCase):
    distribution = pm.Cauchy
    params = {"alpha": 1.0, "beta": 1.0}


class TestHalfCauchy(BaseTestCases.BaseTestCase):
    distribution = pm.HalfCauchy
    params = {"beta": 1.0}


class TestGamma(BaseTestCases.BaseTestCase):
    distribution = pm.Gamma
    params = {"alpha": 1.0, "beta": 1.0}


class TestInverseGamma(BaseTestCases.BaseTestCase):
    distribution = pm.InverseGamma
    params = {"alpha": 0.5, "beta": 0.5}


class TestChiSquared(BaseTestCases.BaseTestCase):
    distribution = pm.ChiSquared
    params = {"nu": 2.0}


class TestWeibull(BaseTestCases.BaseTestCase):
    distribution = pm.Weibull
    params = {"alpha": 1.0, "beta": 1.0}


class TestExGaussian(BaseTestCases.BaseTestCase):
    distribution = pm.ExGaussian
    params = {"mu": 0.0, "sigma": 1.0, "nu": 1.0}


class TestVonMises(BaseTestCases.BaseTestCase):
    distribution = pm.VonMises
    params = {"mu": 0.0, "kappa": 1.0}


class TestGumbel(BaseTestCases.BaseTestCase):
    distribution = pm.Gumbel
    params = {"mu": 0.0, "beta": 1.0}


class TestLogistic(BaseTestCases.BaseTestCase):
    distribution = pm.Logistic
    params = {"mu": 0.0, "s": 1.0}


class TestLogitNormal(BaseTestCases.BaseTestCase):
    distribution = pm.LogitNormal
    params = {"mu": 0.0, "sigma": 1.0}


class TestBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.Binomial
    params = {"n": 5, "p": 0.5}


class TestBetaBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.BetaBinomial
    params = {"n": 5, "alpha": 1.0, "beta": 1.0}


class TestBernoulli(BaseTestCases.BaseTestCase):
    distribution = pm.Bernoulli
    params = {"p": 0.5}


class TestDiscreteWeibull(BaseTestCases.BaseTestCase):
    distribution = pm.DiscreteWeibull
    params = {"q": 0.25, "beta": 2.0}


class TestPoisson(BaseTestCases.BaseTestCase):
    distribution = pm.Poisson
    params = {"mu": 1.0}


class TestNegativeBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.NegativeBinomial
    params = {"mu": 1.0, "alpha": 1.0}


class TestConstant(BaseTestCases.BaseTestCase):
    distribution = pm.Constant
    params = {"c": 3}


class TestZeroInflatedPoisson(BaseTestCases.BaseTestCase):
    distribution = pm.ZeroInflatedPoisson
    params = {"theta": 1.0, "psi": 0.3}


class TestZeroInflatedNegativeBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.ZeroInflatedNegativeBinomial
    params = {"mu": 1.0, "alpha": 1.0, "psi": 0.3}


class TestZeroInflatedBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.ZeroInflatedBinomial
    params = {"n": 10, "p": 0.6, "psi": 0.3}


class TestDiscreteUniform(BaseTestCases.BaseTestCase):
    distribution = pm.DiscreteUniform
    params = {"lower": 0.0, "upper": 10.0}


class TestGeometric(BaseTestCases.BaseTestCase):
    distribution = pm.Geometric
    params = {"p": 0.5}


class TestHyperGeometric(BaseTestCases.BaseTestCase):
    distribution = pm.HyperGeometric
    params = {"N": 50, "k": 25, "n": 10}


class TestMoyal(BaseTestCases.BaseTestCase):
    distribution = pm.Moyal
    params = {"mu": 0.0, "sigma": 1.0}


class TestCategorical(BaseTestCases.BaseTestCase):
    distribution = pm.Categorical
    params = {"p": np.ones(BaseTestCases.BaseTestCase.shape)}

    def get_random_variable(
        self, shape, with_vector_params=False, **kwargs
    ):  # don't transform categories
        return super().get_random_variable(shape, with_vector_params=False, **kwargs)

    def test_probability_vector_shape(self):
        """Check that if a 2d array of probabilities are passed to categorical correct shape is returned"""
        p = np.ones((10, 5))
        assert pm.Categorical.dist(p=p).random().shape == (10,)
        assert pm.Categorical.dist(p=p).random(size=4).shape == (4, 10)
        p = np.ones((3, 7, 5))
        assert pm.Categorical.dist(p=p).random().shape == (3, 7)
        assert pm.Categorical.dist(p=p).random(size=4).shape == (4, 3, 7)


class TestDirichlet(SeededTest):
    @pytest.mark.parametrize(
        "shape, size",
        [
            ((2), (1)),
            ((2), (2)),
            ((2, 2), (2, 100)),
            ((3, 4), (3, 4)),
            ((3, 4), (3, 4, 100)),
            ((3, 4), (100)),
            ((3, 4), (1)),
        ],
    )
    def test_dirichlet_random_shape(self, shape, size):
        out_shape = to_tuple(size) + to_tuple(shape)
        assert pm.Dirichlet.dist(a=np.ones(shape)).random(size=size).shape == out_shape


class TestScalarParameterSamples(SeededTest):
    def test_bounded(self):
        # A bit crude...
        BoundedNormal = pm.Bound(pm.Normal, upper=0)

        def ref_rand(size, tau):
            return -st.halfnorm.rvs(size=size, loc=0, scale=tau ** -0.5)

        pymc3_random(BoundedNormal, {"tau": Rplus}, ref_rand=ref_rand)

    def test_uniform(self):
        def ref_rand(size, lower, upper):
            return st.uniform.rvs(size=size, loc=lower, scale=upper - lower)

        pymc3_random(pm.Uniform, {"lower": -Rplus, "upper": Rplus}, ref_rand=ref_rand)

    def test_normal(self):
        def ref_rand(size, mu, sigma):
            return st.norm.rvs(size=size, loc=mu, scale=sigma)

        pymc3_random(pm.Normal, {"mu": R, "sigma": Rplus}, ref_rand=ref_rand)

    def test_truncated_normal(self):
        def ref_rand(size, mu, sigma, lower, upper):
            return st.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, size=size, loc=mu, scale=sigma
            )

        pymc3_random(
            pm.TruncatedNormal,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig, "upper": Rplusbig},
            ref_rand=ref_rand,
        )

    def test_truncated_normal_lower(self):
        def ref_rand(size, mu, sigma, lower):
            return st.truncnorm.rvs((lower - mu) / sigma, np.inf, size=size, loc=mu, scale=sigma)

        pymc3_random(
            pm.TruncatedNormal, {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig}, ref_rand=ref_rand
        )

    def test_truncated_normal_upper(self):
        def ref_rand(size, mu, sigma, upper):
            return st.truncnorm.rvs(-np.inf, (upper - mu) / sigma, size=size, loc=mu, scale=sigma)

        pymc3_random(
            pm.TruncatedNormal, {"mu": R, "sigma": Rplusbig, "upper": Rplusbig}, ref_rand=ref_rand
        )

    def test_skew_normal(self):
        def ref_rand(size, alpha, mu, sigma):
            return st.skewnorm.rvs(size=size, a=alpha, loc=mu, scale=sigma)

        pymc3_random(pm.SkewNormal, {"mu": R, "sigma": Rplus, "alpha": R}, ref_rand=ref_rand)

    def test_half_normal(self):
        def ref_rand(size, tau):
            return st.halfnorm.rvs(size=size, loc=0, scale=tau ** -0.5)

        pymc3_random(pm.HalfNormal, {"tau": Rplus}, ref_rand=ref_rand)

    def test_wald(self):
        # Cannot do anything too exciting as scipy wald is a
        # location-scale model of the *standard* wald with mu=1 and lam=1
        def ref_rand(size, mu, lam, alpha):
            return st.wald.rvs(size=size, loc=alpha)

        pymc3_random(
            pm.Wald,
            {"mu": Domain([1.0, 1.0, 1.0]), "lam": Domain([1.0, 1.0, 1.0]), "alpha": Rplus},
            ref_rand=ref_rand,
        )

    def test_beta(self):
        def ref_rand(size, alpha, beta):
            return clipped_beta_rvs(a=alpha, b=beta, size=size)

        pymc3_random(pm.Beta, {"alpha": Rplus, "beta": Rplus}, ref_rand=ref_rand)

    def test_exponential(self):
        def ref_rand(size, lam):
            return nr.exponential(scale=1.0 / lam, size=size)

        pymc3_random(pm.Exponential, {"lam": Rplus}, ref_rand=ref_rand)

    def test_laplace(self):
        def ref_rand(size, mu, b):
            return st.laplace.rvs(mu, b, size=size)

        pymc3_random(pm.Laplace, {"mu": R, "b": Rplus}, ref_rand=ref_rand)

    def test_laplace_asymmetric(self):
        def ref_rand(size, kappa, b, mu):
            u = np.random.uniform(size=size)
            switch = kappa ** 2 / (1 + kappa ** 2)
            non_positive_x = mu + kappa * np.log(u * (1 / switch)) / b
            positive_x = mu - np.log((1 - u) * (1 + kappa ** 2)) / (kappa * b)
            draws = non_positive_x * (u <= switch) + positive_x * (u > switch)
            return draws

        pymc3_random(pm.AsymmetricLaplace, {"b": Rplus, "kappa": Rplus, "mu": R}, ref_rand=ref_rand)

    def test_lognormal(self):
        def ref_rand(size, mu, tau):
            return np.exp(mu + (tau ** -0.5) * st.norm.rvs(loc=0.0, scale=1.0, size=size))

        pymc3_random(pm.Lognormal, {"mu": R, "tau": Rplusbig}, ref_rand=ref_rand)

    def test_student_t(self):
        def ref_rand(size, nu, mu, lam):
            return st.t.rvs(nu, mu, lam ** -0.5, size=size)

        pymc3_random(pm.StudentT, {"nu": Rplus, "mu": R, "lam": Rplus}, ref_rand=ref_rand)

    def test_cauchy(self):
        def ref_rand(size, alpha, beta):
            return st.cauchy.rvs(alpha, beta, size=size)

        pymc3_random(pm.Cauchy, {"alpha": R, "beta": Rplusbig}, ref_rand=ref_rand)

    def test_half_cauchy(self):
        def ref_rand(size, beta):
            return st.halfcauchy.rvs(scale=beta, size=size)

        pymc3_random(pm.HalfCauchy, {"beta": Rplusbig}, ref_rand=ref_rand)

    def test_gamma_alpha_beta(self):
        def ref_rand(size, alpha, beta):
            return st.gamma.rvs(alpha, scale=1.0 / beta, size=size)

        pymc3_random(pm.Gamma, {"alpha": Rplusbig, "beta": Rplusbig}, ref_rand=ref_rand)

    def test_gamma_mu_sigma(self):
        def ref_rand(size, mu, sigma):
            return st.gamma.rvs(mu ** 2 / sigma ** 2, scale=sigma ** 2 / mu, size=size)

        pymc3_random(pm.Gamma, {"mu": Rplusbig, "sigma": Rplusbig}, ref_rand=ref_rand)

    def test_inverse_gamma(self):
        def ref_rand(size, alpha, beta):
            return st.invgamma.rvs(a=alpha, scale=beta, size=size)

        pymc3_random(pm.InverseGamma, {"alpha": Rplus, "beta": Rplus}, ref_rand=ref_rand)

    def test_pareto(self):
        def ref_rand(size, alpha, m):
            return st.pareto.rvs(alpha, scale=m, size=size)

        pymc3_random(pm.Pareto, {"alpha": Rplusbig, "m": Rplusbig}, ref_rand=ref_rand)

    def test_ex_gaussian(self):
        def ref_rand(size, mu, sigma, nu):
            return nr.normal(mu, sigma, size=size) + nr.exponential(scale=nu, size=size)

        pymc3_random(pm.ExGaussian, {"mu": R, "sigma": Rplus, "nu": Rplus}, ref_rand=ref_rand)

    def test_vonmises(self):
        def ref_rand(size, mu, kappa):
            return st.vonmises.rvs(size=size, loc=mu, kappa=kappa)

        pymc3_random(pm.VonMises, {"mu": R, "kappa": Rplus}, ref_rand=ref_rand)

    def test_triangular(self):
        def ref_rand(size, lower, upper, c):
            scale = upper - lower
            c_ = (c - lower) / scale
            return st.triang.rvs(size=size, loc=lower, scale=scale, c=c_)

        pymc3_random(
            pm.Triangular, {"lower": Runif, "upper": Runif + 3, "c": Runif + 1}, ref_rand=ref_rand
        )

    def test_flat(self):
        with pm.Model():
            f = pm.Flat("f")
            with pytest.raises(ValueError):
                f.random(1)

    def test_half_flat(self):
        with pm.Model():
            f = pm.HalfFlat("f")
            with pytest.raises(ValueError):
                f.random(1)

    def test_binomial(self):
        pymc3_random_discrete(pm.Binomial, {"n": Nat, "p": Unit}, ref_rand=st.binom.rvs)

    @pytest.mark.xfail(
        sys.platform.startswith("win"),
        reason="Known issue: https://github.com/pymc-devs/pymc3/pull/4269",
    )
    def test_beta_binomial(self):
        pymc3_random_discrete(
            pm.BetaBinomial, {"n": Nat, "alpha": Rplus, "beta": Rplus}, ref_rand=self._beta_bin
        )

    def _beta_bin(self, n, alpha, beta, size=None):
        return st.binom.rvs(n, st.beta.rvs(a=alpha, b=beta, size=size))

    def test_bernoulli(self):
        pymc3_random_discrete(
            pm.Bernoulli, {"p": Unit}, ref_rand=lambda size, p=None: st.bernoulli.rvs(p, size=size)
        )

    def test_poisson(self):
        pymc3_random_discrete(pm.Poisson, {"mu": Rplusbig}, size=500, ref_rand=st.poisson.rvs)

    def test_negative_binomial(self):
        def ref_rand(size, alpha, mu):
            return st.nbinom.rvs(alpha, alpha / (mu + alpha), size=size)

        pymc3_random_discrete(
            pm.NegativeBinomial,
            {"mu": Rplusbig, "alpha": Rplusbig},
            size=100,
            fails=50,
            ref_rand=ref_rand,
        )

    def test_geometric(self):
        pymc3_random_discrete(pm.Geometric, {"p": Unit}, size=500, fails=50, ref_rand=nr.geometric)

    def test_hypergeometric(self):
        def ref_rand(size, N, k, n):
            return st.hypergeom.rvs(M=N, n=k, N=n, size=size)

        pymc3_random_discrete(
            pm.HyperGeometric,
            {
                "N": Domain([10, 11, 12, 13], "int64"),
                "k": Domain([4, 5, 6, 7], "int64"),
                "n": Domain([6, 7, 8, 9], "int64"),
            },
            size=500,
            fails=50,
            ref_rand=ref_rand,
        )

    def test_discrete_uniform(self):
        def ref_rand(size, lower, upper):
            return st.randint.rvs(lower, upper + 1, size=size)

        pymc3_random_discrete(
            pm.DiscreteUniform, {"lower": -NatSmall, "upper": NatSmall}, ref_rand=ref_rand
        )

    def test_discrete_weibull(self):
        def ref_rand(size, q, beta):
            u = np.random.uniform(size=size)

            return np.ceil(np.power(np.log(1 - u) / np.log(q), 1.0 / beta)) - 1

        pymc3_random_discrete(
            pm.DiscreteWeibull, {"q": Unit, "beta": Rplusdunif}, ref_rand=ref_rand
        )

    @pytest.mark.parametrize("s", [2, 3, 4])
    def test_categorical_random(self, s):
        def ref_rand(size, p):
            return nr.choice(np.arange(p.shape[0]), p=p, size=size)

        pymc3_random_discrete(pm.Categorical, {"p": Simplex(s)}, ref_rand=ref_rand)

    def test_constant_dist(self):
        def ref_rand(size, c):
            return c * np.ones(size, dtype=int)

        pymc3_random_discrete(pm.Constant, {"c": I}, ref_rand=ref_rand)

    def test_mv_normal(self):
        def ref_rand(size, mu, cov):
            return st.multivariate_normal.rvs(mean=mu, cov=cov, size=size)

        def ref_rand_tau(size, mu, tau):
            return ref_rand(size, mu, linalg.inv(tau))

        def ref_rand_chol(size, mu, chol):
            return ref_rand(size, mu, np.dot(chol, chol.T))

        def ref_rand_uchol(size, mu, chol):
            return ref_rand(size, mu, np.dot(chol.T, chol))

        for n in [2, 3]:
            pymc3_random(
                pm.MvNormal,
                {"mu": Vector(R, n), "cov": PdMatrix(n)},
                size=100,
                valuedomain=Vector(R, n),
                ref_rand=ref_rand,
            )
            pymc3_random(
                pm.MvNormal,
                {"mu": Vector(R, n), "tau": PdMatrix(n)},
                size=100,
                valuedomain=Vector(R, n),
                ref_rand=ref_rand_tau,
            )
            pymc3_random(
                pm.MvNormal,
                {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
                size=100,
                valuedomain=Vector(R, n),
                ref_rand=ref_rand_chol,
            )
            pymc3_random(
                pm.MvNormal,
                {"mu": Vector(R, n), "chol": PdMatrixCholUpper(n)},
                size=100,
                valuedomain=Vector(R, n),
                ref_rand=ref_rand_uchol,
                extra_args={"lower": False},
            )

    def test_matrix_normal(self):
        def ref_rand(size, mu, rowcov, colcov):
            return st.matrix_normal.rvs(mean=mu, rowcov=rowcov, colcov=colcov, size=size)

        # def ref_rand_tau(size, mu, tau):
        #     return ref_rand(size, mu, linalg.inv(tau))

        def ref_rand_chol(size, mu, rowchol, colchol):
            return ref_rand(
                size, mu, rowcov=np.dot(rowchol, rowchol.T), colcov=np.dot(colchol, colchol.T)
            )

        def ref_rand_chol_transpose(size, mu, rowchol, colchol):
            colchol = colchol.T
            return ref_rand(
                size, mu, rowcov=np.dot(rowchol, rowchol.T), colcov=np.dot(colchol, colchol.T)
            )

        def ref_rand_uchol(size, mu, rowchol, colchol):
            return ref_rand(
                size, mu, rowcov=np.dot(rowchol.T, rowchol), colcov=np.dot(colchol.T, colchol)
            )

        for n in [2, 3]:
            pymc3_random(
                pm.MatrixNormal,
                {"mu": RealMatrix(n, n), "rowcov": PdMatrix(n), "colcov": PdMatrix(n)},
                size=100,
                valuedomain=RealMatrix(n, n),
                ref_rand=ref_rand,
            )
            # pymc3_random(pm.MatrixNormal, {'mu': RealMatrix(n, n), 'tau': PdMatrix(n)},
            #              size=n, valuedomain=RealMatrix(n, n), ref_rand=ref_rand_tau)
            pymc3_random(
                pm.MatrixNormal,
                {"mu": RealMatrix(n, n), "rowchol": PdMatrixChol(n), "colchol": PdMatrixChol(n)},
                size=100,
                valuedomain=RealMatrix(n, n),
                ref_rand=ref_rand_chol,
            )
            # pymc3_random(
            #     pm.MvNormal,
            #     {'mu': RealMatrix(n, n), 'rowchol': PdMatrixCholUpper(n), 'colchol': PdMatrixCholUpper(n)},
            #     size=n, valuedomain=RealMatrix(n, n), ref_rand=ref_rand_uchol,
            #     extra_args={'lower': False}
            # )

            # 2 sample test fails because cov becomes different if chol is transposed beforehand.
            # This implicity means we need transpose of chol after drawing values in
            # MatrixNormal.random method to match stats.matrix_normal.rvs method
            with pytest.raises(AssertionError):
                pymc3_random(
                    pm.MatrixNormal,
                    {
                        "mu": RealMatrix(n, n),
                        "rowchol": PdMatrixChol(n),
                        "colchol": PdMatrixChol(n),
                    },
                    size=100,
                    valuedomain=RealMatrix(n, n),
                    ref_rand=ref_rand_chol_transpose,
                )

    @pytest.mark.xfail(
        condition=sys.platform.startswith("win"),
        reason="Compilation problems. See https://github.com/pymc-devs/pymc/issues/5253",
    )
    def test_kronecker_normal(self):
        def ref_rand(size, mu, covs, sigma):
            cov = pm.math.kronecker(covs[0], covs[1]).eval()
            cov += sigma ** 2 * np.identity(cov.shape[0])
            return st.multivariate_normal.rvs(mean=mu, cov=cov, size=size)

        def ref_rand_chol(size, mu, chols, sigma):
            covs = [np.dot(chol, chol.T) for chol in chols]
            return ref_rand(size, mu, covs, sigma)

        def ref_rand_evd(size, mu, evds, sigma):
            covs = []
            for eigs, Q in evds:
                covs.append(np.dot(Q, np.dot(np.diag(eigs), Q.T)))
            return ref_rand(size, mu, covs, sigma)

        sizes = [2, 3]
        sigmas = [0, 1]
        for n, sigma in zip(sizes, sigmas):
            N = n ** 2
            covs = [RandomPdMatrix(n), RandomPdMatrix(n)]
            chols = list(map(np.linalg.cholesky, covs))
            evds = list(map(np.linalg.eigh, covs))
            dom = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)
            mu = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)

            std_args = {"mu": mu}
            cov_args = {"covs": covs}
            chol_args = {"chols": chols}
            evd_args = {"evds": evds}
            if sigma is not None and sigma != 0:
                std_args["sigma"] = Domain([sigma], edges=(None, None))
            else:
                for args in [cov_args, chol_args, evd_args]:
                    args["sigma"] = sigma

            pymc3_random(
                pm.KroneckerNormal,
                std_args,
                valuedomain=dom,
                ref_rand=ref_rand,
                extra_args=cov_args,
                model_args=cov_args,
            )
            pymc3_random(
                pm.KroneckerNormal,
                std_args,
                valuedomain=dom,
                ref_rand=ref_rand_chol,
                extra_args=chol_args,
                model_args=chol_args,
            )
            pymc3_random(
                pm.KroneckerNormal,
                std_args,
                valuedomain=dom,
                ref_rand=ref_rand_evd,
                extra_args=evd_args,
                model_args=evd_args,
            )

    def test_mv_t(self):
        def ref_rand(size, nu, Sigma, mu):
            normal = st.multivariate_normal.rvs(cov=Sigma, size=size)
            chi2 = st.chi2.rvs(df=nu, size=size)[..., None]
            return mu + (normal / np.sqrt(chi2 / nu))

        for n in [2, 3]:
            pymc3_random(
                pm.MvStudentT,
                {"nu": Domain([5, 10, 25, 50]), "Sigma": PdMatrix(n), "mu": Vector(R, n)},
                size=100,
                valuedomain=Vector(R, n),
                ref_rand=ref_rand,
            )

    def test_dirichlet(self):
        def ref_rand(size, a):
            return st.dirichlet.rvs(a, size=size)

        for n in [2, 3]:
            pymc3_random(
                pm.Dirichlet,
                {"a": Vector(Rplus, n)},
                valuedomain=Simplex(n),
                size=100,
                ref_rand=ref_rand,
            )

    def test_dirichlet_multinomial(self):
        def ref_rand(size, a, n):
            k = a.shape[-1]
            out = np.empty((size, k), dtype=int)
            for i in range(size):
                p = nr.dirichlet(a)
                x = nr.multinomial(n=n, pvals=p)
                out[i, :] = x
            return out

        for n in [2, 3]:
            pymc3_random_discrete(
                pm.DirichletMultinomial,
                {"a": Vector(Rplus, n), "n": Nat},
                valuedomain=Vector(Nat, n),
                size=1000,
                ref_rand=ref_rand,
            )

    @pytest.mark.parametrize(
        "a, shape, n",
        [
            [[0.25, 0.25, 0.25, 0.25], 4, 2],
            [[0.25, 0.25, 0.25, 0.25], (1, 4), 3],
            [[0.25, 0.25, 0.25, 0.25], (10, 4), [2] * 10],
            [[0.25, 0.25, 0.25, 0.25], (10, 1, 4), 5],
            [[[0.25, 0.25, 0.25, 0.25]], (2, 4), [7, 11]],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (2, 4), 13],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (1, 2, 4), [23, 29]],
            [
                [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
                (10, 2, 4),
                [31, 37],
            ],
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], (2, 4), [17, 19]],
        ],
    )
    def test_dirichlet_multinomial_shape(self, a, shape, n):
        a = np.asarray(a)
        with pm.Model() as model:
            m = pm.DirichletMultinomial("m", n=n, a=a, shape=shape)
        samp0 = m.random()
        samp1 = m.random(size=1)
        samp2 = m.random(size=2)

        shape_ = to_tuple(shape)
        assert to_tuple(samp0.shape) == shape_
        assert to_tuple(samp1.shape) == (1, *shape_)
        assert to_tuple(samp2.shape) == (2, *shape_)

    @pytest.mark.parametrize(
        "n, a, shape, expectation",
        [
            ([5], [[1000, 1, 1], [1, 1, 1000]], (2, 3), does_not_raise()),
            ([5, 3], [[1000, 1, 1], [1, 1, 1000]], (2, 3), does_not_raise()),
            ([[5]], [[1000, 1, 1], [1, 1, 1000]], (2, 3), pytest.raises(ShapeError)),
            ([[5], [3]], [[1000, 1, 1], [1, 1, 1000]], (2, 3), pytest.raises(ShapeError)),
        ],
    )
    def test_dirichlet_multinomial_dist_ShapeError(self, n, a, shape, expectation):
        m = pm.DirichletMultinomial.dist(n=n, a=a, shape=shape)
        with expectation:
            m.random()

    def test_multinomial(self):
        def ref_rand(size, p, n):
            return nr.multinomial(pvals=p, n=n, size=size)

        for n in [2, 3]:
            pymc3_random_discrete(
                pm.Multinomial,
                {"p": Simplex(n), "n": Nat},
                valuedomain=Vector(Nat, n),
                size=100,
                ref_rand=ref_rand,
            )

    def test_gumbel(self):
        def ref_rand(size, mu, beta):
            return st.gumbel_r.rvs(loc=mu, scale=beta, size=size)

        pymc3_random(pm.Gumbel, {"mu": R, "beta": Rplus}, ref_rand=ref_rand)

    def test_logistic(self):
        def ref_rand(size, mu, s):
            return st.logistic.rvs(loc=mu, scale=s, size=size)

        pymc3_random(pm.Logistic, {"mu": R, "s": Rplus}, ref_rand=ref_rand)

    def test_logitnormal(self):
        def ref_rand(size, mu, sigma):
            return expit(st.norm.rvs(loc=mu, scale=sigma, size=size))

        pymc3_random(pm.LogitNormal, {"mu": R, "sigma": Rplus}, ref_rand=ref_rand)

    def test_moyal(self):
        def ref_rand(size, mu, sigma):
            return st.moyal.rvs(loc=mu, scale=sigma, size=size)

        pymc3_random(pm.Moyal, {"mu": R, "sigma": Rplus}, ref_rand=ref_rand)

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_interpolated(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                def ref_rand(size):
                    return st.norm.rvs(loc=mu, scale=sigma, size=size)

                class TestedInterpolated(pm.Interpolated):
                    def __init__(self, **kwargs):
                        x_points = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
                        pdf_points = st.norm.pdf(x_points, loc=mu, scale=sigma)
                        super().__init__(x_points=x_points, pdf_points=pdf_points, **kwargs)

                pymc3_random(TestedInterpolated, {}, ref_rand=ref_rand)

    @pytest.mark.skip(
        "Wishart random sampling not implemented.\n"
        "See https://github.com/pymc-devs/pymc3/issues/538"
    )
    def test_wishart(self):
        # Wishart non current recommended for use:
        # https://github.com/pymc-devs/pymc3/issues/538
        # for n in [2, 3]:
        #     pymc3_random_discrete(Wisvaluedomainhart,
        #                           {'n': Domain([2, 3, 4, 2000]) , 'V': PdMatrix(n) },
        #                           valuedomain=PdMatrix(n),
        #                           ref_rand=lambda n=None, V=None, size=None: \
        #                           st.wishart(V, df=n, size=size))
        pass

    def test_lkj(self):
        for n in [2, 10, 50]:
            # pylint: disable=cell-var-from-loop
            shape = n * (n - 1) // 2

            def ref_rand(size, eta):
                beta = eta - 1 + n / 2
                return (st.beta.rvs(size=(size, shape), a=beta, b=beta) - 0.5) * 2

            class TestedLKJCorr(pm.LKJCorr):
                def __init__(self, **kwargs):
                    kwargs.pop("shape", None)
                    super().__init__(n=n, **kwargs)

            pymc3_random(
                TestedLKJCorr,
                {"eta": Domain([1.0, 10.0, 100.0])},
                size=10000 // n,
                ref_rand=ref_rand,
            )

    def test_normalmixture(self):
        def ref_rand(size, w, mu, sigma):
            component = np.random.choice(w.size, size=size, p=w)
            return np.random.normal(mu[component], sigma[component], size=size)

        pymc3_random(
            pm.NormalMixture,
            {
                "w": Simplex(2),
                "mu": Domain([[0.05, 2.5], [-5.0, 1.0]], edges=(None, None)),
                "sigma": Domain([[1, 1], [1.5, 2.0]], edges=(None, None)),
            },
            extra_args={"comp_shape": 2},
            size=1000,
            ref_rand=ref_rand,
        )
        pymc3_random(
            pm.NormalMixture,
            {
                "w": Simplex(3),
                "mu": Domain([[-5.0, 1.0, 2.5]], edges=(None, None)),
                "sigma": Domain([[1.5, 2.0, 3.0]], edges=(None, None)),
            },
            extra_args={"comp_shape": 3},
            size=1000,
            ref_rand=ref_rand,
        )


def test_mixture_random_shape():
    # test the shape broadcasting in mixture random
    y = np.concatenate([nr.poisson(5, size=10), nr.poisson(9, size=10)])
    with pm.Model() as m:
        comp0 = pm.Poisson.dist(mu=np.ones(2))
        w0 = pm.Dirichlet("w0", a=np.ones(2), shape=(2,))
        like0 = pm.Mixture("like0", w=w0, comp_dists=comp0, observed=y)

        comp1 = pm.Poisson.dist(mu=np.ones((20, 2)), shape=(20, 2))
        w1 = pm.Dirichlet("w1", a=np.ones(2), shape=(2,))
        like1 = pm.Mixture("like1", w=w1, comp_dists=comp1, observed=y)

        comp2 = pm.Poisson.dist(mu=np.ones(2))
        w2 = pm.Dirichlet("w2", a=np.ones(2), shape=(20, 2))
        like2 = pm.Mixture("like2", w=w2, comp_dists=comp2, observed=y)

        comp3 = pm.Poisson.dist(mu=np.ones(2), shape=(20, 2))
        w3 = pm.Dirichlet("w3", a=np.ones(2), shape=(20, 2))
        like3 = pm.Mixture("like3", w=w3, comp_dists=comp3, observed=y)

    rand0, rand1, rand2, rand3 = draw_values(
        [like0, like1, like2, like3], point=m.test_point, size=100
    )
    assert rand0.shape == (100, 20)
    assert rand1.shape == (100, 20)
    assert rand2.shape == (100, 20)
    assert rand3.shape == (100, 20)

    with m:
        ppc = pm.sample_posterior_predictive([m.test_point], samples=200)
    assert ppc["like0"].shape == (200, 20)
    assert ppc["like1"].shape == (200, 20)
    assert ppc["like2"].shape == (200, 20)
    assert ppc["like3"].shape == (200, 20)


@pytest.mark.xfail
def test_mixture_random_shape_fast():
    # test the shape broadcasting in mixture random
    y = np.concatenate([nr.poisson(5, size=10), nr.poisson(9, size=10)])
    with pm.Model() as m:
        comp0 = pm.Poisson.dist(mu=np.ones(2))
        w0 = pm.Dirichlet("w0", a=np.ones(2), shape=(2,))
        like0 = pm.Mixture("like0", w=w0, comp_dists=comp0, observed=y)

        comp1 = pm.Poisson.dist(mu=np.ones((20, 2)), shape=(20, 2))
        w1 = pm.Dirichlet("w1", a=np.ones(2), shape=(2,))
        like1 = pm.Mixture("like1", w=w1, comp_dists=comp1, observed=y)

        comp2 = pm.Poisson.dist(mu=np.ones(2))
        w2 = pm.Dirichlet("w2", a=np.ones(2), shape=(20, 2))
        like2 = pm.Mixture("like2", w=w2, comp_dists=comp2, observed=y)

        comp3 = pm.Poisson.dist(mu=np.ones(2), shape=(20, 2))
        w3 = pm.Dirichlet("w3", a=np.ones(2), shape=(20, 2))
        like3 = pm.Mixture("like3", w=w3, comp_dists=comp3, observed=y)

    rand0, rand1, rand2, rand3 = draw_values(
        [like0, like1, like2, like3], point=m.test_point, size=100
    )
    assert rand0.shape == (100, 20)
    assert rand1.shape == (100, 20)
    assert rand2.shape == (100, 20)
    assert rand3.shape == (100, 20)

    # I *think* that the mixture means that this is not going to work,
    # but I could be wrong. [2019/08/22:rpg]
    with m:
        ppc = pm.fast_sample_posterior_predictive([m.test_point], samples=200)
    assert ppc["like0"].shape == (200, 20)
    assert ppc["like1"].shape == (200, 20)
    assert ppc["like2"].shape == (200, 20)
    assert ppc["like3"].shape == (200, 20)


class TestDensityDist:
    @pytest.mark.parametrize("shape", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random_sampleable(self, shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1, shape=shape)
            obs = pm.DensityDist(
                "density_dist",
                normal_dist.logp,
                observed=np.random.randn(100, *shape),
                shape=shape,
                random=normal_dist.random,
            )
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        size = 100
        ppc = pm.sample_posterior_predictive(trace, samples=samples, model=model, size=size)
        assert ppc["density_dist"].shape == (samples, size) + obs.distribution.shape

        # ppc = pm.fast_sample_posterior_predictive(trace, samples=samples, model=model, size=size)
        # assert ppc['density_dist'].shape == (samples, size) + obs.distribution.shape

    @pytest.mark.parametrize("shape", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random_sampleable_failure(self, shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1, shape=shape)
            pm.DensityDist(
                "density_dist",
                normal_dist.logp,
                observed=np.random.randn(100, *shape),
                shape=shape,
                random=normal_dist.random,
                wrap_random_with_dist_shape=False,
            )
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        with pytest.raises(RuntimeError):
            pm.sample_posterior_predictive(trace, samples=samples, model=model, size=100)

        with pytest.raises((TypeError, RuntimeError)):
            pm.fast_sample_posterior_predictive(trace, samples=samples, model=model, size=100)

    @pytest.mark.parametrize("shape", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random_sampleable_hidden_error(self, shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1, shape=shape)
            obs = pm.DensityDist(
                "density_dist",
                normal_dist.logp,
                observed=np.random.randn(100, *shape),
                shape=shape,
                random=normal_dist.random,
                wrap_random_with_dist_shape=False,
                check_shape_in_random=False,
            )
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        ppc = pm.sample_posterior_predictive(trace, samples=samples, model=model)
        assert len(ppc["density_dist"]) == samples
        assert ((samples,) + obs.distribution.shape) != ppc["density_dist"].shape

        ppc = pm.fast_sample_posterior_predictive(trace, samples=samples, model=model)
        assert len(ppc["density_dist"]) == samples
        assert ((samples,) + obs.distribution.shape) != ppc["density_dist"].shape

    def test_density_dist_with_random_sampleable_handcrafted_success(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1)
            rvs = pm.Normal.dist(mu, 1, shape=100).random
            obs = pm.DensityDist(
                "density_dist",
                normal_dist.logp,
                observed=np.random.randn(100),
                random=rvs,
                wrap_random_with_dist_shape=False,
            )
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        size = 100
        ppc = pm.sample_posterior_predictive(trace, samples=samples, model=model, size=size)
        assert ppc["density_dist"].shape == (samples, size) + obs.distribution.shape

    @pytest.mark.xfail
    def test_density_dist_with_random_sampleable_handcrafted_success_fast(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1)
            rvs = pm.Normal.dist(mu, 1, shape=100).random
            obs = pm.DensityDist(
                "density_dist",
                normal_dist.logp,
                observed=np.random.randn(100),
                random=rvs,
                wrap_random_with_dist_shape=False,
            )
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        size = 100

        ppc = pm.fast_sample_posterior_predictive(trace, samples=samples, model=model, size=size)
        assert ppc["density_dist"].shape == (samples, size) + obs.distribution.shape

    def test_density_dist_without_random_not_sampleable(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            normal_dist = pm.Normal.dist(mu, 1)
            pm.DensityDist("density_dist", normal_dist.logp, observed=np.random.randn(100))
            trace = pm.sample(100, cores=1, return_inferencedata=False)

        samples = 500
        with pytest.raises(ValueError):
            pm.sample_posterior_predictive(trace, samples=samples, model=model, size=100)

        with pytest.raises((TypeError, ValueError)):
            pm.fast_sample_posterior_predictive(trace, samples=samples, model=model, size=100)


class TestNestedRandom(SeededTest):
    def build_model(self, distribution, shape, nested_rvs_info):
        with pm.Model() as model:
            nested_rvs = {}
            for rv_name, info in nested_rvs_info.items():
                try:
                    value, nested_shape = info
                    loc = 0.0
                except ValueError:
                    value, nested_shape, loc = info
                if value is None:
                    nested_rvs[rv_name] = pm.Uniform(
                        rv_name,
                        0 + loc,
                        1 + loc,
                        shape=nested_shape,
                    )
                else:
                    nested_rvs[rv_name] = value * np.ones(nested_shape)
            rv = distribution(
                "target",
                shape=shape,
                **nested_rvs,
            )
        return model, rv, nested_rvs

    def sample_prior(self, distribution, shape, nested_rvs_info, prior_samples):
        model, rv, nested_rvs = self.build_model(
            distribution,
            shape,
            nested_rvs_info,
        )
        with model:
            return pm.sample_prior_predictive(prior_samples)

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "alpha"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_NegativeBinomial(
        self,
        prior_samples,
        shape,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.NegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "psi", "mu", "alpha"],
        [
            [10, (3,), (0.5, tuple()), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, (3,)), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, tuple()), (None, (3,)), (None, tuple())],
            [10, (3,), (0.5, (3,)), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_ZeroInflatedNegativeBinomial(
        self,
        prior_samples,
        shape,
        psi,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.ZeroInflatedNegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(psi=psi, mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "nu", "sigma"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_Rice(
        self,
        prior_samples,
        shape,
        nu,
        sigma,
    ):
        prior = self.sample_prior(
            distribution=pm.Rice,
            shape=shape,
            nested_rvs_info=dict(nu=nu, sigma=sigma),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "sigma", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_TruncatedNormal(
        self,
        prior_samples,
        shape,
        mu,
        sigma,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.TruncatedNormal,
            shape=shape,
            nested_rvs_info=dict(mu=mu, sigma=sigma, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "c", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (-1.0, (3,)), (2, tuple())],
            [10, (3,), (None, tuple()), (-1.0, tuple()), (None, tuple(), 1)],
            [10, (3,), (None, (3,)), (-1.0, tuple()), (None, tuple(), 1)],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (-1.0, tuple()),
                (None, (3,), 1),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, tuple(), -1),
                (None, (3,), 1),
            ],
        ],
        ids=str,
    )
    def test_Triangular(
        self,
        prior_samples,
        shape,
        c,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.Triangular,
            shape=shape,
            nested_rvs_info=dict(c=c, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape


def generate_shapes(include_params=False):
    # fmt: off
    mudim_as_event = [
        [None, 1, 3, 10, (10, 3), 100],
        [(3,)],
        [(1,), (3,)],
        ["cov", "chol", "tau"]
    ]
    # fmt: on
    mudim_as_dist = [
        [None, 1, 3, 10, (10, 3), 100],
        [(10, 3)],
        [(1,), (3,), (1, 1), (1, 3), (10, 1), (10, 3)],
        ["cov", "chol", "tau"],
    ]
    if not include_params:
        del mudim_as_event[-1]
        del mudim_as_dist[-1]
    data = itertools.chain(itertools.product(*mudim_as_event), itertools.product(*mudim_as_dist))
    return data


class TestMvNormal(SeededTest):
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape", "param"],
        generate_shapes(include_params=True),
        ids=str,
    )
    def test_with_np_arrays(self, sample_shape, dist_shape, mu_shape, param):
        dist = pm.MvNormal.dist(mu=np.ones(mu_shape), **{param: np.eye(3)}, shape=dist_shape)
        output_shape = to_tuple(sample_shape) + dist_shape
        assert dist.random(size=sample_shape).shape == output_shape

    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_chol_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            mv = pm.MvNormal("mv", mu, chol=chol, shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape

    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_cov_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            mv = pm.MvNormal("mv", mu, cov=pm.math.dot(chol, chol.T), shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape

    def test_issue_3758(self):
        np.random.seed(42)
        ndim = 50
        with pm.Model() as model:
            a = pm.Normal("a", sigma=100, shape=ndim)
            b = pm.Normal("b", mu=a, sigma=1, shape=ndim)
            c = pm.MvNormal("c", mu=a, chol=np.linalg.cholesky(np.eye(ndim)), shape=ndim)
            d = pm.MvNormal("d", mu=a, cov=np.eye(ndim), shape=ndim)
            samples = pm.sample_prior_predictive(1000)

        for var in "abcd":
            assert not np.isnan(np.std(samples[var]))

        for var in "bcd":
            std = np.std(samples[var] - samples["a"])
            np.testing.assert_allclose(std, 1, rtol=1e-2)

    def test_issue_3829(self):
        with pm.Model() as model:
            x = pm.MvNormal("x", mu=np.zeros(5), cov=np.eye(5), shape=(2, 5))
            trace_pp = pm.sample_prior_predictive(50)

        assert np.shape(trace_pp["x"][0]) == (2, 5)

    def test_issue_3706(self):
        N = 10
        Sigma = np.eye(2)

        with pm.Model() as model:

            X = pm.MvNormal("X", mu=np.zeros(2), cov=Sigma, shape=(N, 2))
            betas = pm.Normal("betas", 0, 1, shape=2)
            y = pm.Deterministic("y", pm.math.dot(X, betas))

            prior_pred = pm.sample_prior_predictive(1)

        assert prior_pred["X"].shape == (1, N, 2)


def test_matrix_normal_random_with_random_variables():
    """
    This test checks for shape correctness when using MatrixNormal distribution
    with parameters as random variables.
    Originally reported - https://github.com/pymc-devs/pymc3/issues/3585
    """
    K = 3
    D = 15
    mu_0 = np.zeros((D, K))
    lambd = 1.0
    with pm.Model() as model:
        sd_dist = pm.HalfCauchy.dist(beta=2.5)
        packedL = pm.LKJCholeskyCov("packedL", eta=2, n=D, sd_dist=sd_dist)
        L = pm.expand_packed_triangular(D, packedL, lower=True)
        Sigma = pm.Deterministic("Sigma", L.dot(L.T))  # D x D covariance
        mu = pm.MatrixNormal(
            "mu", mu=mu_0, rowcov=(1 / lambd) * Sigma, colcov=np.eye(K), shape=(D, K)
        )
        prior = pm.sample_prior_predictive(2)

    assert prior["mu"].shape == (2, D, K)


@pytest.mark.parametrize("n", [2, 5])
def test_lkj_corr_with_prior_predictive(n):
    """
    This test checks for shape correctness when using pm.sample_prior_predictive
    on LKJCorr distribution.
    Originally reported - https://github.com/pymc-devs/pymc3/issues/4778
    """
    with pm.Model():
        lkj = pm.LKJCorr("lkj", eta=1, n=n)
        prior = pm.sample_prior_predictive(500)

    upper_triangular_shape = n * (n - 1) // 2
    assert prior["lkj"].shape == (500, upper_triangular_shape)


class TestMvGaussianRandomWalk(SeededTest):
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape", "param"],
        generate_shapes(include_params=True),
        ids=str,
    )
    def test_with_np_arrays(self, sample_shape, dist_shape, mu_shape, param):
        dist = pm.MvGaussianRandomWalk.dist(
            mu=np.ones(mu_shape), **{param: np.eye(3)}, shape=dist_shape
        )
        output_shape = to_tuple(sample_shape) + dist_shape
        assert dist.random(size=sample_shape).shape == output_shape

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_chol_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            mv = pm.MvGaussianRandomWalk("mv", mu, chol=chol, shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_cov_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            mv = pm.MvGaussianRandomWalk("mv", mu, cov=pm.math.dot(chol, chol.T), shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape
