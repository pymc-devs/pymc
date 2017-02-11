from __future__ import division

from nose import SkipTest
from nose_parameterized import parameterized
from nose.plugins.attrib import attr
import numpy as np
import scipy.stats as st
import numpy.random as nr

import pymc3 as pm
from .helpers import SeededTest
from .test_distributions import (build_model, Domain, product, R, Rplus, Rplusbig, Rplusdunif, Unit, Nat,
                                 NatSmall, I, Simplex, Vector, PdMatrix)


def pymc3_random(dist, paramdomains, ref_rand, valuedomain=Domain([0]),
                 size=10000, alpha=0.05, fails=10):
    model = build_model(dist, valuedomain, paramdomains)
    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        p = alpha
        # Allow KS test to fail (i.e., the samples be different)
        # a certain number of times. Crude, but necessary.
        f = fails
        while p <= alpha and f > 0:
            s0 = model.named_vars['value'].random(size=size, point=pt)
            s1 = ref_rand(size=size, **pt)
            _, p = st.ks_2samp(np.atleast_1d(s0).flatten(),
                               np.atleast_1d(s1).flatten())
            f -= 1
        assert p > alpha, str(pt)


def pymc3_random_discrete(dist, paramdomains,
                          valuedomain=Domain([0]), ref_rand=None,
                          size=100000, alpha=0.05, fails=20):
    model = build_model(dist, valuedomain, paramdomains)
    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        p = alpha
        # Allow Chisq test to fail (i.e., the samples be different)
        # a certain number of times.
        f = fails
        while p <= alpha and f > 0:
            o = model.named_vars['value'].random(size=size, point=pt)
            e = ref_rand(size=size, **pt)
            o = np.atleast_1d(o).flatten()
            e = np.atleast_1d(e).flatten()
            observed = dict(zip(*np.unique(o, return_counts=True)))
            expected = dict(zip(*np.unique(e, return_counts=True)))
            for e in expected.keys():
                expected[e] = (observed.get(e, 0), expected[e])
            k = np.array([v for v in expected.values()])
            if np.all(k[:, 0] == k[:, 1]):
                p = 1.
            else:
                _, p = st.chisquare(k[:, 0], k[:, 1])
            f -= 1
        assert p > alpha, str(pt)


class TestDrawValues(SeededTest):
    def test_draw_scalar_parameters(self):
        with pm.Model():
            y = pm.Normal('y1', mu=0., sd=1.)
            mu, tau = pm.distributions.draw_values([y.distribution.mu, y.distribution.tau])
        self.assertAlmostEqual(mu, 0.)
        self.assertAlmostEqual(tau, 1.)

    def test_draw_point_replacement(self):
        with pm.Model():
            mu = pm.Normal('mu', mu=0., tau=1e-3)
            sigma = pm.Gamma('sigma', alpha=1., beta=1., transform=None)
            y = pm.Normal('y', mu=mu, sd=sigma)
            mu2, tau2 = pm.distributions.draw_values([y.distribution.mu, y.distribution.tau],
                                                     point={'mu': 5., 'sigma': 2.})
        self.assertAlmostEqual(mu2, 5.)
        self.assertAlmostEqual(tau2, 1 / 2.**2)

    def test_random_sample_returns_nd_array(self):
        with pm.Model():
            mu = pm.Normal('mu', mu=0., tau=1e-3)
            sigma = pm.Gamma('sigma', alpha=1., beta=1., transform=None)
            y = pm.Normal('y', mu=mu, sd=sigma)
            mu, tau = pm.distributions.draw_values([y.distribution.mu, y.distribution.tau])
        self.assertIsInstance(mu, np.ndarray)
        self.assertIsInstance(tau, np.ndarray)


class BaseTestCases(object):
    class BaseTestCase(SeededTest):
        shape = 5

        def __init__(self, *args, **kwargs):
            super(BaseTestCases.BaseTestCase, self).__init__(*args, **kwargs)
            self.model = pm.Model()

        def get_random_variable(self, shape, with_vector_params=False, name=None):
            if with_vector_params:
                params = {key: value * np.ones(self.shape, dtype=np.dtype(type(value))) for
                          key, value in self.params.items()}
            else:
                params = self.params
            if name is None:
                name = self.distribution.__name__
            with self.model:
                if shape is None:
                    return self.distribution(name, transform=None, **params)
                else:
                    return self.distribution(name, shape=shape, transform=None, **params)

        @staticmethod
        def sample_random_variable(random_variable, size):
            try:
                return random_variable.random(size=size)
            except AttributeError:
                return random_variable.distribution.random(size=size)

        def test_scalar_parameter_shape(self):
            rv = self.get_random_variable(None)
            for size in (None, 5, (4, 5)):
                if size is None:
                    expected = 1,
                else:
                    expected = np.atleast_1d(size).tolist()
                actual = np.atleast_1d(self.sample_random_variable(rv, size)).shape
                self.assertSequenceEqual(expected, actual)

        def test_scalar_shape(self):
            shape = 10
            rv = self.get_random_variable(shape)
            for size in (None, 5, (4, 5)):
                if size is None:
                    expected = []
                else:
                    expected = np.atleast_1d(size).tolist()
                expected.append(shape)
                actual = np.atleast_1d(self.sample_random_variable(rv, size)).shape
                self.assertSequenceEqual(expected, actual)

        def test_parameters_1d_shape(self):
            rv = self.get_random_variable(self.shape, with_vector_params=True)
            for size in (None, 5, (4, 5)):
                if size is None:
                    expected = []
                else:
                    expected = np.atleast_1d(size).tolist()
                expected.append(self.shape)
                actual = self.sample_random_variable(rv, size).shape
                self.assertSequenceEqual(expected, actual)

        def test_broadcast_shape(self):
            broadcast_shape = (2 * self.shape, self.shape)
            rv = self.get_random_variable(broadcast_shape, with_vector_params=True)
            for size in (None, 5, (4, 5)):
                if size is None:
                    expected = []
                else:
                    expected = np.atleast_1d(size).tolist()
                expected.extend(broadcast_shape)
                actual = np.atleast_1d(self.sample_random_variable(rv, size)).shape
                self.assertSequenceEqual(expected, actual)

        def test_different_shapes_and_sample_sizes(self):
            shapes = [(), (1,), (1, 1), (1, 2), (10, 10, 1), (10, 10, 2)]
            prefix = self.distribution.__name__
            expected = []
            actual = []
            for shape in shapes:
                rv = self.get_random_variable(shape, name='%s_%s' % (prefix, shape))
                for size in (None, 1, 5, (4, 5)):
                    if size is None:
                        s = []
                    else:
                        try:
                            s = list(size)
                        except TypeError:
                            s = [size]
                    s.extend(shape)
                    e = tuple(s)
                    a = self.sample_random_variable(rv, size).shape
                    expected.append(e)
                    actual.append(a)
            self.assertSequenceEqual(expected, actual)


class TestNormal(BaseTestCases.BaseTestCase):
    distribution = pm.Normal
    params = {'mu': 0., 'tau': 1.}


class TestSkewNormal(BaseTestCases.BaseTestCase):
    distribution = pm.SkewNormal
    params = {'mu': 0., 'sd': 1., 'alpha': 5.}


class TestHalfNormal(BaseTestCases.BaseTestCase):
    distribution = pm.HalfNormal
    params = {'tau': 1.}


class TestUniform(BaseTestCases.BaseTestCase):
    distribution = pm.Uniform
    params = {'lower': 0., 'upper': 1.}

class TestTriangular(BaseTestCases.BaseTestCase):
    distribution = pm.Triangular
    params = {'c': 0.5,'lower': 0., 'upper': 1.}


class TestWald(BaseTestCases.BaseTestCase):
    distribution = pm.Wald
    params = {'mu': 1., 'lam': 1., 'alpha': 0.}


class TestBeta(BaseTestCases.BaseTestCase):
    distribution = pm.Beta
    params = {'alpha': 1., 'beta': 1.}


class TestExponential(BaseTestCases.BaseTestCase):
    distribution = pm.Exponential
    params = {'lam': 1.}


class TestLaplace(BaseTestCases.BaseTestCase):
    distribution = pm.Laplace
    params = {'mu': 1., 'b': 1.}


class TestLognormal(BaseTestCases.BaseTestCase):
    distribution = pm.Lognormal
    params = {'mu': 1., 'tau': 1.}


class TestStudentT(BaseTestCases.BaseTestCase):
    distribution = pm.StudentT
    params = {'nu': 5., 'mu': 0., 'lam': 1.}


class TestPareto(BaseTestCases.BaseTestCase):
    distribution = pm.Pareto
    params = {'alpha': 0.5, 'm': 1.}


class TestCauchy(BaseTestCases.BaseTestCase):
    distribution = pm.Cauchy
    params = {'alpha': 1., 'beta': 1.}


class TestHalfCauchy(BaseTestCases.BaseTestCase):
    distribution = pm.HalfCauchy
    params = {'beta': 1.}


class TestGamma(BaseTestCases.BaseTestCase):
    distribution = pm.Gamma
    params = {'alpha': 1., 'beta': 1.}


class TestInverseGamma(BaseTestCases.BaseTestCase):
    distribution = pm.InverseGamma
    params = {'alpha': 0.5, 'beta': 0.5}


class TestChiSquared(BaseTestCases.BaseTestCase):
    distribution = pm.ChiSquared
    params = {'nu': 2.}


class TestWeibull(BaseTestCases.BaseTestCase):
    distribution = pm.Weibull
    params = {'alpha': 1., 'beta': 1.}


class TestExGaussian(BaseTestCases.BaseTestCase):
    distribution = pm.ExGaussian
    params = {'mu': 0., 'sigma': 1., 'nu': 1.}


class TestVonMises(BaseTestCases.BaseTestCase):
    distribution = pm.VonMises
    params = {'mu': 0., 'kappa': 1.}


class TestBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.Binomial
    params = {'n': 5, 'p': 0.5}


class TestBetaBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.BetaBinomial
    params = {'n': 5, 'alpha': 1., 'beta': 1.}


class TestBernoulli(BaseTestCases.BaseTestCase):
    distribution = pm.Bernoulli
    params = {'p': 0.5}


class TestDiscreteWeibull(BaseTestCases.BaseTestCase):
    distribution = pm.DiscreteWeibull
    params = {'q': 0.25, 'beta': 2.}


class TestPoisson(BaseTestCases.BaseTestCase):
    distribution = pm.Poisson
    params = {'mu': 1.}


class TestNegativeBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.NegativeBinomial
    params = {'mu': 1., 'alpha': 1.}


class TestConstant(BaseTestCases.BaseTestCase):
    distribution = pm.Constant
    params = {'c': 3}


class TestZeroInflatedPoisson(BaseTestCases.BaseTestCase):
    distribution = pm.ZeroInflatedPoisson
    params = {'theta': 1., 'psi': 0.3}


class TestZeroInflatedNegativeBinomial(BaseTestCases.BaseTestCase):
    distribution = pm.ZeroInflatedNegativeBinomial
    params = {'mu': 1., 'alpha': 1., 'psi': 0.3}


class TestDiscreteUniform(BaseTestCases.BaseTestCase):
    distribution = pm.DiscreteUniform
    params = {'lower': 0., 'upper': 10.}


class TestGeometric(BaseTestCases.BaseTestCase):
    distribution = pm.Geometric
    params = {'p': 0.5}


class TestCategorical(BaseTestCases.BaseTestCase):
    distribution = pm.Categorical
    params = {'p': np.ones(BaseTestCases.BaseTestCase.shape)}

    def get_random_variable(self, shape, with_vector_params=False, **kwargs):  # don't transform categories
        return super(TestCategorical, self).get_random_variable(shape, with_vector_params=False, **kwargs)


@attr('scalar_parameter_samples')
class ScalarParameterSamples(SeededTest):
    def test_bounded(self):
        # A bit crude...
        BoundedNormal = pm.Bound(pm.Normal, upper=0)

        def ref_rand(size, tau):
            return -st.halfnorm.rvs(size=size, loc=0, scale=tau ** -0.5)
        pymc3_random(BoundedNormal, {'tau': Rplus}, ref_rand=ref_rand)

    def test_uniform(self):
        def ref_rand(size, lower, upper):
            return st.uniform.rvs(size=size, loc=lower, scale=upper - lower)

        pymc3_random(pm.Uniform, {'lower': -Rplus, 'upper': Rplus}, ref_rand=ref_rand)

    def test_normal(self):
        def ref_rand(size, mu, sd):
            return st.norm.rvs(size=size, loc=mu, scale=sd)
        pymc3_random(pm.Normal, {'mu': R, 'sd': Rplus}, ref_rand=ref_rand)

    def test_skew_normal(self):
        def ref_rand(size, alpha, mu, sd):
            return st.skewnorm.rvs(size=size, a=alpha, loc=mu, scale=sd)
        pymc3_random(pm.SkewNormal, {'mu': R, 'sd': Rplus, 'alpha': R}, ref_rand=ref_rand)

    def test_half_normal(self):
        def ref_rand(size, tau):
            return st.halfnorm.rvs(size=size, loc=0, scale=tau ** -0.5)
        pymc3_random(pm.HalfNormal, {'tau': Rplus}, ref_rand=ref_rand)

    def test_wald(self):
        # Cannot do anything too exciting as scipy wald is a
        # location-scale model of the *standard* wald with mu=1 and lam=1
        def ref_rand(size, mu, lam, alpha):
            return st.wald.rvs(size=size, loc=alpha)
        pymc3_random(pm.Wald,
                     {'mu': Domain([1., 1., 1.]), 'lam': Domain(
                         [1., 1., 1.]), 'alpha': Rplus},
                     ref_rand=ref_rand)

    def test_beta(self):
        def ref_rand(size, alpha, beta):
            return st.beta.rvs(a=alpha, b=beta, size=size)
        pymc3_random(pm.Beta, {'alpha': Rplus, 'beta': Rplus}, ref_rand=ref_rand)

    def test_exponential(self):
        def ref_rand(size, lam):
            return nr.exponential(scale=1. / lam, size=size)
        pymc3_random(pm.Exponential, {'lam': Rplus}, ref_rand=ref_rand)

    def test_laplace(self):
        def ref_rand(size, mu, b):
            return st.laplace.rvs(mu, b, size=size)
        pymc3_random(pm.Laplace, {'mu': R, 'b': Rplus}, ref_rand=ref_rand)

    def test_lognormal(self):
        def ref_rand(size, mu, tau):
            return np.exp(mu + (tau ** -0.5) * st.norm.rvs(loc=0., scale=1., size=size))
        pymc3_random(pm.Lognormal, {'mu': R, 'tau': Rplusbig}, ref_rand=ref_rand)

    def test_student_t(self):
        def ref_rand(size, nu, mu, lam):
            return st.t.rvs(nu, mu, lam**-.5, size=size)
        pymc3_random(pm.StudentT, {'nu': Rplus, 'mu': R, 'lam': Rplus}, ref_rand=ref_rand)

    def test_cauchy(self):
        def ref_rand(size, alpha, beta):
            return st.cauchy.rvs(alpha, beta, size=size)
        pymc3_random(pm.Cauchy, {'alpha': R, 'beta': Rplusbig}, ref_rand=ref_rand)

    def test_half_cauchy(self):
        def ref_rand(size, beta):
            return st.halfcauchy.rvs(scale=beta, size=size)
        pymc3_random(pm.HalfCauchy, {'beta': Rplusbig}, ref_rand=ref_rand)

    def test_gamma_alpha_beta(self):
        def ref_rand(size, alpha, beta):
            return st.gamma.rvs(alpha, scale=1. / beta, size=size)
        pymc3_random(pm.Gamma, {'alpha': Rplusbig, 'beta': Rplusbig}, ref_rand=ref_rand)

    def test_gamma_mu_sd(self):
        def ref_rand(size, mu, sd):
            return st.gamma.rvs(mu**2 / sd**2, scale=sd ** 2 / mu, size=size)
        pymc3_random(pm.Gamma, {'mu': Rplusbig, 'sd': Rplusbig}, ref_rand=ref_rand)

    def test_inverse_gamma(self):
        def ref_rand(size, alpha, beta):
            return st.invgamma.rvs(a=alpha, scale=beta, size=size)
        pymc3_random(pm.InverseGamma, {'alpha': Rplus, 'beta': Rplus}, ref_rand=ref_rand)

    def test_pareto(self):
        def ref_rand(size, alpha, m):
            return st.pareto.rvs(alpha, scale=m, size=size)
        pymc3_random(pm.Pareto, {'alpha': Rplusbig, 'm': Rplusbig}, ref_rand=ref_rand)

    def test_ex_gaussian(self):
        def ref_rand(size, mu, sigma, nu):
            return nr.normal(mu, sigma, size=size) + nr.exponential(scale=nu, size=size)
        pymc3_random(pm.ExGaussian, {'mu': R, 'sigma': Rplus, 'nu': Rplus}, ref_rand=ref_rand)

    def test_vonmises(self):
        def ref_rand(size, mu, kappa):
            return st.vonmises.rvs(size=size, loc=mu, kappa=kappa)
        pymc3_random(pm.VonMises, {'mu': R, 'kappa': Rplus}, ref_rand=ref_rand)

    def test_flat(self):
        with pm.Model():
            f = pm.Flat('f')
            with self.assertRaises(ValueError):
                f.random(1)

    def test_binomial(self):
        pymc3_random_discrete(pm.Binomial, {'n': Nat, 'p': Unit}, ref_rand=st.binom.rvs)

    def test_beta_binomial(self):
        pymc3_random_discrete(pm.BetaBinomial, {'n': Nat, 'alpha': Rplus, 'beta': Rplus},
                              ref_rand=self._beta_bin)

    def _beta_bin(self, n, alpha, beta, size=None):
        return st.binom.rvs(n, st.beta.rvs(a=alpha, b=beta, size=size))

    def test_bernoulli(self):
        pymc3_random_discrete(pm.Bernoulli, {'p': Unit},
                              ref_rand=lambda size, p=None: st.bernoulli.rvs(p, size=size))

    def test_poisson(self):
        pymc3_random_discrete(pm.Poisson, {'mu': Rplusbig}, size=500, ref_rand=st.poisson.rvs)

    def test_negative_binomial(self):
        def ref_rand(size, alpha, mu):
            return st.nbinom.rvs(alpha, alpha / (mu + alpha), size=size)
        pymc3_random_discrete(pm.NegativeBinomial, {'mu': Rplusbig, 'alpha': Rplusbig},
                              size=100, fails=50, ref_rand=ref_rand)

    def test_geometric(self):
        pymc3_random_discrete(pm.Geometric, {'p': Unit}, size=500, fails=50, ref_rand=nr.geometric)

    def test_discrete_uniform(self):
        def ref_rand(size, lower, upper):
            return st.randint.rvs(lower, upper, size=size)
        pymc3_random_discrete(pm.DiscreteUniform, {'lower': -NatSmall, 'upper': NatSmall},
                              ref_rand=ref_rand)

    def test_discrete_weibull(self):
        def ref_rand(size, q, beta):
            u = np.random.uniform(size=size)

            return np.ceil(np.power(np.log(1 - u) / np.log(q), 1. / beta)) - 1

        pymc3_random_discrete(pm.DiscreteWeibull, {'q': Unit, 'beta': Rplusdunif},
                              ref_rand=ref_rand)

    @parameterized.expand([(2,), (3,), (4,)])
    def test_categorical_random(self, s):
        def ref_rand(size, p):
            return nr.choice(np.arange(p.shape[0]), p=p, size=size)
        pymc3_random_discrete(pm.Categorical, {'p': Simplex(s)}, ref_rand=ref_rand)

    def test_constant_dist(self):
        def ref_rand(size, c):
            return c * np.ones(size, dtype=int)
        pymc3_random_discrete(pm.Constant, {'c': I}, ref_rand=ref_rand)

    def test_mv_normal(self):
        def ref_rand(size, mu, cov):
            return st.multivariate_normal.rvs(mean=mu, cov=cov, size=size)
        for n in [2, 3]:
            pymc3_random(pm.MvNormal, {'mu': Vector(R, n), 'cov': PdMatrix(n)},
                         size=100, valuedomain=Vector(R, n), ref_rand=ref_rand)

    def test_mv_t(self):
        def ref_rand(size, nu, Sigma, mu):
            normal = st.multivariate_normal.rvs(cov=Sigma, size=size).T
            chi2 = st.chi2.rvs(df=nu, size=size)
            return mu + np.sqrt(nu) * (normal / chi2).T
        for n in [2, 3]:
            pymc3_random(pm.MvStudentT,
                         {'nu': Domain([5, 10, 25, 50]), 'Sigma': PdMatrix(
                             n), 'mu': Vector(R, n)},
                         size=100, valuedomain=Vector(R, n), ref_rand=ref_rand)

    def test_dirichlet(self):
        def ref_rand(size, a):
            return st.dirichlet.rvs(a, size=size)
        for n in [2, 3]:
            pymc3_random(pm.Dirichlet, {'a': Vector(Rplus, n)},
                         valuedomain=Simplex(n), size=100, ref_rand=ref_rand)

    def test_multinomial(self):
        def ref_rand(size, p, n):
            return nr.multinomial(pvals=p, n=n, size=size)
        for n in [2, 3]:
            pymc3_random_discrete(pm.Multinomial, {'p': Simplex(n), 'n': Nat},
                                  valuedomain=Vector(Nat, n), size=100, ref_rand=ref_rand)

    def test_wishart(self):
        # Wishart non current recommended for use:
        # https://github.com/pymc-devs/pymc3/issues/538
        raise SkipTest('Wishart random sampling not implemented.\n'
                       'See https://github.com/pymc-devs/pymc3/issues/538')
        # for n in [2, 3]:
        #     pymc3_random_discrete(Wisvaluedomainhart,
        #                           {'n': Domain([2, 3, 4, 2000]) , 'V': PdMatrix(n) },
        #                           valuedomain=PdMatrix(n),
        #                           ref_rand=lambda n=None, V=None, size=None: \
        #                           st.wishart(V, df=n, size=size))

    def test_lkj(self):
        # TODO: generate random numbers.
        raise SkipTest('LJK random sampling not implemented yet.')
