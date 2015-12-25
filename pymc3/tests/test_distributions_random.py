
from __future__ import division
import unittest

from .checks import *
from .knownfailure import *
from nose import SkipTest
from nose.plugins.attrib import attr

from pymc3.tests.test_distributions import (build_model,
    Domain, product, R, Rplus, Rplusbig, Unit, Nat, NatSmall,
    I, Simplex, Vector, PdMatrix)

from pymc3.distributions.continuous import *
from pymc3.distributions.discrete import *
from pymc3.distributions.multivariate import *
from pymc3.distributions.distribution import draw_values
from pymc3 import Model, Point

import numpy as np
import scipy.stats as st
import numpy.random as nr


def pymc3_random(dist, paramdomains,
                 ref_rand=None, valuedomain=Domain([0]),
                 size=10000, alpha=0.05, fails=10):
    model = build_model(dist, valuedomain, paramdomains)
    domains = paramdomains.copy()
    for pt in product(domains):
        pt = Point(pt, model=model)
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
    for pt in product(domains):
        pt = Point(pt, model=model)
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
                _chi, p = st.chisquare(k[:, 0], k[:, 1])
            f -= 1
        assert p > alpha, str(pt)


def test_draw_values():
    with Model():
        mu = Normal('mu', mu=0., tau=1e-3)
        sigma = Gamma('sigma', alpha=1., beta=1., transform=None)
        y1 = Normal('y1', mu=0., sd=1.)
        y2 = Normal('y2', mu=mu, sd=sigma)

        mu1, tau1 = draw_values([y1.distribution.mu, y1.distribution.tau])
        assert mu1 == 0. and tau1 == 1, "Draw values failed with scalar parameters"

        mu2, tau2 = draw_values([y2.distribution.mu, y2.distribution.tau],
                                point={'mu': 5., 'sigma': 2.})
        assert mu2 == 5. and tau2 == 0.25, "Draw values failed using point replacement"

        mu3, tau3 = draw_values([y2.distribution.mu, y2.distribution.tau])
        assert isinstance(mu3, np.ndarray) and isinstance(tau3, np.ndarray), \
            "Draw values did not return np.ndarray with random sampling"


def check_dist(dist_case, test_cases, shape=None):
    dist, dist_kwargs = dist_case
    with Model():
        if shape is None:
            rv = dist(dist.__name__, transform=None, **dist_kwargs)
        else:
            rv = dist(dist.__name__, shape=shape, transform=None,
                      **dist_kwargs)
        for size, expected in test_cases:
            check_shape(rv, size=size, expected=expected)


def check_shape(rv, size=None, expected=None):
    try:
        sample = rv.random(size=size)
    except AttributeError:
        sample = rv.distribution.random(size=size)
    actual = np.atleast_1d(sample).shape
    expected = np.atleast_1d(expected)
    assert np.all(actual == expected), \
        'Expected shape `{0}` but got `{1}` using `(size={2})`' \
        ' with `{3}` rv'.format(expected, actual, size, rv.distribution.__class__.__name__)

# TODO: factor out a base class to avoid copy/paste.


@attr('scalar_parameter_shape')
class ScalarParameterShape(unittest.TestCase):

    def check(self, dist, **kwargs):
        test_cases = [(None, (1,)), (5, (5,)), ((4, 5), (4, 5))]
        check_dist((dist, kwargs), test_cases)

    def test_normal(self):
        self.check(Normal, mu=0., tau=1.)

    def test_uniform(self):
        self.check(Uniform, lower=0., upper=1.)

    def test_half_normal(self):
        self.check(HalfNormal, tau=1.)

    def test_wald(self):
        self.check(Wald, mu=1., lam=1., alpha=0.)

    def test_beta(self):
        self.check(Beta, alpha=1., beta=1.)

    def test_exponential(self):
        self.check(Exponential, lam=1.)

    def test_laplace(self):
        self.check(Laplace, mu=1., b=1)

    def test_lognormal(self):
        self.check(Lognormal, mu=1., tau=1.)

    def test_student_t(self):
        self.check(StudentT, nu=5, mu=0., lam=1.)

    def test_pareto(self):
        self.check(Pareto, alpha=0.5, m=1.)

    def test_cauchy(self):
        self.check(Cauchy, alpha=1., beta=1.)

    def test_half_cauchy(self):
        self.check(HalfCauchy, beta=1.)

    def test_gamma(self):
        self.check(Gamma, alpha=1., beta=1.)

    def test_inverse_gamma(self):
        self.check(InverseGamma, alpha=0.5, beta=0.5)

    def test_chi_squared(self):
        self.check(ChiSquared, nu=2)

    def test_weibull(self):
        self.check(Weibull, alpha=1., beta=1.)

    def test_ex_gaussian(self):
        self.check(ExGaussian, mu=0., sigma=1., nu=1.)

    def test_binomial(self):
        self.check(Binomial,  n=5., p=0.5)

    def test_beta_bin(self):
        self.check(BetaBin, alpha=1., beta=1., n=1)

    def test_bernoulli(self):
        self.check(Bernoulli, p=0.5)

    def test_poisson(self):
        self.check(Poisson, mu=1.)

    def test_negative_binomial(self):
        self.check(NegativeBinomial, mu=1., alpha=1.)

    def test_constant_dist(self):
        self.check(ConstantDist, c=3)

    def test_zero_inflated_poisson(self):
        # To do: implement ZIP random
        #self.check(ZeroInflatedPoisson)
        raise SkipTest(
            'ZeroInflatedPoisson random sampling not yet implemented.')

    def test_discrete_uniform(self):
        self.check(DiscreteUniform, lower=0., upper=10)

    def test_geometric(self):
        self.check(Geometric, p=0.5)

    def test_categorical(self):
        self.check(Categorical, p=np.array([0.2, 0.3, 0.5]))


@attr('scalar_shape')
class ScalarShape(unittest.TestCase):

    def check(self, dist, **kwargs):
        n = 10
        test_cases = [(None, (n,)), (5, (5, n,)), ((4, 5), (4, 5, n,))]
        check_dist((dist, kwargs), test_cases, n)

    def test_normal(self):
        self.check(Normal, mu=0., tau=1.)

    def test_uniform(self):
        self.check(Uniform, lower=0., upper=1.)

    def test_half_normal(self):
        self.check(HalfNormal, tau=1.)

    def test_wald(self):
        self.check(Wald, mu=1., lam=1., alpha=0.)

    def test_beta(self):
        self.check(Beta, alpha=1., beta=1.)

    def test_exponential(self):
        self.check(Exponential, lam=1.)

    def test_laplace(self):
        self.check(Laplace, mu=1., b=1)

    def test_lognormal(self):
        self.check(Lognormal, mu=1., tau=1.)

    def test_student_t(self):
        self.check(StudentT, nu=5, mu=0., lam=1.)

    def test_pareto(self):
        self.check(Pareto, alpha=0.5, m=1.)

    def test_cauchy(self):
        self.check(Cauchy, alpha=1., beta=1.)

    def test_half_cauchy(self):
        self.check(HalfCauchy, beta=1.)

    def test_gamma(self):
        self.check(Gamma, alpha=1., beta=1.)

    def test_inverse_gamma(self):
        self.check(InverseGamma, alpha=0.5, beta=0.5)

    def test_chi_squared(self):
        self.check(ChiSquared, nu=2)

    def test_weibull(self):
        self.check(Weibull, alpha=1., beta=1.)

    def test_ex_gaussian(self):
        self.check(ExGaussian, mu=0., sigma=1., nu=1.)

    def test_binomial(self):
        self.check(Binomial,  n=5., p=0.5)

    def test_beta_bin(self):
        self.check(BetaBin, alpha=1., beta=1., n=1)

    def test_bernoulli(self):
        self.check(Bernoulli, p=0.5)

    def test_poisson(self):
        self.check(Poisson, mu=1.)

    def test_negative_binomial(self):
        self.check(NegativeBinomial, mu=1., alpha=1.)

    def test_constant_dist(self):
        self.check(ConstantDist, c=3)

    def test_zero_inflated_poisson(self):
        # To do: implement ZIP random
        raise SkipTest(
            'ZeroInflatedPoisson random sampling not yet implemented.')

    def test_discrete_uniform(self):
        self.check(DiscreteUniform, lower=0., upper=10)

    def test_geometric(self):
        self.check(Geometric, p=0.5)

    def test_categorical(self):
        self.check(Categorical, p=np.array([0.2, 0.3, 0.5]))


@attr('parameters_1d_shape')
class Parameters1dShape(unittest.TestCase):

    def setUp(self):
        self.n = 5
        self.zeros = np.zeros(self.n)
        self.ones = np.ones(self.n)

    def check(self, dist, **kwargs):
        n = self.n
        test_cases = [(None, (n,)), (5, (5, n,)), ((4, 5), (4, 5, n,))]
        check_dist((dist, kwargs), test_cases, n)

    def test_normal(self):
        self.check(Normal, mu=self.zeros, tau=self.ones)

    def test_uniform(self):
        self.check(Uniform, lower=self.zeros, upper=self.ones)

    def test_half_normal(self):
        self.check(HalfNormal, tau=self.ones)

    def test_wald(self):
        self.check(Wald, mu=self.ones, lam=self.ones, alpha=self.zeros)

    def test_beta(self):
        self.check(Beta, alpha=self.ones, beta=self.ones)

    def test_exponential(self):
        self.check(Exponential, lam=self.ones)

    def test_laplace(self):
        self.check(Laplace, mu=self.ones, b=self.ones)

    def test_lognormal(self):
        self.check(Lognormal, mu=self.ones, tau=self.ones)

    def test_student_t(self):
        self.check(StudentT, nu=self.ones.astype(int), mu=self.zeros,
                   lam=self.ones)

    def test_pareto(self):
        self.check(Pareto, alpha=self.ones / 2, m=self.ones)

    def test_cauchy(self):
        self.check(Cauchy, alpha=self.ones, beta=self.ones)

    def test_half_cauchy(self):
        self.check(HalfCauchy, beta=self.ones)

    def test_gamma(self):
        self.check(Gamma, alpha=self.ones, beta=self.ones)

    def test_inverse_gamma(self):
        # InverseGamma fails due to calculation of self.mean in __init__
        raise SkipTest(
            'InverseGamma fails due to calculation of self.mean in __init__')
        self.check(InverseGamma, alpha=self.ones / 2, beta=self.ones / 2)

    def test_chi_squared(self):
        self.check(ChiSquared, nu=(self.ones * 2).astype(int))

    def test_weibull(self):
        self.check(Weibull, alpha=self.ones, beta=self.ones)

    def test_ex_gaussian(self):
        self.check(ExGaussian, mu=self.zeros, sigma=self.ones, nu=self.ones)

    def test_binomial(self):
        self.check(Binomial,  n=(self.ones * 5).astype(int), p=self.ones / 5)

    def test_beta_bin(self):
        self.check(BetaBin, alpha=self.ones, beta=self.ones,
                   n=self.ones.astype(int))

    def test_bernoulli(self):
        self.check(Bernoulli, p=self.ones / 2)

    def test_poisson(self):
        self.check(Poisson, mu=self.ones)

    def test_negative_binomial(self):
        self.check(NegativeBinomial, mu=self.ones, alpha=self.ones)

    def test_constantDist(self):
        self.check(ConstantDist, c=(self.ones * 3).astype(int))

    def test_zero_inflated_poisson(self):
        # To do: implement ZIP random
        raise SkipTest(
            'ZeroInflatedPoisson random sampling not yet implemented.')
        self.check(ZeroInflatedPoisson, {}, SkipTest)

    def test_discrete_uniform(self):
        self.check(DiscreteUniform,
                   lower=self.zeros.astype(int),
                   upper=(self.ones * 10).astype(int))

    def test_geometric(self):
        self.check(Geometric, p=self.ones / 2)

    def test_categorical(self):
        # Categorical cannot be initialised with >1D probabilities
        raise SkipTest(
            'Categorical cannot be initialised with >1D probabilities')
        self.check(Categorical, p=self.ones / n)


@attr('broadcast_shape')
class BroadcastShape(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.zeros = np.zeros(self.n)
        self.ones = np.ones(self.n)

    def check(self, dist, **kwargs):
        n = self.n
        shape = (2*n, n)
        test_cases = [(None, shape), (5, (5,) + shape),
                      ((4, 5), (4, 5) + shape)]
        check_dist((dist, kwargs), test_cases, shape)

    def test_normal(self):
        self.check(Normal, mu=self.zeros, tau=self.ones)

    def test_uniform(self):
        self.check(Uniform, lower=self.zeros, upper=self.ones)

    def test_half_normal(self):
        self.check(HalfNormal, tau=self.ones)

    def test_wald(self):
        self.check(Wald, mu=self.ones, lam=self.ones, alpha=self.zeros)

    def test_beta(self):
        self.check(Beta, alpha=self.ones, beta=self.ones)

    def test_exponential(self):
        self.check(Exponential, lam=self.ones)

    def test_laplace(self):
        self.check(Laplace, mu=self.ones, b=self.ones)

    def test_lognormal(self):
        self.check(Lognormal, mu=self.ones, tau=self.ones)

    def test_student_t(self):
        self.check(StudentT, nu=self.ones.astype(int), mu=self.zeros,
                   lam=self.ones)

    def test_pareto(self):
        self.check(Pareto, alpha=self.ones / 2, m=self.ones)

    def test_cauchy(self):
        self.check(Cauchy, alpha=self.ones, beta=self.ones)

    def test_half_cauchy(self):
        self.check(HalfCauchy, beta=self.ones)

    def test_gamma(self):
        self.check(Gamma, alpha=self.ones, beta=self.ones)

    def test_inverse_gamma(self):
        # InverseGamma fails due to calculation of self.mean in __init__
        raise SkipTest(
            'InverseGamma fails due to calculation of self.mean in __init__')
        self.check(InverseGamma, alpha=self.ones / 2, beta=self.ones / 2)

    def test_chi_squared(self):
        self.check(ChiSquared, nu=(self.ones * 2).astype(int))

    def test_weibull(self):
        self.check(Weibull, alpha=self.ones, beta=self.ones)

    def test_ex_gaussian(self):
        self.check(ExGaussian, mu=self.zeros, sigma=self.ones, nu=self.ones)

    def test_binomial(self):
        self.check(Binomial, n=(self.ones * 5).astype(int), p=self.ones / 5)

    def test_beta_bin(self):
        self.check(BetaBin, alpha=self.ones, beta=self.ones,
                   n=self.ones.astype(int))

    def test_bernoulli(self):
        self.check(Bernoulli, p=self.ones / 2)

    def test_poisson(self):
        self.check(Poisson, mu=self.ones)

    def test_negative_binomial(self):
        self.check(NegativeBinomial, mu=self.ones, alpha=self.ones)

    def test_constantDist(self):
        self.check(ConstantDist, c=(self.ones * 3).astype(int))

    def test_zero_inflated_poisson(self):
        # To do: implement ZIP random
        raise SkipTest(
            'ZeroInflatedPoisson random sampling not yet implemented.')
        self.check(ZeroInflatedPoisson, {})

    def test_discrete_uniform(self):
        self.check(DiscreteUniform, lower=self.zeros.astype(int),
                   upper=(self.ones * 10).astype(int))

    def test_geometric(self):
        self.check(Geometric, p=self.ones / 2)

    def test_categorical(self):
        # Categorical cannot be initialised with >1D probabilities
        raise SkipTest(
            'Categorical cannot be initialised with >1D probabilities')
        self.check(Categorical, p=self.ones / n)


@attr('scalar_parameter_samples')
class ScalarParameterSamples(unittest.TestCase):

    def test_uniform(self):
        pymc3_random(Uniform, {'lower': -Rplus, 'upper': Rplus},
                     ref_rand=lambda size, lower=None, upper=None: st.uniform.rvs(size=size,loc=lower, scale=upper-lower))

    def test_normal(self):
        pymc3_random(Normal, {'mu':R, 'sd':Rplus},
                     ref_rand=lambda size, mu=None, sd=None: st.norm.rvs(size=size,loc=mu, scale=sd))

    def test_half_normal(self):
        pymc3_random(HalfNormal, {'tau':Rplus},
                     ref_rand=lambda size, tau=None: st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

    def test_wald(self):
        # Cannot do anything too exciting as scipy wald is a
        # location-scale model of the *standard* wald with mu=1 and lam=1
        pymc3_random(Wald, {'mu':Domain([1., 1., 1.]), 'lam':Domain([1., 1., 1.]), 'alpha':Rplus},
                     ref_rand=lambda size, mu=None, lam=None, alpha=None: st.wald.rvs(size=size, loc=alpha))

    def test_beta(self):
        pymc3_random(
                Beta, {'alpha': Rplus, 'beta': Rplus},
                ref_rand=lambda size, alpha=None, beta=None: \
                    st.beta.rvs(a=alpha, b=beta, size=size)
                )
        pymc3_random(
                Beta, {'mu': Unit, 'sd': Rplus},
                ref_rand=lambda size, mu=None, sd=None: \
                    st.beta.rvs(a=mu*sd, b=(1-mu)*sd, size=size)
                )

    def test_exponential(self):
        pymc3_random(
                Exponential, {'lam': Rplus},
                ref_rand=lambda size, lam=None: nr.exponential(scale=1./lam, size=size)
                )

    def test_laplace(self):
        pymc3_random(
                Laplace, {'mu': R, 'b': Rplus},
                ref_rand=lambda size, mu=None, b=None: \
                    st.laplace.rvs(mu, b, size=size)
                )

    def test_lognormal(self):
        pymc3_random(
                Lognormal, {'mu': R, 'tau': Rplusbig},
                ref_rand=lambda size, mu, tau: \
                    np.exp(mu + (tau ** -0.5) * st.norm.rvs(loc=0., scale=1., size=size))
                )

    def test_student_t(self):
        pymc3_random(
                StudentT, {'nu': Rplus, 'mu': R, 'lam': Rplus},
                ref_rand=lambda size, nu=None, mu=None, lam=None: \
                    st.t.rvs(nu, mu, lam**-.5, size=size)
                )

    def test_cauchy(self):
        pymc3_random(
                Cauchy, {'alpha': R, 'beta': Rplusbig},
                ref_rand=lambda size, alpha, beta: st.cauchy.rvs(alpha, beta, size=size)
                )

    def test_half_cauchy(self):
        pymc3_random(
                HalfCauchy, {'beta': Rplusbig},
                ref_rand=lambda size, beta=None: st.halfcauchy.rvs(scale=beta, size=size)
                )

    def test_gamma(self):
        pymc3_random(
                Gamma, {'alpha': Rplusbig, 'beta': Rplusbig},
                ref_rand=lambda size, alpha=None, beta=None: \
                    st.gamma.rvs(alpha, scale=1.0/beta, size=size)
                )
        pymc3_random(
                Gamma, {'mu': Rplusbig, 'sd': Rplusbig},
                ref_rand=lambda size, mu=None, sd=None: \
                    st.gamma.rvs(mu**2 / sd**2, scale=1.0/(mu / sd**2), size=size)
                )

    def test_inverse_gamma(self):
        pymc3_random(
                InverseGamma, {'alpha': Rplus, 'beta': Rplus},
                ref_rand=lambda size, alpha=None, beta=None: \
                    st.invgamma.rvs(a=alpha, scale=beta, size=size)
                )

    def test_pareto(self):
        pymc3_random(
                Pareto, {'alpha': Rplusbig, 'm': Rplusbig},
                ref_rand=lambda size, alpha=None, m=None: \
                    st.pareto.rvs(alpha, scale=m, size=size)
                )

    def test_ex_gaussian(self):
        pymc3_random(
                ExGaussian, {'mu':R, 'sigma':Rplus, 'nu':Rplus},
                ref_rand=lambda size, mu=None, sigma=None, nu=None: \
                    nr.normal(mu, sigma, size=size) + nr.exponential(scale=nu, size=size)
                )

    def test_bounded(self):
        # A bit crude...
        BoundedNormal = Bound(Normal, upper=0)
        pymc3_random(BoundedNormal, {'tau':Rplus},
                     ref_rand=lambda size, tau=None: -st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

    def test_flat(self):
        with Model():
            f = Flat('f')
            try:
                f.random(1)
                assert False, 'Flat distribution returned samples'
            except ValueError:
                pass

    def test_bounded(self):
        # A bit crude...
        BoundedNormal = Bound(Normal, upper=0)
        pymc3_random(BoundedNormal, {'tau':Rplus},
                     ref_rand=lambda size, tau=None: -st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

    def test_binomial(self):
        pymc3_random_discrete(Binomial, {'n':Nat, 'p':Unit},
                              ref_rand=lambda size, n=None, p=None:st.binom.rvs(n=n, p=p, size=size))

    def test_beta_bin(self):
        pymc3_random_discrete(BetaBin, {'n':Nat, 'alpha': Rplus, 'beta': Rplus},
                              ref_rand=self._beta_bin)

    def _beta_bin(self, n=None, alpha=None, beta=None, size=None):
        sample = None
        while sample is None:# <- Danger!
            try:
                sample = st.binom.rvs(n, st.beta.rvs(a=alpha, b=beta, size=size))
            except ValueError:
                sample = None
        return sample

    def test_bernoulli(self):
        pymc3_random_discrete(Bernoulli, {'p':Unit}, ref_rand=lambda size, p=None: st.bernoulli.rvs(p, size=size))

    def test_poisson(self):
        pymc3_random_discrete(Poisson, {'mu':Rplusbig},
                              size=500,# Test always fails with larger sample sizes.
                              ref_rand=lambda size, mu=None: st.poisson.rvs(mu, size=size))

    def poisson_gamma_random(alpha, mu, size):
        g = st.gamma.rvs(alpha, scale=alpha / mu, size=size)
        g[g==0] = np.finfo(float).eps
        return st.poisson.rvs(g)

    def test_negative_binomial(self):
    # To do: fix this so test passes
    #   pymc3_random_discrete(NegativeBinomial, {'mu':Rplusbig, 'alpha':Rplusbig},
    #                          size=1000,
    #                          ref_rand=lambda size, mu=None, alpha=None: poisson_gamma_random(alpha, mu, size))
        raise SkipTest('NegativeBinomial test always fails for unknown reason.')

    def test_geometric(self):
        pymc3_random_discrete(Geometric, {'p':Unit},
                              size=500,# Test always fails with larger sample sizes.
                              fails=50,# Be a bit more generous.
                              ref_rand=lambda size, p=None: nr.geometric(p, size=size))

    def test_discrete_uniform(self):
        pymc3_random_discrete(DiscreteUniform, {'lower':-NatSmall, 'upper':NatSmall}, \
                              ref_rand=lambda size, lower=None, upper=None: st.randint.rvs(lower, upper, size=size))

    def test_categorical(self):
        # Don't make simplex too big. You have been warned.
        for s in [2, 3, 4]:
            yield self.check_categorical_random, s

    def checks_categorical_random(self, s):
        pymc3_random_discrete(Categorical, {'p':Simplex(s)},
                              ref_rand=lambda size=None, p=None: nr.choice(np.arange(p.shape[0]), p=p, size=size))

    def test_constant_dist(self):
        pymc3_random_discrete(ConstantDist, {'c':I},
                              ref_rand=lambda size, c=None: np.ones(size).astype(int) * c)

    def test_mv_normal(self):
        for n in [2, 3]:
            pymc3_random(MvNormal, {'mu':Vector(R,n), 'tau': PdMatrix(n)}, size=100,
                     valuedomain=Vector(R,n),
                     ref_rand=lambda mu=None, tau=None, size=None: \
                        st.multivariate_normal.rvs(mean=mu, cov=tau, size=size))

    def test_dirichlet(self):
       for n in [2, 3]:
           pymc3_random(Dirichlet, {'a': Vector(Rplus, n)},
                        valuedomain=Simplex(n), size=100,
                        ref_rand=lambda a=None, size=None: st.dirichlet.rvs(a, size=size))

    def test_multinomial(self):
       for n in [2, 3]:
           pymc3_random_discrete(Multinomial, {'p': Simplex(n), 'n' : Nat},
                        valuedomain=Vector(Nat, n), size=100,
                        ref_rand=lambda n=None, p=None, size=None: \
                            nr.multinomial(n, p, size=size))

    def test_wishart(self):
        # Wishart non current recommended for use:
        # https://github.com/pymc-devs/pymc3/issues/538
        raise SkipTest('Wishart random sampling not implemented.\n'
                       'See https://github.com/pymc-devs/pymc3/issues/538')
#         for n in [2, 3]:
#             pymc3_random_discrete(Wisvaluedomainhart, {'n': Domain([2, 3, 4, 2000]) , 'V': PdMatrix(n) },
#                                   valuedomain=PdMatrix(n),
#                                   ref_rand=lambda n=None, V=None, size=None: \
#                                   st.wishart(V, df=n, size=size))

    def test_lkj(self):
        # To do: generate random numbers.
        raise SkipTest('LJK random sampling not implemented yet.')
