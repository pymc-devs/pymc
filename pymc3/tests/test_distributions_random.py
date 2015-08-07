
from __future__ import division
import unittest

import itertools
from .checks import *
from .knownfailure import *

from pymc3.tests.test_distributions import (build_model,
    Domain, product, R, Rplus, Rplusbig, Unit, Nat, NatSmall, 
    I, Simplex)

from pymc3.distributions.continuous import *
from pymc3.distributions.discrete import *
from pymc3.distributions.distribution import draw_values#, get_sample_shape
from pymc3 import Model, Point

import numpy as np
import scipy.stats as st
import numpy.random as nr

def pymc3_random(dist, paramdomains, ref_rand=None, size=10000, alpha=0.05, fails=10):
    model = build_model(dist, Domain([0]), paramdomains)
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
            _, p = st.ks_2samp(s0, s1)
            f -= 1
        assert p > alpha, str(pt)

def pymc3_random_discrete(dist, paramdomains, ref_rand=None, size=100000, alpha=0.05, fails=20):
    model = build_model(dist, Domain([0]), paramdomains)
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
                                  point={'mu':5., 'sigma':2.})
        assert mu2 == 5. and tau2 == 0.25, "Draw values failed using point replacement"

        mu3, tau3 = draw_values([y2.distribution.mu, y2.distribution.tau])
        assert isinstance(mu3, np.ndarray) and isinstance(tau3, np.ndarray), \
            "Draw values did not return np.ndarray with random sampling"

def test_uniform_random():
    pymc3_random(Uniform, {'lower':-Rplus, 'upper':Rplus},
                 ref_rand=lambda size, lower=None, upper=None: st.uniform.rvs(size=size,loc=lower, scale=upper-lower))

def test_normal_random():
    pymc3_random(Normal, {'mu':R, 'sd':Rplus},
                 ref_rand=lambda size, mu=None, sd=None: st.norm.rvs(size=size,loc=mu, scale=sd))

def test_halfnormal_random():
    pymc3_random(HalfNormal, {'tau':Rplus},
                 ref_rand=lambda size, tau=None: st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

def test_wald_random():
    # Cannot do anything too exciting as scipy wald is a
    # location-scale model of the *standard* wald with mu=1 and lam=1
    pymc3_random(Wald, {'mu':Domain([1., 1., 1.]), 'lam':Domain([1., 1., 1.]), 'alpha':Rplus},
                 ref_rand=lambda size, mu=None, lam=None, alpha=None: st.wald.rvs(size=size, loc=alpha))

def test_beta_random():
    pymc3_random(
            Beta, {'alpha': Rplus, 'beta': Rplus},
            lambda size, alpha=None, beta=None: st.beta.rvs(a=alpha, b=beta, size=size)
            )
    pymc3_random(
            Beta, {'mu': Unit, 'sd': Rplus},
            lambda size, mu=None, sd=None: st.beta.rvs(a=mu*sd, b=(1-mu)*sd, size=size)
            )

def test_exponential_random():
    pymc3_random(
            Exponential, {'lam': Rplus},
            lambda size, lam=None: nr.exponential(scale=1./lam, size=size)
            )

def test_laplace_random():
    pymc3_random(
            Laplace, {'mu': R, 'b': Rplus},
            lambda size, mu=None, b=None: st.laplace.rvs(mu, b, size=size)
            )

def test_lognormal_random():
    pymc3_random(
            Lognormal, {'mu': R, 'tau': Rplusbig},
            lambda size, mu, tau: np.exp(mu + (tau ** -0.5) * st.norm.rvs(loc=0., scale=1., size=size))
            )

def test_t_random():
    pymc3_random(
            T, {'nu': Rplus, 'mu': R, 'lam': Rplus},
            lambda size, nu=None, mu=None, lam=None: st.t.rvs(nu, mu, lam**-.5, size=size)
            )

def test_cauchy_random():
    pymc3_random(
            Cauchy, {'alpha': R, 'beta': Rplusbig},
            lambda size, alpha, beta: st.cauchy.rvs(alpha, beta, size=size)
            )

def test_half_cauchy_random():
    pymc3_random(
            HalfCauchy, {'beta': Rplusbig},
            lambda size, beta=None: st.halfcauchy.rvs(scale=beta, size=size)
            )

def test_gamma_random():
    pymc3_random(
            Gamma, {'alpha': Rplusbig, 'beta': Rplusbig},
            lambda size, alpha=None, beta=None: st.gamma.rvs(alpha, scale=1.0/beta, size=size)
            )
    pymc3_random(
            Gamma, {'mu': Rplusbig, 'sd': Rplusbig},
            lambda size, mu=None, sd=None: st.gamma.rvs(mu**2 / sd**2, scale=1.0/(mu / sd**2), size=size)
            )

def test_inverse_gamma_random():
    pymc3_random(
            InverseGamma, {'alpha': Rplus, 'beta': Rplus},
            lambda size, alpha=None, beta=None: st.invgamma.rvs(a=alpha, scale=beta, size=size)
            )

def test_pareto_random():
    pymc3_random(
            Pareto, {'alpha': Rplusbig, 'm': Rplusbig},
            lambda size, alpha=None, m=None: st.pareto.rvs(alpha, scale=m, size=size)
            )

def test_ex_gaussian_random():
    pymc3_random(
            ExGaussian, {'mu':R, 'sigma':Rplus, 'nu':Rplus},
            lambda size, mu=None, sigma=None, nu=None: nr.normal(mu, sigma, size=size) + \
                nr.exponential(scale=nu, size=size)
            )

def test_bounded_random():
    # A bit crude...
    BoundedNormal = Bound(Normal, upper=0)
    pymc3_random(BoundedNormal, {'tau':Rplus},
                 ref_rand=lambda size, tau=None: -st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

def test_flat_random():
    with Model():
        f = Flat('f')
        try:
            f.random(1)
            assert False, 'Flat distribution returned samples'
        except ValueError:
            pass

def test_bounded_random():
    # A bit crude...
    BoundedNormal = Bound(Normal, upper=0)
    pymc3_random(BoundedNormal, {'tau':Rplus},
                 ref_rand=lambda size, tau=None: -st.halfnorm.rvs(size=size,loc=0, scale=tau ** -0.5))

def test_binomial_random():
    pymc3_random_discrete(Binomial, {'n':Nat, 'p':Unit},
                 ref_rand=lambda size, n=None, p=None:st.binom.rvs(n=n, p=p, size=size))

def test_betabin_random():
    pymc3_random_discrete(BetaBin, {'n':Nat, 'alpha': Rplus, 'beta': Rplus},
                 ref_rand=lambda size, n=None, alpha=None, beta=None: \
                    st.binom.rvs(n, st.beta.rvs(a=alpha, b=beta, size=size)))

def test_bernoulli_random():
    pymc3_random_discrete(Bernoulli, {'p':Unit}, ref_rand=lambda size, p=None: st.bernoulli.rvs(p, size=size))

def test_poisson_random():
    pymc3_random_discrete(Poisson, {'mu':Rplusbig},
                          size=500,# Test always fails with larger sample sizes.
                          ref_rand=lambda size, mu=None: st.poisson.rvs(mu, size=size))

def poisson_gamma_random(alpha, mu, size):
    g = st.gamma.rvs(alpha, scale=alpha / mu, size=size)
    g[g==0] = np.finfo(float).eps
    return st.poisson.rvs(g)

def test_negative_binomial_random():
# To do: fix this so test passes
#   pymc3_random_discrete(NegativeBinomial, {'mu':Rplusbig, 'alpha':Rplusbig},
#                          size=1000,
#                          ref_rand=lambda size, mu=None, alpha=None: poisson_gamma_random(alpha, mu, size))
    pass

def test_geometric_random():
    pymc3_random_discrete(Geometric, {'p':Unit},
                          size=500,# Test always fails with larger sample sizes.
                          ref_rand=lambda size, p=None: nr.geometric(p, size=size))

def test_discrete_uniform_random():
    pymc3_random_discrete(DiscreteUniform, {'lower':-NatSmall, 'upper':NatSmall}, \
                 ref_rand=lambda size, lower=None, upper=None: nr.random_integers(lower, upper, size=size))

def test_categorical_random():
    # Don't make simplex too big. You have been warned.
    for s in [2, 3, 4]:
        yield check_categorical_random, s

def check_categorical_random(s):
    pymc3_random_discrete(Categorical, {'p':Simplex(s)},
                          size=1,
                          ref_rand=lambda size, p=None: nr.multinomial(1, p, size=size))

def test_constant_dist_random():
    pymc3_random_discrete(ConstantDist, {'c':I},
                 ref_rand=lambda size, c=None: np.ones(size).astype(int) * c)
