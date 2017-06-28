from __future__ import division

import itertools

from .helpers import SeededTest, select_by_precision
from ..vartypes import continuous_types
from ..model import Model, Point, Potential
from ..blocking import DictToVarBijection, DictToArrayBijection, ArrayOrdering
from ..distributions import (DensityDist, Categorical, Multinomial, VonMises, Dirichlet,
                             MvStudentT, MvNormal, ZeroInflatedPoisson, GaussianRandomWalk,
                             ZeroInflatedNegativeBinomial, Constant, Poisson, Bernoulli, Beta,
                             BetaBinomial, HalfStudentT, StudentT, Weibull, Pareto, NormalMixture,
                             InverseGamma, Gamma, Cauchy, HalfCauchy, Lognormal, Laplace,
                             NegativeBinomial, Geometric, Exponential, ExGaussian, Normal,
                             Flat, LKJCorr, Wald, ChiSquared, HalfNormal, DiscreteUniform,
                             Bound, Uniform, Triangular, Binomial, SkewNormal, DiscreteWeibull, Gumbel,
                             Interpolated, ZeroInflatedBinomial)
from ..distributions import continuous
from pymc3.theanof import floatX
from numpy import array, inf, log, exp
from numpy.testing import assert_almost_equal
import numpy.random as nr
import numpy as np
import pytest

from scipy import integrate
import scipy.stats.distributions as sp
import scipy.stats
import theano
import theano.tensor as tt


def get_lkj_cases():
    """
    Log probabilities calculated using the formulas in:
    http://www.sciencedirect.com/science/article/pii/S0047259X09000876
    """
    tri = np.array([0.7, 0.0, -0.7])
    return [
        (tri, 1, 3, 1.5963125911388549),
        (tri, 3, 3, -7.7963493376312742),
        (tri, 0, 3, -np.inf),
        (np.array([1.1, 0.0, -0.7]), 1, 3, -np.inf),
        (np.array([0.7, 0.0, -1.1]), 1, 3, -np.inf)
    ]


LKJ_CASES = get_lkj_cases()


class Domain(object):
    def __init__(self, vals, dtype=None, edges=None, shape=None):
        avals = array(vals, dtype=dtype)
        if dtype is None and not str(avals.dtype).startswith('int'):
            avals = avals.astype(theano.config.floatX)
        vals = [array(v, dtype=avals.dtype) for v in vals]

        if edges is None:
            edges = array(vals[0]), array(vals[-1])
            vals = vals[1:-1]
        if shape is None:
            shape = avals[0].shape

        self.vals = vals
        self.shape = shape

        self.lower, self.upper = edges
        self.dtype = avals.dtype

    def __add__(self, other):
        return Domain(
            [v + other for v in self.vals],
            self.dtype,
            (self.lower + other, self.upper + other),
            self.shape)

    def __mul__(self, other):
        return Domain(
            [v * other for v in self.vals],
            self.dtype,
            (self.lower * other, self.upper * other),
            self.shape)

    def __neg__(self):
        return Domain(
            [-v for v in self.vals],
            self.dtype,
            (-self.lower, -self.upper),
            self.shape)


def product(domains, n_samples=-1):
    """Get an iterator over a product of domains.

    Args:
        domains: a dictionary of (name, object) pairs, where the objects
                 must be "domain-like", as in, have a `.vals` property
        n_samples: int, maximum samples to return.  -1 to return whole product

    Returns:
        list of the cartesian product of the domains
    """
    try:
        names, domains = zip(*domains.items())
    except ValueError:  # domains.items() is empty
        return []
    all_vals = [zip(names, val) for val in itertools.product(*[d.vals for d in domains])]
    if n_samples > 0 and len(all_vals) > n_samples:
            return (all_vals[j] for j in nr.choice(len(all_vals), n_samples, replace=False))
    return all_vals


R = Domain([-inf, -2.1, -1, -.01, .0, .01, 1, 2.1, inf])
Rplus = Domain([0, .01, .1, .9, .99, 1, 1.5, 2, 100, inf])
Rplusbig = Domain([0, .5, .9, .99, 1, 1.5, 2, 20, inf])
Rminusbig = Domain([-inf, -2, -1.5, -1, -.99, -.9, -.5, -0.01, 0])
Unit = Domain([0, .001, .1, .5, .75, .99, 1])

Circ = Domain([-np.pi, -2.1, -1, -.01, .0, .01, 1, 2.1, np.pi])

Runif = Domain([-1, -.4, 0, .4, 1])
Rdunif = Domain([-10, 0, 10.])
Rplusunif = Domain([0, .5, inf])
Rplusdunif = Domain([2, 10, 100], 'int64')

I = Domain([-1000, -3, -2, -1, 0, 1, 2, 3, 1000], 'int64')

NatSmall = Domain([0, 3, 4, 5, 1000], 'int64')
Nat = Domain([0, 1, 2, 3, 2000], 'int64')
NatBig = Domain([0, 1, 2, 3, 5000, 50000], 'int64')
PosNat = Domain([1, 2, 3, 2000], 'int64')

Bool = Domain([0, 0, 1, 1], 'int64')


def build_model(distfam, valuedomain, vardomains, extra_args=None):
    if extra_args is None:
        extra_args = {}
    with Model() as m:
        vals = {}
        for v, dom in vardomains.items():
            vals[v] = Flat(v, dtype=dom.dtype, shape=dom.shape,
                           testval=dom.vals[0])
        vals.update(extra_args)
        distfam('value', shape=valuedomain.shape, transform=None, **vals)
    return m


def integrate_nd(f, domain, shape, dtype):
    if shape == () or shape == (1,):
        if dtype in continuous_types:
            return integrate.quad(f, domain.lower, domain.upper, epsabs=1e-8)[0]
        else:
            return sum(f(j) for j in range(domain.lower, domain.upper + 1))
    elif shape == (2,):
        def f2(a, b):
            return f([a, b])

        return integrate.dblquad(f2, domain.lower[0], domain.upper[0],
                                 lambda _: domain.lower[1],
                                 lambda _: domain.upper[1])[0]
    elif shape == (3,):
        def f3(a, b, c):
            return f([a, b, c])

        return integrate.tplquad(f3, domain.lower[0], domain.upper[0],
                                 lambda _: domain.lower[1],
                                 lambda _: domain.upper[1],
                                 lambda _, __: domain.lower[2],
                                 lambda _, __: domain.upper[2])[0]
    else:
        raise ValueError("Dont know how to integrate shape: " + str(shape))


def multinomial_logpdf(value, n, p):
    if value.sum() == n and (0 <= value).all() and (value <= n).all():
        logpdf = scipy.special.gammaln(n + 1)
        logpdf -= scipy.special.gammaln(value + 1).sum()
        logpdf += logpow(p, value).sum()
        return logpdf
    else:
        return -inf


def beta_mu_sd(value, mu, sd):
    kappa = mu * (1 - mu) / sd**2 - 1
    if kappa > 0:
        return sp.beta.logpdf(value, mu * kappa, (1 - mu) * kappa)
    else:
        return -inf


class ProductDomain(object):
    def __init__(self, domains):
        self.vals = list(itertools.product(*[d.vals for d in domains]))
        self.shape = (len(domains),) + domains[0].shape
        self.lower = [d.lower for d in domains]
        self.upper = [d.upper for d in domains]
        self.dtype = domains[0].dtype


def Vector(D, n):
    return ProductDomain([D] * n)


def RealMatrix(n, m):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.random.randn(n, m))
    return Domain(vals, edges=(None, None))


def simplex_values(n):
    if n == 1:
        yield array([1.0])
    else:
        for v in Unit.vals:
            for vals in simplex_values(n - 1):
                yield np.concatenate([[v], (1 - v) * vals])


def scipy_exponweib_sucks(value, alpha, beta):
    """
    This function is required because SciPy's implementation of
    the Weibull PDF fails for some valid combinations of parameters, while the
    log-PDF fails for others.
    """
    pdf = np.log(sp.exponweib.pdf(value, 1, alpha, scale=beta))
    if np.isinf(pdf):
        return sp.exponweib.logpdf(value, 1, alpha, scale=beta)
    return floatX(pdf)


def normal_logpdf_tau(value, mu, tau):
    return normal_logpdf_cov(value, mu, np.linalg.inv(tau)).sum()


def normal_logpdf_cov(value, mu, cov):
    return scipy.stats.multivariate_normal.logpdf(value, mu, cov).sum()


def normal_logpdf_chol(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol, chol.T)).sum()


def normal_logpdf_chol_upper(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol.T, chol)).sum()


def betafn(a):
    return floatX(scipy.special.gammaln(a).sum(-1) - scipy.special.gammaln(a.sum(-1)))


def logpow(v, p):
    return np.choose(v == 0, [p * np.log(v), 0])


def discrete_weibull_logpmf(value, q, beta):
    return floatX(np.log(np.power(q, np.power(value, beta))
                  - np.power(q, np.power(value + 1, beta))))


def dirichlet_logpdf(value, a):
    return floatX((-betafn(a) + logpow(value, a - 1).sum(-1)).sum())


def categorical_logpdf(value, p):
    if value >= 0 and value <= len(p):
        return floatX(np.log(p[value]))
    else:
        return -inf


def mvt_logpdf(value, nu, Sigma, mu=0):
    d = len(Sigma)
    X = np.atleast_2d(value) - mu
    Q = X.dot(np.linalg.inv(Sigma)).dot(X.T).sum()
    log_det = np.log(np.linalg.det(Sigma))

    log_pdf = scipy.special.gammaln(0.5 * (nu + d))
    log_pdf -= 0.5 * (d * np.log(np.pi * nu) + log_det)
    log_pdf -= scipy.special.gammaln(nu / 2.)
    log_pdf -= 0.5 * (nu + d) * np.log(1 + Q / nu)
    return log_pdf


class Simplex(object):
    def __init__(self, n):
        self.vals = list(simplex_values(n))
        self.shape = (n,)
        self.dtype = Unit.dtype


class MultiSimplex(object):
    def __init__(self, n_dependent, n_independent):
        self.vals = []
        for simplex_value in itertools.product(simplex_values(n_dependent), repeat=n_independent):
            self.vals.append(np.vstack(simplex_value))
        self.shape = (n_independent, n_dependent)
        self.dtype = Unit.dtype


def PdMatrix(n):
    if n == 1:
        return PdMatrix1
    elif n == 2:
        return PdMatrix2
    elif n == 3:
        return PdMatrix3
    else:
        raise ValueError("n out of bounds")

PdMatrix1 = Domain([np.eye(1), [[.5]]], edges=(None, None))

PdMatrix2 = Domain([np.eye(2), [[.5, .05], [.05, 4.5]]], edges=(None, None))

PdMatrix3 = Domain(
    [np.eye(3), [[.5, .1, 0], [.1, 1, 0], [0, 0, 2.5]]], edges=(None, None))


PdMatrixChol1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixChol2 = Domain([np.eye(2), [[0.1, 0], [10, 1]]], edges=(None, None))
PdMatrixChol3 = Domain([np.eye(3), [[0.1, 0, 0], [10, 100, 0], [0, 1, 10]]],
                       edges=(None, None))


def PdMatrixChol(n):
    if n == 1:
        return PdMatrixChol1
    elif n == 2:
        return PdMatrixChol2
    elif n == 3:
        return PdMatrixChol3
    else:
        raise ValueError("n out of bounds")


PdMatrixCholUpper1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixCholUpper2 = Domain([np.eye(2), [[0.1, 10], [0, 1]]], edges=(None, None))
PdMatrixCholUpper3 = Domain([np.eye(3), [[0.1, 10, 0], [0, 100, 1], [0, 0, 10]]],
                            edges=(None, None))


def PdMatrixCholUpper(n):
    if n == 1:
        return PdMatrixCholUpper1
    elif n == 2:
        return PdMatrixCholUpper2
    elif n == 3:
        return PdMatrixCholUpper3
    else:
        raise ValueError("n out of bounds")


class TestMatchesScipy(SeededTest):
    def pymc3_matches_scipy(self, pymc3_dist, domain, paramdomains, scipy_dist, decimal=None, extra_args={}):
        model = build_model(pymc3_dist, domain, paramdomains, extra_args)
        value = model.named_vars['value']

        def logp(args):
            return scipy_dist(**args)
        self.check_logp(model, value, domain, paramdomains, logp, decimal=decimal)

    def check_logp(self, model, value, domain, paramdomains, logp_reference, decimal=None):
        domains = paramdomains.copy()
        domains['value'] = domain
        logp = model.fastlogp
        for pt in product(domains, n_samples=100):
            pt = Point(pt, model=model)
            if decimal is None:
                decimal = select_by_precision(float64=6, float32=3)
            assert_almost_equal(logp(pt), logp_reference(pt), decimal=decimal, err_msg=str(pt))

    def check_int_to_1(self, model, value, domain, paramdomains):
        pdf = model.fastfn(exp(model.logpt))
        for pt in product(paramdomains, n_samples=100):
            pt = Point(pt, value=value.tag.test_value, model=model)
            bij = DictToVarBijection(value, (), pt)
            pdfx = bij.mapf(pdf)
            area = integrate_nd(pdfx, domain, value.dshape, value.dtype)
            assert_almost_equal(area, 1, err_msg=str(pt))

    def check_dlogp(self, model, value, domain, paramdomains):
        try:
            from numdifftools import Gradient
        except ImportError:
            return
        if not model.cont_vars:
            return

        domains = paramdomains.copy()
        domains['value'] = domain
        bij = DictToArrayBijection(
            ArrayOrdering(model.cont_vars), model.test_point)
        dlogp = bij.mapf(model.fastdlogp(model.cont_vars))
        logp = bij.mapf(model.fastlogp)

        def wrapped_logp(x):
            try:
                return logp(x)
            except:
                return np.nan

        ndlogp = Gradient(wrapped_logp)
        for pt in product(domains, n_samples=100):
            pt = Point(pt, model=model)
            pt = bij.map(pt)
            decimals = select_by_precision(float64=6, float32=4)
            assert_almost_equal(dlogp(pt), ndlogp(pt), decimal=decimals, err_msg=str(pt))

    def checkd(self, distfam, valuedomain, vardomains, checks=None, extra_args={}):
        if checks is None:
            checks = (self.check_int_to_1, self.check_dlogp)
        m = build_model(distfam, valuedomain, vardomains, extra_args=extra_args)
        for check in checks:
            check(m, m.named_vars['value'], valuedomain, vardomains)

    def test_uniform(self):
        self.pymc3_matches_scipy(
            Uniform, Runif, {'lower': -Rplusunif, 'upper': Rplusunif},
            lambda value, lower, upper: sp.uniform.logpdf(value, lower, upper - lower))

    def test_triangular(self):
        self.pymc3_matches_scipy(
            Triangular, Runif, {'lower': -Rplusunif, 'c': Runif, 'upper': Rplusunif},
            lambda value, c, lower, upper: sp.triang.logpdf(value, c-lower, lower, upper-lower))


    def test_bound_normal(self):
        PositiveNormal = Bound(Normal, lower=0.)
        self.pymc3_matches_scipy(PositiveNormal, Rplus, {'mu': Rplus, 'sd': Rplus},
                                 lambda value, mu, sd: sp.norm.logpdf(value, mu, sd),
                                 decimal=select_by_precision(float64=6, float32=-1))
        with Model(): x = PositiveNormal('x', mu=0, sd=1, transform=None)
        assert np.isinf(x.logp({'x':-1}))

    def test_discrete_unif(self):
        self.pymc3_matches_scipy(
            DiscreteUniform, Rdunif, {'lower': -Rplusdunif, 'upper': Rplusdunif},
            lambda value, lower, upper: sp.randint.logpmf(value, lower, upper + 1))

    def test_flat(self):
        self.pymc3_matches_scipy(Flat, Runif, {}, lambda value: 0)

    def test_normal(self):
        self.pymc3_matches_scipy(Normal, R, {'mu': R, 'sd': Rplus},
                                 lambda value, mu, sd: sp.norm.logpdf(value, mu, sd),
                                 decimal=select_by_precision(float64=6, float32=1)
                                 )

    def test_half_normal(self):
        self.pymc3_matches_scipy(HalfNormal, Rplus, {'sd': Rplus},
                                 lambda value, sd: sp.halfnorm.logpdf(value, scale=sd),
                                 decimal=select_by_precision(float64=6, float32=-1)
                                 )

    def test_chi_squared(self):
        self.pymc3_matches_scipy(ChiSquared, Rplus, {'nu': Rplusdunif},
                                 lambda value, nu: sp.chi2.logpdf(value, df=nu))

    def test_wald_scipy(self):
        self.pymc3_matches_scipy(Wald, Rplus, {'mu': Rplus},
                                 lambda value, mu: sp.invgauss.logpdf(value, mu),
                                 decimal=select_by_precision(float64=6, float32=1)
                                 )

    @pytest.mark.parametrize('value,mu,lam,phi,alpha,logp', [
        (.5, .001, .5, None, 0., -124500.7257914),
        (1., .5, .001, None, 0., -4.3733162),
        (2., 1., None, None, 0., -2.2086593),
        (5., 2., 2.5, None, 0., -3.4374500),
        (7.5, 5., None, 1., 0., -3.2199074),
        (15., 10., None, .75, 0., -4.0360623),
        (50., 15., None, .66666, 0., -6.1801249),
        (.5, .001, 0.5, None, 0., -124500.7257914),
        (1., .5, .001, None, .5, -3.3330954),
        (2., 1., None, None, 1., -0.9189385),
        (5., 2., 2.5, None, 2., -2.2128783),
        (7.5, 5., None, 1., 2.5, -2.5283764),
        (15., 10., None, .75, 5., -3.3653647),
        (50., 15., None, .666666, 10., -5.6481874)
    ])
    def test_wald(self, value, mu, lam, phi, alpha, logp):
        # Log probabilities calculated using the dIG function from the R package gamlss.
        # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or
        # http://www.gamlss.org/.
        with Model() as model:
            Wald('wald', mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
        pt = {'wald': value}
        decimals = select_by_precision(float64=6, float32=1)
        assert_almost_equal(model.fastlogp(pt), logp, decimal=decimals, err_msg=str(pt))

    def test_beta(self):
        self.pymc3_matches_scipy(Beta, Unit, {'alpha': Rplus, 'beta': Rplus},
                                 lambda value, alpha, beta: sp.beta.logpdf(value, alpha, beta))
        self.pymc3_matches_scipy(Beta, Unit, {'mu': Unit, 'sd': Rplus}, beta_mu_sd)

    def test_exponential(self):
        self.pymc3_matches_scipy(Exponential, Rplus, {'lam': Rplus},
                                 lambda value, lam: sp.expon.logpdf(value, 0, 1 / lam))

    def test_geometric(self):
        self.pymc3_matches_scipy(Geometric, Nat, {'p': Unit},
                                 lambda value, p: np.log(sp.geom.pmf(value, p)))

    def test_negative_binomial(self):
        def test_fun(value, mu, alpha):
            return sp.nbinom.logpmf(value, alpha, 1 - mu / (mu + alpha))
        self.pymc3_matches_scipy(NegativeBinomial, Nat, {
                            'mu': Rplus, 'alpha': Rplus}, test_fun)

    def test_laplace(self):
        self.pymc3_matches_scipy(Laplace, R, {'mu': R, 'b': Rplus},
                                 lambda value, mu, b: sp.laplace.logpdf(value, mu, b))

    def test_lognormal(self):
        self.pymc3_matches_scipy(
            Lognormal, Rplus, {'mu': R, 'tau': Rplusbig},
            lambda value, mu, tau: floatX(sp.lognorm.logpdf(value, tau**-.5, 0, np.exp(mu))))

    def test_t(self):
        self.pymc3_matches_scipy(StudentT, R, {'nu': Rplus, 'mu': R, 'lam': Rplus},
                                 lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam**-0.5))

    def test_cauchy(self):
        self.pymc3_matches_scipy(Cauchy, R, {'alpha': R, 'beta': Rplusbig},
                                 lambda value, alpha, beta: sp.cauchy.logpdf(value, alpha, beta))

    def test_half_cauchy(self):
        self.pymc3_matches_scipy(HalfCauchy, Rplus, {'beta': Rplusbig},
                                 lambda value, beta: sp.halfcauchy.logpdf(value, scale=beta))

    def test_gamma(self):
        self.pymc3_matches_scipy(
            Gamma, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
            lambda value, alpha, beta: sp.gamma.logpdf(value, alpha, scale=1.0 / beta))

        def test_fun(value, mu, sd):
            return sp.gamma.logpdf(value, mu**2 / sd**2, scale=1.0 / (mu / sd**2))
        self.pymc3_matches_scipy(
            Gamma, Rplus, {'mu': Rplusbig, 'sd': Rplusbig}, test_fun)

    def test_inverse_gamma(self):
        self.pymc3_matches_scipy(
            InverseGamma, Rplus, {'alpha': Rplus, 'beta': Rplus},
            lambda value, alpha, beta: sp.invgamma.logpdf(value, alpha, scale=beta))

    def test_pareto(self):
        self.pymc3_matches_scipy(Pareto, Rplus, {'alpha': Rplusbig, 'm': Rplusbig},
                                 lambda value, alpha, m: sp.pareto.logpdf(value, alpha, scale=m))

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
    def test_weibull(self):
        self.pymc3_matches_scipy(Weibull, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
                                 scipy_exponweib_sucks,
                                 )

    def test_student_tpos(self):
        # TODO: this actually shouldn't pass
        self.pymc3_matches_scipy(HalfStudentT, Rplus, {'nu': Rplus, 'mu': R, 'lam': Rplus},
                                 lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam**-.5))

    def test_skew_normal(self):
        self.pymc3_matches_scipy(SkewNormal, R, {'mu': R, 'sd': Rplusbig, 'alpha': R},
                                 lambda value, alpha, mu, sd: sp.skewnorm.logpdf(value, alpha, mu, sd))

    def test_binomial(self):
        self.pymc3_matches_scipy(Binomial, Nat, {'n': NatSmall, 'p': Unit},
                                 lambda value, n, p: sp.binom.logpmf(value, n, p))

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")  # Too lazy to propagate decimal parameter through the whole chain of deps
    def test_beta_binomial(self):
        self.checkd(BetaBinomial, Nat, {'alpha': Rplus, 'beta': Rplus, 'n': NatSmall})

    def test_bernoulli(self):
        self.pymc3_matches_scipy(Bernoulli, Bool, {'p': Unit},
                                 lambda value, p: sp.bernoulli.logpmf(value, p))

    def test_discrete_weibull(self):
        self.pymc3_matches_scipy(DiscreteWeibull, Nat,
                {'q': Unit, 'beta': Rplusdunif}, discrete_weibull_logpmf)

    def test_poisson(self):
        self.pymc3_matches_scipy(Poisson, Nat, {'mu': Rplus},
                                 lambda value, mu: sp.poisson.logpmf(value, mu))

    def test_bound_poisson(self):
        NonZeroPoisson = Bound(Poisson, lower=1.)
        self.pymc3_matches_scipy(NonZeroPoisson, PosNat, {'mu': Rplus},
                                lambda value, mu: sp.poisson.logpmf(value, mu))

        with Model(): x = NonZeroPoisson('x', mu=4)
        assert np.isinf(x.logp({'x':0}))

    def test_constantdist(self):
        self.pymc3_matches_scipy(Constant, I, {'c': I},
                                 lambda value, c: np.log(c == value))

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")  # Too lazy to propagate decimal parameter through the whole chain of deps
    def test_zeroinflatedpoisson(self):
        self.checkd(ZeroInflatedPoisson, Nat, {'theta': Rplus, 'psi': Unit})

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")  # Too lazy to propagate decimal parameter through the whole chain of deps
    def test_zeroinflatednegativebinomial(self):
        self.checkd(ZeroInflatedNegativeBinomial, Nat,
                    {'mu': Rplusbig, 'alpha': Rplusbig, 'psi': Unit})

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")  # Too lazy to propagate decimal parameter through the whole chain of deps
    def test_zeroinflatedbinomial(self):
        self.checkd(ZeroInflatedBinomial, Nat,
                    {'n': NatSmall, 'p': Unit, 'psi': Unit})

    @pytest.mark.parametrize('n', [1, 2, 3])
    def test_mvnormal(self, n):
        self.pymc3_matches_scipy(MvNormal, RealMatrix(5, n),
                                 {'mu': Vector(R, n), 'tau': PdMatrix(n)},
                                 normal_logpdf_tau)
        self.pymc3_matches_scipy(MvNormal, Vector(R, n),
                                 {'mu': Vector(R, n), 'tau': PdMatrix(n)},
                                 normal_logpdf_tau)
        self.pymc3_matches_scipy(MvNormal, RealMatrix(5, n),
                                 {'mu': Vector(R, n), 'cov': PdMatrix(n)},
                                 normal_logpdf_cov)
        self.pymc3_matches_scipy(MvNormal, Vector(R, n),
                                 {'mu': Vector(R, n), 'cov': PdMatrix(n)},
                                 normal_logpdf_cov)
        self.pymc3_matches_scipy(MvNormal, RealMatrix(5, n),
                                 {'mu': Vector(R, n), 'chol': PdMatrixChol(n)},
                                 normal_logpdf_chol,
                                 decimal=select_by_precision(float64=6, float32=-1))
        self.pymc3_matches_scipy(MvNormal, Vector(R, n),
                                 {'mu': Vector(R, n), 'chol': PdMatrixChol(n)},
                                 normal_logpdf_chol,
                                 decimal=select_by_precision(float64=6, float32=0))

        def MvNormalUpper(*args, **kwargs):
            return MvNormal(lower=False, *args, **kwargs)

        self.pymc3_matches_scipy(MvNormalUpper, Vector(R, n),
                                 {'mu': Vector(R, n), 'chol': PdMatrixCholUpper(n)},
                                 normal_logpdf_chol_upper,
                                 decimal=select_by_precision(float64=6, float32=0))

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
    def test_mvnormal_indef(self):
        cov_val = np.array([[1, 0.5], [0.5, -2]])
        cov = tt.matrix('cov')
        cov.tag.test_value = np.eye(2)
        mu = floatX(np.zeros(2))
        x = tt.vector('x')
        x.tag.test_value = np.zeros(2)
        logp = MvNormal.dist(mu=mu, cov=cov).logp(x)
        f_logp = theano.function([cov, x], logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = tt.grad(logp, cov)
        f_dlogp = theano.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        logp = MvNormal.dist(mu=mu, tau=cov).logp(x)
        f_logp = theano.function([cov, x], logp)
        assert f_logp(cov_val, np.ones(2)) == -np.inf
        dlogp = tt.grad(logp, cov)
        f_dlogp = theano.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

    def test_mvnormal_init_fail(self):
        with Model():
            with pytest.raises(ValueError):
                x = MvNormal('x', mu=np.zeros(3), shape=3)
            with pytest.raises(ValueError):
                x = MvNormal('x', mu=np.zeros(3), cov=np.eye(3), tau=np.eye(3), shape=3)

    @pytest.mark.parametrize('n', [1, 2])
    def test_mvt(self, n):
        self.pymc3_matches_scipy(MvStudentT, Vector(R, n),
                                 {'nu': Rplus, 'Sigma': PdMatrix(n), 'mu': Vector(R, n)},
                                 mvt_logpdf)

    @pytest.mark.parametrize('n', [2, 3])
    def test_wishart(self, n):
        # This check compares the autodiff gradient to the numdiff gradient.
        # However, due to the strict constraints of the wishart,
        # it is impossible to numerically determine the gradient as a small
        # pertubation breaks the symmetry. Thus disabling.
        #
        # self.checkd(Wishart, PdMatrix(n), {'n': Domain([2, 3, 4, 2000]), 'V': PdMatrix(n)},
        #             checks=[self.check_dlogp])
        pass

    @pytest.mark.parametrize('x,eta,n,lp', LKJ_CASES)
    def test_lkj(self, x, eta, n, lp):
        with Model() as model:
            LKJCorr('lkj', eta=eta, n=n, transform=None)

        pt = {'lkj': x}
        decimals = select_by_precision(float64=6, float32=4)
        assert_almost_equal(model.fastlogp(pt), lp, decimal=decimals, err_msg=str(pt))

    @pytest.mark.parametrize('n', [2, 3])
    def test_dirichlet(self, n):
        self.pymc3_matches_scipy(Dirichlet, Simplex(
            n), {'a': Vector(Rplus, n)}, dirichlet_logpdf)

    def test_dirichlet_2D(self):
        self.pymc3_matches_scipy(Dirichlet, MultiSimplex(2, 2),
                                 {'a': Vector(Vector(Rplus, 2), 2)}, dirichlet_logpdf)

    @pytest.mark.parametrize('n', [2, 3])
    def test_multinomial(self, n):
        self.pymc3_matches_scipy(Multinomial, Vector(Nat, n), {'p': Simplex(n), 'n': Nat},
                                 multinomial_logpdf)

    def test_multinomial_vec(self):
        vals = np.array([[2,4,4], [3,3,4]])
        p = np.array([0.2, 0.3, 0.5])
        n = 10

        with Model() as model_single:
            Multinomial('m', n=n, p=p, shape=len(p))

        with Model() as model_many:
            Multinomial('m', n=n, p=p, shape=vals.shape)

        assert_almost_equal(sum([model_single.fastlogp({'m': val}) for val in vals]),
                            model_many.fastlogp({'m': vals}),
                            decimal=4)

    def test_multinomial_vec_1d_n(self):
        vals = np.array([[2,4,4], [4,3,4]])
        p = np.array([0.2, 0.3, 0.5])
        ns = np.array([10, 11])

        with Model() as model:
            Multinomial('m', n=ns, p=p, shape=vals.shape)

        assert_almost_equal(sum([multinomial_logpdf(val, n, p) for val, n in zip(vals, ns)]),
                            model.fastlogp({'m': vals}),
                            decimal=4)

    def test_multinomial_vec_1d_n_2d_p(self):
        vals = np.array([[2,4,4], [4,3,4]])
        ps = np.array([[0.2, 0.3, 0.5],
                       [0.9, 0.09, 0.01]])
        ns = np.array([10, 11])

        with Model() as model:
            Multinomial('m', n=ns, p=ps, shape=vals.shape)

        assert_almost_equal(sum([multinomial_logpdf(val, n, p) for val, n, p in zip(vals, ns, ps)]),
                            model.fastlogp({'m': vals}),
                            decimal=4)

    def test_multinomial_vec_2d_p(self):
        vals = np.array([[2,4,4], [3,3,4]])
        ps = np.array([[0.2, 0.3, 0.5],
                       [0.3, 0.3, 0.4]])
        n = 10

        with Model() as model:
            Multinomial('m', n=n, p=ps, shape=vals.shape)

        assert_almost_equal(sum([multinomial_logpdf(val, n, p) for val, p in zip(vals, ps)]),
                            model.fastlogp({'m': vals}),
                            decimal=4)

    def test_categorical_bounds(self):
        with Model():
            x = Categorical('x', p=np.array([0.2, 0.3, 0.5]))
            assert np.isinf(x.logp({'x': -1}))
            assert np.isinf(x.logp({'x': 3}))

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_categorical(self, n):
        self.pymc3_matches_scipy(Categorical, Domain(range(n), 'int64'), {'p': Simplex(n)},
                                 lambda value, p: categorical_logpdf(value, p))

    def test_densitydist(self):
        def logp(x):
            return -log(2 * .5) - abs(x - .5) / .5
        self.checkd(DensityDist, R, {}, extra_args={'logp': logp})

    def test_addpotential(self):
        with Model() as model:
            value = Normal('value', 1, 1)
            Potential('value_squared', -value ** 2)
            self.check_dlogp(model, value, R, {})

    def test_get_tau_sd(self):
        sd = np.array([2])
        assert_almost_equal(continuous.get_tau_sd(sd=sd), [1. / sd**2, sd])

    @pytest.mark.parametrize('value,mu,sigma,nu,logp', [
        (0.5, -50.000, 0.500, 0.500, -99.8068528),
        (1.0, -1.000, 0.001, 0.001, -1992.5922447),
        (2.0, 0.001, 1.000, 1.000, -1.6720416),
        (5.0, 0.500, 2.500, 2.500, -2.4543644),
        (7.5, 2.000, 5.000, 5.000, -2.8259429),
        (15.0, 5.000, 7.500, 7.500, -3.3093854),
        (50.0, 50.000, 10.000, 10.000, -3.6436067),
        (1000.0, 500.000, 10.000, 20.000, -27.8707323)
    ])
    def test_ex_gaussian(self, value, mu, sigma, nu, logp):
        """Log probabilities calculated using the dexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        with Model() as model:
            ExGaussian('eg', mu=mu, sigma=sigma, nu=nu)
        pt = {'eg': value}
        assert_almost_equal(model.fastlogp(pt), logp, decimal=select_by_precision(float64=6, float32=2), err_msg=str(pt))

    def test_vonmises(self):
        self.pymc3_matches_scipy(
            VonMises, R, {'mu': Circ, 'kappa': Rplus},
            lambda value, mu, kappa: floatX(sp.vonmises.logpdf(value, kappa, loc=mu)))

    def test_gumbel(self):
        def gumbel(value, mu, beta):
            return floatX(sp.gumbel_r.logpdf(value, loc=mu, scale=beta))
        self.pymc3_matches_scipy(Gumbel, R, {'mu': R, 'beta': Rplusbig}, gumbel)

    def test_multidimensional_beta_construction(self):
        with Model():
            Beta('beta', alpha=1., beta=1., shape=(10, 20))

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_interpolated(self):
        for mu in R.vals:
            for sd in Rplus.vals:
                #pylint: disable=cell-var-from-loop
                xmin = mu - 5 * sd
                xmax = mu + 5 * sd

                class TestedInterpolated (Interpolated):
                    def __init__(self, **kwargs):
                        x_points = np.linspace(xmin, xmax, 100000)
                        pdf_points = sp.norm.pdf(x_points, loc=mu, scale=sd)
                        super(TestedInterpolated, self).__init__(
                            x_points=x_points,
                            pdf_points=pdf_points,
                            **kwargs
                        )

                def ref_pdf(value):
                    return np.where(
                        np.logical_and(value >= xmin, value <= xmax),
                        sp.norm.logpdf(value, mu, sd),
                        -np.inf * np.ones(value.shape)
                    )

                self.pymc3_matches_scipy(TestedInterpolated, R, {}, ref_pdf)


def test_repr_latex_():
    with Model():
        x0 = Binomial('Discrete', p=.5, n=10)
        x1 = Normal('Continuous', mu=0., sd=1.)
        x2 = GaussianRandomWalk('Timeseries', mu=x1, sd=1., shape=2)
        x3 = MvStudentT('Multivariate', nu=5, mu=x2, Sigma=np.diag(np.ones(2)), shape=2)
        x4 = NormalMixture('Mixture', w=np.array([.5, .5]), mu=x3, sd=x0)

    assert x0._repr_latex_() == '$Discrete \\sim \\text{Binomial}' \
                                '(\\mathit{n}=10, \\mathit{p}=0.5)$'
    assert x1._repr_latex_() == '$Continuous \\sim \\text{Normal}' \
                                '(\\mathit{mu}=0.0, \\mathit{sd}=1.0)$'
    assert x2._repr_latex_() == '$Timeseries \\sim \\text' \
                                '{GaussianRandomWalk}(\\mathit{mu}=Continuous, ' \
                                '\\mathit{sd}=1.0)$'
    assert x3._repr_latex_() == '$Multivariate \\sim \\text{MvStudentT}' \
                                '(\\mathit{nu}=5, \\mathit{mu}=Timeseries, ' \
                                '\\mathit{Sigma}=array)$'
    assert x4._repr_latex_() == '$Mixture \\sim \\text{NormalMixture}' \
                                '(\\mathit{w}=array, \\mathit{mu}=Multivariate, ' \
                                '\\mathit{sigma}=f(Discrete))$'


def test_discrete_trafo():
    with pytest.raises(ValueError) as err:
        Binomial.dist(n=5, p=0.5, transform='log')
    err.match('Transformations for discrete distributions')
    with Model():
        with pytest.raises(ValueError) as err:
            Binomial('a', n=5, p=0.5, transform='log')
        err.match('Transformations for discrete distributions')
