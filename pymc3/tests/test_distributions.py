from __future__ import division
import unittest

import itertools
from .checks import *
from pymc3 import *
from numpy import array, inf
import numpy
from numpy.linalg import inv

from scipy import integrate
import scipy.stats.distributions  as sp
import scipy.stats

from .knownfailure import *



class Domain(object):
    def __init__(self, vals, dtype=None, edges=None, shape=None):
        avals = array(vals)

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
        return Domain([v + other for v in self.vals], self.dtype, (self.lower + other, self.upper + other), self.shape)

    def __mul__(self, other):
        return Domain([v * other for v in self.vals], self.dtype, (self.lower * other, self.upper * other), self.shape)

    def __neg__(self):
        return Domain([-v for v in self.vals], self.dtype, (-self.lower, -self.upper), self.shape)


def product(domains):
    names = [name for (name, domain) in domains.items()]
    domains = [domain for (name, domain) in domains.items()]

    for val in itertools.product(*[d.vals for d in domains]):
        yield zip(names, val)

R = Domain([-inf, -2.1, -1, -.01, .0, .01, 1, 2.1, inf])
Rplus = Domain([0, .01, .1, .9, .99, 1, 1.5, 2, 100, inf])
Rplusbig = Domain([0, .5, .9, .99, 1, 1.5, 2, 20, inf])
Unit = Domain([0, .001, .1, .5, .75, .99, 1])

Runif = Domain([-1, -.4, 0, .4, 1])
Rdunif = Domain([-10, 0, 10.])
Rplusunif = Domain([0, .5, inf])
Rplusdunif = Domain([2, 10, 100], 'int64')

I = Domain([-1000, -3, -2, -1, 0, 1, 2, 3, 1000], 'int64')

NatSmall = Domain([0, 3, 4, 5, 1000], 'int64')
Nat = Domain([0, 1, 2, 3, 2000], 'int64')
NatBig = Domain([0, 1, 2, 3, 5000, 50000], 'int64')

Bool = Domain([0, 0, 1, 1], 'int64')





class ProductDomain(object):
    def __init__(self, domains):
        self.vals = list(itertools.product(*[d.vals for d in domains]))

        self.shape = (len(domains),) + domains[0].shape

        self.lower = [d.lower for d in domains]
        self.upper = [d.upper for d in domains]

        self.dtype = domains[0].dtype

def Vector(D, n):
    return ProductDomain([D] *n)

def simplex_values(n):
    if n == 1:
        yield array([1.0])
    else:
        for v in Unit.vals:
            for vals in simplex_values(n-1):
                yield np.concatenate([[v], (1-v) * vals])

class Simplex(object):
    def __init__(self, n):
        self.vals = list(simplex_values(n))

        self.shape = (n,)
        self.dtype = Unit.dtype
        return

class MultiSimplex(object):
    def __init__(self, n_dependent, n_independent):
        transposed_vals = list(itertools.product(list(simplex_values(n_dependent)), repeat=n_independent))
        self.vals = list(np.transpose(transposed_vals, (0, 2, 1)))

        self.shape = (n_dependent, n_independent)
        self.dtype = Unit.dtype
        return

def PdMatrix(n):
    if n == 1:
        return PdMatrix1
    elif n == 2:
        return PdMatrix2
    elif n == 3:
        return PdMatrix3
    else:
        raise ValueError("n out of bounds")

PdMatrix1 = Domain([
    np.eye(1),
    [[.5]]
    ],
    edges = (None,None))

PdMatrix2 = Domain([
    np.eye(2),
    [[.5, .05],
     [.05, 4.5]]
    ],
    edges = (None,None))

PdMatrix3 = Domain([
    np.eye(3),
    [[.5, .1,0],
     [.1, 1, 0],
     [0, 0, 2.5]]
    ],
    edges = (None,None))

def test_uniform():
    pymc3_matches_scipy(
            Uniform, Runif, {'lower': -Rplusunif, 'upper': Rplusunif},
            lambda value, lower, upper: sp.uniform.logpdf(value, lower, upper - lower)
            )


def test_discrete_unif():
    pymc3_matches_scipy(
            DiscreteUniform, Rdunif,
            {'lower': -Rplusdunif, 'upper': Rplusdunif},
            lambda value, lower, upper: sp.randint.logpmf(value, lower, upper+1)
            )

def test_flat():
    pymc3_matches_scipy(
            Flat, Runif, {},
            lambda value: 0
            )


def test_normal():
    pymc3_matches_scipy(
            Normal, R, {'mu': R, 'sd': Rplus},
            lambda value, mu, sd: sp.norm.logpdf(value, mu, sd)
            )


def test_half_normal():
    pymc3_matches_scipy(
            HalfNormal, Rplus, {'sd': Rplus},
            lambda value, sd: sp.halfnorm.logpdf(value, scale=sd)
            )

def test_chi_squared():
    pymc3_matches_scipy(
            ChiSquared, Rplus, {'nu': Rplusdunif},
            lambda value, nu: sp.chi2.logpdf(value, df=nu)
            )



def test_wald_scipy():
    pymc3_matches_scipy(
            Wald, Rplus, {'mu': Rplus},
            lambda value, mu: sp.invgauss.logpdf(value, mu)
            )

def test_wald():
    # Log probabilities calculated using the dIG function from the R package gamlss.
    # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/.
    test_cases = [#
        (.5, .001, .5, None, 0., -124500.7257914),
        (1., .5, .001, None, 0., -4.3733162),
        (2., 1., None, None, 0., -2.2086593),
        (5., 2., 2.5, None, 0., -3.4374500),
        (7.5, 5., None, 1., 0., -3.2199074),
        (15., 10., None, .75, 0., -4.0360623),
        (50., 15., None, .66666 , 0., -6.1801249),
        (.5, .001, 0.5, None, 0., -124500.7257914),
        (1., .5, .001, None, .5, -3.3330954),
        (2., 1., None, None, 1., -0.9189385),
        (5., 2., 2.5, None, 2., -2.2128783),
        (7.5, 5., None, 1., 2.5, -2.5283764),
        (15., 10., None, .75, 5., -3.3653647),
        (50., 15., None, .666666, 10., -5.6481874)
        ]
    for value, mu, lam, phi, alpha, logp in test_cases:
        yield check_wald, value, mu, lam, phi, alpha, logp

def check_wald(value, mu, lam, phi, alpha, logp):
    with Model() as model:
        wald = Wald('wald', mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
    pt = {'wald': value}
    assert_almost_equal(model.fastlogp(pt),
                    logp, decimal=6, err_msg=str(pt))


def test_beta():
    pymc3_matches_scipy(
            Beta, Unit, {'alpha': Rplus, 'beta': Rplus},
            lambda value, alpha, beta: sp.beta.logpdf(value, alpha, beta)
            )
    pymc3_matches_scipy(
            Beta, Unit, {'mu': Unit, 'sd': Rplus},
            lambda value, mu, sd: sp.beta.logpdf(value, mu * sd, (1 - mu) * sd)
            )


def test_exponential():
    pymc3_matches_scipy(
            Exponential, Rplus, {'lam': Rplus},
            lambda value, lam: sp.expon.logpdf(value, 0, 1/lam)
            )

def test_geometric():
    pymc3_matches_scipy(
            Geometric, Nat, {'p': Unit},
            lambda value, p: np.log(sp.geom.pmf(value, p))
            )


def test_negative_binomial():
    pymc3_matches_scipy(
            NegativeBinomial, Nat, {'mu': Rplus, 'alpha': Rplus},
            lambda value, mu, alpha: sp.nbinom.logpmf(value, alpha, 1 - mu/(mu + alpha))
            )


def test_laplace():
    pymc3_matches_scipy(
            Laplace, R, {'mu': R, 'b': Rplus},
            lambda value, mu, b: sp.laplace.logpdf(value, mu, b)
            )


def test_lognormal():
    pymc3_matches_scipy(
            Lognormal, Rplus, {'mu': R, 'tau': Rplusbig},
            lambda value, mu, tau: sp.lognorm.logpdf(value, tau**-.5, 0, np.exp(mu))
            )

def test_t():
    pymc3_matches_scipy(
            StudentT, R, {'nu': Rplus, 'mu': R, 'lam': Rplus},
            lambda value, nu, mu, lam: sp.t.logpdf(value, nu, mu, lam**-.5)
            )


def test_cauchy():
    pymc3_matches_scipy(
            Cauchy, R, {'alpha': R, 'beta': Rplusbig},
            lambda value, alpha, beta: sp.cauchy.logpdf(value, alpha, beta)
            )

def test_half_cauchy():
    pymc3_matches_scipy(
            HalfCauchy, Rplus, {'beta': Rplusbig},
            lambda value, beta: sp.halfcauchy.logpdf(value, scale=beta)
            )

def test_gamma():
    pymc3_matches_scipy(
            Gamma, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
            lambda value, alpha, beta: sp.gamma.logpdf(value, alpha, scale=1.0/beta)
            )
    pymc3_matches_scipy(
            Gamma, Rplus, {'mu': Rplusbig, 'sd': Rplusbig},
            lambda value, mu, sd: sp.gamma.logpdf(value, mu**2 / sd**2, scale=1.0/(mu / sd**2))
            )

def test_inverse_gamma():
    pymc3_matches_scipy(
            InverseGamma, Rplus, {'alpha': Rplus, 'beta': Rplus},
            lambda value, alpha, beta: sp.invgamma.logpdf(value, alpha, scale=beta)
            )

def test_pareto():
    pymc3_matches_scipy(
            Pareto, Rplus, {'alpha': Rplusbig, 'm': Rplusbig},
            lambda value, alpha, m: sp.pareto.logpdf(value, alpha, scale=m)
            )

def scipy_exponweib_sucks(value, alpha, beta):
    """
    This function is required because SciPy's implementation of
    the Weibull PDF fails for some valid combinations of parameters, while the
    log-PDF fails for others.
    """
    pdf = numpy.log(sp.exponweib.pdf(value, 1, alpha, scale=beta))
    logpdf = sp.exponweib.logpdf(value, 1, alpha, scale=beta)

    if np.isinf(pdf):
        return logpdf
    return pdf

def test_weibull():
    pymc3_matches_scipy(
            Weibull, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
            scipy_exponweib_sucks
            )


def test_binomial():
    pymc3_matches_scipy(
            Binomial, Nat, {'n': NatSmall, 'p': Unit},
            lambda value, n, p: sp.binom.logpmf(value, n, p)
            )

def test_betabin():
    checkd(BetaBin, Nat, {'alpha': Rplus, 'beta': Rplus, 'n': NatSmall})


def test_bernoulli():
    pymc3_matches_scipy(
            Bernoulli, Bool, {'p': Unit},
            lambda value, p: sp.bernoulli.logpmf(value, p)
            )

def test_poisson():
    pymc3_matches_scipy(
            Poisson, Nat, {'mu': Rplus},
            lambda value, mu: sp.poisson.logpmf(value, mu)
            )

def test_constantdist():
    pymc3_matches_scipy(
            ConstantDist, I, {'c': I},
            lambda value, c: np.log(c == value)
            )

def test_zeroinflatedpoisson():
    checkd(ZeroInflatedPoisson, I, {'theta': Rplus, 'z': Bool})

def test_mvnormal():
    for n in [1, 2]:
        yield check_mvnormal, n

def check_mvnormal(n):
    pymc3_matches_scipy(
            MvNormal, Vector(R,n), {'mu': Vector(R,n), 'tau': PdMatrix(n)},
            normal_logpdf
            )

def normal_logpdf(value, mu, tau):
    (k,) = value.shape
    return  (-k/2)* np.log(2*np.pi) + .5 * np.log(np.linalg.det(tau)) - .5*(value-mu).dot(tau).dot(value -mu)

def test_wishart():
    # for n in [2,3]:
    #     yield check_wishart,n
    # This check compares the autodiff gradient to the numdiff gradient.
    # However, due to the strict constraints of the wishart,
    # it is impossible to numerically determine the gradient as a small
    # pertubation breaks the symmetry. Thus disabling.
    pass

def check_wishart(n):
    checkd(Wishart, PdMatrix(n), {'n': Domain([2, 3, 4, 2000]) , 'V': PdMatrix(n) }, checks = [check_dlogp])

def test_lkj():
    """
    Log probabilities calculated using the formulas in:
    http://www.sciencedirect.com/science/article/pii/S0047259X09000876
    """
    tri = np.array([0.7, 0.0, -0.7])
    test_cases = [
        (tri, 1, 1.5963125911388549),
        (tri, 3, -7.7963493376312742),
        (tri, 0, -np.inf),
        (np.array([1.1, 0.0, -0.7]), 1, -np.inf),
        (np.array([0.7, 0.0, -1.1]), 1, -np.inf)
    ]
    for t, n, logp in test_cases:
        yield check_lkj, t, n, 3, logp

def check_lkj(x, n, p, lp):
    with Model() as model:
        lkj = LKJCorr('lkj', n=n, p=p)
    pt = {'lkj': x}
    assert_almost_equal(model.fastlogp(pt),
                        lp, decimal=6, err_msg=str(pt))

def betafn(a):
    return scipy.special.gammaln(a).sum(0) - scipy.special.gammaln(a.sum(0))

def logpow(v, p):
    return np.choose(v==0, [p * np.log(v), 0])

def dirichlet_logpdf(value, a):
    return (-betafn(a) + logpow(value, a-1).sum(0)).sum()

def test_dirichlet():
    for n in [2,3]:
        yield check_dirichlet, n
    yield check_dirichlet2D, 2, 2

def check_dirichlet(n):
        pymc3_matches_scipy(
                Dirichlet, Simplex(n), {'a': Vector(Rplus, n) },
                dirichlet_logpdf
                )

def check_dirichlet2D(ndep, nind):
        pymc3_matches_scipy(
                Dirichlet, MultiSimplex(ndep, nind), {'a': Vector(Vector(Rplus, nind), ndep) },
                dirichlet_logpdf
                )

def multinomial_logpdf(value, n, p):
    if value.sum() == n and all(value >= 0) and all(value <= n) :
        return scipy.special.gammaln(n+1) - scipy.special.gammaln(value+1).sum() + logpow(p, value).sum()
    else:
        return -inf

def test_multinomial():
    for n in [2,3]:
        yield check_multinomial, n

def check_multinomial(n):
        pymc3_matches_scipy(
                Multinomial, Vector(Nat, n), {'p': Simplex(n), 'n' : Nat },
                multinomial_logpdf)

def categorical_logpdf(value, p):
    if value >= 0 and value <= len(p):
        return np.log(p[value])
    else:
        return -inf

def test_categorical():
    for n in [2,3,4]:
        yield check_categorical, n

def check_categorical(n):
    pymc3_matches_scipy(
        Categorical, Domain(range(n), 'int64'), {'p': Simplex(n)},
        lambda value, p: categorical_logpdf(value, p)
        )

def test_densitydist():
    def logp(x):
        return -log(2 * .5) - abs(x - .5) / .5

    checkd(DensityDist, R, {}, extra_args={'logp': logp})


def test_addpotential():
    with Model() as model:
        value = Normal('value', 1, 1)
        value_squared = Potential('value_squared', -value ** 2)

        check_dlogp(model, value, R, {})



def pymc3_matches_scipy(pymc3_dist, domain, paramdomains, scipy_dist, extra_args={}):
    model= build_model(pymc3_dist, domain, paramdomains, extra_args)
    value = model.named_vars['value']

    def logp(args):
        return scipy_dist(**args)

    check_logp(model, value, domain, paramdomains, logp)



def test_bound():
    with Model() as model:
        PositiveNormal = Bound(Normal, -.2)
        value = PositiveNormal('value', 1, 1)

        Rplus2 = Domain([-.2, -.19, -.1, 0, .5, 1, inf])

        check_dlogp(model, value, Rplus2, {})

def test_get_tau_sd():
    sd = np.array([2])
    assert_almost_equal(distributions.continuous.get_tau_sd(sd=sd), [1./sd**2, sd])

def check_int_to_1(model, value, domain, paramdomains):
    pdf = model.fastfn(exp(model.logpt))

    for pt in product(paramdomains):
        pt = Point(pt, value=value.tag.test_value, model=model)

        bij = DictToVarBijection(value, (), pt)
        pdfx = bij.mapf(pdf)

        area = integrate_nd(pdfx, domain, value.dshape, value.dtype)

        assert_almost_equal(area, 1, err_msg=str(pt))

def integrate_nd(f, domain, shape, dtype):

    if shape == () or shape == (1,):
        if dtype in continuous_types:
            return integrate.quad(f, domain.lower, domain.upper, epsabs=1e-8)[0]
        else:
            return np.sum(list(map(f, np.arange(domain.lower, domain.upper + 1))))
    elif shape == (2,):
        def f2(a, b):
            return f([a, b])

        return integrate.dblquad(f2,
                                 domain.lower[0],
                                 domain.upper[0],
                                 lambda a: domain.lower[1],
                                 lambda a: domain.upper[1])[0]

    elif shape == (3,):
        def f3(a, b, c):
            return f([a, b, c])

        return integrate.tplquad(f3,
                                 domain.lower[0], domain.upper[0],
                                 lambda a: domain.lower[1],
                                 lambda a: domain.upper[1],
                                 lambda a, b: domain.lower[2],
                                 lambda a, b: domain.upper[2])[0]
    else:
        raise ValueError("Dont know how to integrate shape: " + str(shape))


def check_dlogp(model, value, domain, paramdomains):
    try:
        from numdifftools import Gradient
    except ImportError:
        return

    domains = paramdomains.copy()
    domains['value'] = domain

    bij = DictToArrayBijection(
        ArrayOrdering(model.cont_vars), model.test_point)

    if not model.cont_vars:
        return

    dlogp = bij.mapf(model.fastdlogp(model.cont_vars))
    logp = bij.mapf(model.fastlogp)


    def wrapped_logp(x):
        try :
            return logp(x)
        except:
            return np.nan

    ndlogp = Gradient(wrapped_logp)

    for pt in product(domains):
        pt = Point(pt, model=model)

        pt = bij.map(pt)

        assert_almost_equal(dlogp(pt), ndlogp(pt),
                            decimal=6, err_msg=str(pt))

def check_logp(model, value, domain, paramdomains, logp_reference):
    domains = paramdomains.copy()
    domains['value'] = domain


    logp = model.fastlogp

    for pt in product(domains):
        pt = Point(pt, model=model)

        assert_almost_equal(logp(pt), logp_reference(pt),
                            decimal=6, err_msg=str(pt))


def build_model(distfam, valuedomain, vardomains, extra_args={}):
    with Model() as m:
        vars = dict((v, Flat(
            v, dtype=dom.dtype, shape=dom.shape, testval=dom.vals[0])) for v, dom in vardomains.items())
        vars.update(extra_args)

        value = distfam(
            'value', shape=valuedomain.shape, transform=None, **vars)
    return m


def checkd(distfam, valuedomain, vardomains,
           checks=(check_int_to_1, check_dlogp), extra_args={}):

        m = build_model(distfam, valuedomain, vardomains, extra_args=extra_args)


        for check in checks:
            check(m, m.named_vars['value'], valuedomain, vardomains)



def test_ex_gaussian():
    # Log probabilities calculated using the dexGAUS function from the R package gamlss.
    # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/.
    test_cases = [
        (0.5, -50.000, 0.500, 0.500, -99.8068528),
        (1.0, -1.000, 0.001, 0.001, -1992.5922447),
        (2.0, 0.001, 1.000, 1.000, -1.6720416),
        (5.0, 0.500, 2.500, 2.500, -2.4543644),
        (7.5, 2.000, 5.000, 5.000, -2.8259429),
        (15.0, 5.000, 7.500, 7.500, -3.3093854),
        (50.0, 50.000, 10.000, 10.000, -3.6436067),
        (1000.0, 500.000, 10.000, 20.000, -27.8707323)
        ]
    for value, mu, sigma, nu, logp in test_cases:
        yield check_ex_gaussian, value, mu, sigma, nu, logp

def check_ex_gaussian(value, mu, sigma, nu, logp):
    with Model() as model:
        ig = ExGaussian('eg', mu=mu, sigma=sigma, nu=nu)
    pt = {'eg': value}
    assert_almost_equal(model.fastlogp(pt),
                logp, decimal=6, err_msg=str(pt))


def test_multidimensional_beta_construction():
    with Model() as m:
        testMultiBeta = Beta('beta', alpha = 1., beta = 1., shape=(10,20))
