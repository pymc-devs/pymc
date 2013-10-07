from itertools import product
from checks import *
from pymc import *
from numpy import array, inf
import numpy

from scipy import integrate
from knownfailure import *


class Domain(object):
    def __init__(self, testvals, dtype = None, edges=None, shape = None):
        atestvals = array(testvals)

        if edges is None:
            edges = array(testvals[0]), array(testvals[-1])
            testvals = testvals[1:-1]
        if shape is None:
            shape = atestvals[0].shape


        self.vals = testvals
        self.shape = shape

        self.lower, self.upper = edges
        self.dtype = atestvals.dtype

def test_set(domains): 
    return product(*[d.vals for d in domains])
        
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


Vec2 =Domain([ 
    [.1, 0.0],
    [-2.3, .1],
    [-2.3, 1.5],
    ], 
    edges = ([ -inf, -inf ], [inf, inf]))

PdMatrix = Domain([
    np.eye(2)
    ], 
    edges = (None,None))
    



def test_unif():
    checkd(Uniform, Runif, {'lower': -Rplusunif, 'upper': Rplusunif})


def test_discrete_unif():
    checkd(DiscreteUniform, Rdunif,
           {'lower': -Rplusdunif, 'upper': Rplusdunif})


def test_flat():
    checkd(Flat, Runif, {}, checks = [check_dlogp])


def test_normal():
    checkd(Normal, R, {'mu': R, 'tau': Rplus})


def test_beta():
    # TODO this fails with `Rplus`
    checkd(Beta, Unit, {'alpha': Rplusbig, 'beta': Rplusbig})


def test_exponential():
    checkd(Exponential, Rplus, {'lam': Rplus})


def test_geometric():
    checkd(Geometric, NatBig, {'p': Unit})


def test_negative_binomial():
    checkd(NegativeBinomial, Nat, {'mu': Rplusbig, 'alpha': Rplusbig})


def test_laplace():
    checkd(Laplace, R, {'mu': R, 'b': Rplus})

def test_lognormal():
    checkd(Lognormal, Rplus, {'mu': R, 'tau': Rplus}, False)

def test_t():
    checkd(T, R, {'nu': Rplus, 'mu': R, 'lam': Rplus})


def test_cauchy():
    checkd(Cauchy, R, {'alpha': R, 'beta': Rplusbig})


def test_gamma():
    checkd(Gamma, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig})


def test_tpos():
    checkd(Tpos, Rplus, {'nu': Rplus, 'mu': R, 'lam': Rplus}, checks = [check_dlogp])


def test_binomial():
    checkd(Binomial, Nat, {'n': NatSmall, 'p': Unit})


def test_betabin():
    checkd(BetaBin, Nat, {'alpha': Rplus, 'beta': Rplus, 'n': NatSmall})


def test_bernoulli():
    checkd(Bernoulli, Bool, {'p': Unit})


def test_poisson():
    checkd(Poisson, Nat, {'mu': Rplus})


def test_constantdist():
    checkd(ConstantDist, I, {'c': I})


def test_zeroinflatedpoisson():
    checkd(ZeroInflatedPoisson, Nat, {'theta': Rplus, 'z': Bool})

def test_mvnormal():
    checkd(MvNormal, Vec2, {'mu': R, 'Tau': PdMatrix}, checks = [check_dlogp])


def test_densitydist():
    def logp(x):
        return -log(2 * .5) - abs(x - .5) / .5

    checkd(DensityDist, R, {}, extra_args={'logp': logp})


def test_addpotential():
    with Model() as model:
        x = Normal('x', 1, 1)
        model.AddPotential(-x ** 2)

        check_dlogp(model, x, [R])



def test_wishart_initialization():
    with Model() as model:
        x = Wishart('wishart_test', n=3, p=2, V=numpy.eye(2), shape = [2,2])


def test_bound():
    with Model() as model:
        PositiveNormal = Bound(Normal, -.2)
        x = PositiveNormal('x', 1, 1)

        check_dlogp(model, x, [Rplus - .2])

def check_int_to_1(model, value, domain, paramdomains):
    pdf = compilef(exp(model.logp))
    names = map(str, model.vars)

    for a in test_set(paramdomains):
        a = a + (value.tag.test_value,)
        pt = Point(zip(names, a), model=model)

        bij = DictToVarBijection(value, (), pt)

        pdfx = bij.mapf(pdf)

        if value.dtype in continuous_types:
            area = integrate.quad(pdfx, domain.lower, domain.upper, epsabs=1e-8)[0]
        else:
            area = np.sum(map(pdfx, np.arange(domain.lower, domain.upper + 1)))

        assert_almost_equal(area, 1, err_msg=str(pt))


def check_dlogp(model, value, domains):
    try:
        from numdifftools import Gradient
    except ImportError:
        return

def check_dlogp(model, value, domain, paramdomains):

    domains = paramdomains + [domain] 
    bij = DictToArrayBijection(
        ArrayOrdering(model.cont_vars), model.test_point)

    if not model.cont_vars:
        return

    print model.named_vars['Tau'].tag.test_value
    dlogp = bij.mapf(model.dlogpc())
    logp = bij.mapf(model.logpc)

    ndlogp = Gradient(logp)
    names = map(str, model.vars)

    for a in test_set(domains):
        pt = Point(zip(names, a), model=model)

        pt = bij.map(pt)

        assert_almost_equal(dlogp(pt), ndlogp(pt),
                            decimal=6, err_msg=str(pt))

def build_model(distfam, valuedomain, vardomains, extra_args={}):
    with Model() as m:
        vars = dict((v, Flat(
            v, dtype=dom.dtype, shape=dom.shape, testval=dom.vals[0])) for v, dom in vardomains.iteritems())
        vars.update(extra_args)

        value = distfam(
            'value', shape=valuedomain.shape, **vars)
    return m 

def checkd(distfam, valuedomain, vardomains,
           checks = [check_int_to_1, check_dlogp], extra_args={}):

        m = build_model(distfam, valuedomain, vardomains, extra_args={})

        domains = [vardomains[str(v)] for v in m.vars[:-1]]

        for check in checks: 
            check(m, m.named_vars['value'], valuedomain, domains)

