import itertools as its
from checks import *
from pymc import *
from numpy import array, inf

from scipy import integrate
from numdifftools import Gradient


R = array([-inf, -2.1, -1, -.01, .0, .01, 1, 2.1, inf])
Rplus = array([0, .01, .1, .9, .99, 1, 1.5, 2, 100, inf])
Rplusbig = array([0, .5, .9, .99, 1, 1.5, 2, 20, inf])
Unit = array([0, .001, .1, .5, .75, .99, 1])

Runif = array([-1, -.4, 0, .4, 1])
Rdunif = array([-10, 0, 10])
Rplusunif = array([0, .5, inf])
Rplusdunif = array([2, 10, 100])

I = array([-1000, -3, -2, -1, 0, 1, 2, 3, 1000], 'int64')

NatSmall = array([0, 3, 4, 5, 1000], 'int64')
Nat = array([0, 1, 2, 3, 2000], 'int64')
NatBig = array([0, 1, 2, 3, 5000, 50000], 'int64')

Bool = array([0, 0, 1, 1], 'int64')


def test_unif():
    checkd(Uniform, Runif, {'lower': -Rplusunif, 'upper': Rplusunif})


def test_discrete_unif():
    checkd(DiscreteUniform, Rdunif,
           {'lower': -Rplusdunif, 'upper': Rplusdunif})


def test_flat():
    checkd(Flat, Runif, {}, False)


def test_normal():
    checkd(Normal, R, {'mu': R, 'tau': Rplus})


def test_beta():
    checkd(Beta, Unit, {'alpha': Rplus * 5, 'beta': Rplus * 5})


def test_exponential():
    checkd(Exponential, Rplus, {'lam': Rplus})


def test_geometric():
    checkd(Geometric, NatBig, {'p': Unit})


def test_negative_binomial():
    checkd(NegativeBinomial, Nat, {'mu': Rplusbig, 'alpha': Rplusbig})


def test_laplace():
    checkd(Laplace, R, {'mu': R, 'b': Rplus})


def test_t():
    checkd(T, R, {'nu': Rplus, 'mu': R, 'lam': Rplus})


def test_cauchy():
    checkd(Cauchy, R, {'alpha': R, 'beta': Rplusbig})


def test_gamma():
    checkd(Gamma, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig})


def test_tpos():
    checkd(Tpos, Rplus, {'nu': Rplus, 'mu': R, 'lam': Rplus}, False)


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
    checkd(ZeroInflatedPoisson, I, {'theta': Rplus, 'z': Bool})


def test_densitydist():
    def logp(x):
        return -log(2 * .5) - abs(x - .5) / .5

    checkd(DensityDist, R, {}, extra_args={'logp': logp})


def test_addpotential():
    with Model() as model:
        x = Normal('x', 1, 1)
        model.AddPotential(-x ** 2)

        check_dlogp(model, x, [R])


def checkd(distfam, valuedomain, vardomains,
           check_int=True, check_der=True, extra_args={}):

    with Model() as m:
        vars = dict((v, Flat(
            v, dtype=dom.dtype)) for v, dom in vardomains.iteritems())
        vars.update(extra_args)
        # print vars
        value = distfam(
            'value', **vars)

        vardomains['value'] = np.array(valuedomain)

        domains = [np.array(vardomains[str(v)]) for v in m.vars]

        if check_int:
            check_int_to_1(m, value, domains)
        if check_der:
            check_dlogp(m, value, domains)


def check_int_to_1(model, value, domains):
    pdf = compilef(exp(model.logp))

    lower, upper = np.min(domains[-1]), np.max(domains[-1])

    domains = [d[1:-1] for d in domains[:-1]]

    for a in its.product(*domains):
        a = a + (value.tag.test_value,)
        pt = Point(dict((
            str(var), val) for var, val in zip(model.vars, a)), model=model)

        bij = DictToVarBijection(value, (), pt)

        pdfx = bij.mapf(pdf)

        if value.dtype in continuous_types:
            area = integrate.quad(pdfx, lower, upper, epsabs=1e-8)[0]
        else:
            area = np.sum(map(pdfx, np.arange(lower, upper + 1)))

        assert_almost_equal(area, 1, err_msg=str(pt))


def check_dlogp(model, value, domains):

    domains = [d[1:-1] for d in domains]
    bij = DictToArrayBijection(
        ArrayOrdering(model.cont_vars), model.test_point)

    if not model.cont_vars:
        return

    dlp = model.dlogpc()
    dlogp = bij.mapf(model.dlogpc())

    lp = model.logpc
    logp = bij.mapf(model.logpc)
    ndlogp = Gradient(logp)

    for a in its.product(*domains):
        pt = Point(dict((
            str(var), val) for var, val in zip(model.vars, a)), model=model)

        pt = bij.map(pt)
