import itertools as its
from checks import * 
from pymc import *
from numpy import inf

from scipy import integrate
from numdifftools import Gradient

def test_distributions():

    R = np.array([-inf, -2.1, -1, -.01, .0, .01, 1, 2.1, inf])
    Rplus = np.array([0, .01,.1, .9, .99, 1, 1.5, 2, 100, inf])
    Unit = np.array([0,.001,  .1, .5, .75, .99, 1])

    Runif = np.array([-2., -.5, 0,.5, 1, 2. ])
    Rplusunif = np.array([0, .5, inf]) 

    yield checkd, Uniform, np.array([-1,-.4, 0, .4, 1]), {'lower' : -Rplusunif, 'upper' : Rplusunif}
    yield checkd, Normal, R, {'mu' : R, 'tau' : Rplus}
    yield checkd, Beta, Unit, {'alpha' : Rplus*5, 'beta' : Rplus*5}


def checkd(distfam, valuedomain, vardomains):

    m = Model()
    
    vars = dict((v , m.Var(v, Flat())) for v in vardomains.keys())
    value = m.Var('value', distfam(**vars))
    vardomains['value'] = valuedomain

    domains = [vardomains[str(v)] for v in m.vars]

    check_int_to_1(m, value, domains)
    check_dlogp(m, value, domains)

    
def check_int_to_1(model, value, domains): 
    pdf = compilef(exp(model.logp))

    lower, upper = np.min(domains[-1]), np.max(domains[-1])

    domains = [d[1:-1] for d in domains[:-1]]


    for a in its.product(*domains):
        a = a + (value.tag.test_value,)
        pt = clean_point(dict( (str(var), val) for var,val in zip(model.vars, a)))

        bij = DictToVarBijection(value,0 ,  pt)
        
        pdfx = bij.mapf(pdf)

        area = integrate.quad(pdfx, lower, upper, epsabs = 1e-8)[0]
        assert_almost_equal(area, 1, err_msg = str(pt))


def check_dlogp(model, value, domains): 

    domains = [d[1:-1] for d in domains]
    bij = DictToArrayBijection(IdxMap(model.vars), model.test_point)

    dlogp = bij.mapf(model.dlogpc())
    
    logp = bij.mapf(model.logpc)
    ndlogp = Gradient(logp)

    for a in its.product(*domains):
        pt = np.array(a)

        assert_almost_equal(dlogp(pt), ndlogp(pt), decimal = 6, err_msg = str(pt))


        
