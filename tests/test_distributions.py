import itertools as its
from checks import * 
from pymc import *
from numpy import array, inf

from scipy import integrate
from numdifftools import Gradient


R = array([-inf, -2.1, -1, -.01, .0, .01, 1, 2.1, inf])
Rplus = array([0, .01,.1, .9, .99, 1, 1.5, 2, 100, inf])
Rplusbig = array([0, .5, .9, .99, 1, 1.5, 2, 20, inf] )
Unit = array([0,.001,  .1, .5, .75, .99, 1])

Runif = array([-1,-.4, 0, .4, 1] )
Rplusunif = array([0, .5, inf] )

I = array([-1000, -3,-2, -1, 0, 1, 2,3, 1000])
Nat = array([0,1,2,3, 5000] )
Bool = array([0,0,1, 1])
Natbig = array([0, 3,4,5, 1000])



def test_distributions():

    yield checkd, Uniform, Runif, {'lower' : -Rplusunif, 'upper' : Rplusunif}
    yield checkd, Normal, R, {'mu' : R, 'tau' : Rplus}
    yield checkd, Beta, Unit, {'alpha' : Rplus*5, 'beta' : Rplus*5}
    yield checkd, Exponential, Rplus, {'lam' : Rplus}
    yield checkd, T, R, {'nu' : Rplus, 'mu' : R, 'lam' : Rplus}
    yield checkd, Cauchy, R, {'alpha' : R, 'beta' : Rplusbig}
    yield checkd, Gamma, Rplus, {'alpha' : Rplusbig, 'beta' : Rplusbig}


    yield checkd, Binomial, Nat, {'n' : Natbig, 'p' : Unit}
    yield checkd, BetaBin, Nat, {'alpha' : Rplus, 'beta' : Rplus, 'n' : Natbig}
    yield checkd, Bernoulli, Bool, {'p' : Unit}
    yield checkd, Poisson, Nat, {'mu' : Rplus}
    yield checkd, ConstantDist, I, {'c' : I}




def checkd(distfam, valuedomain, vardomains):

    m = Model()

    vars = dict((v , m.Var(v, Flat(), dtype = dom.dtype)) for v,dom in vardomains.iteritems())
    value = m.Var('value', distfam(**vars),shape = [1], testval = valuedomain[len(valuedomain)//2])
    vardomains['value'] = np.array(valuedomain)

    domains = [np.array(vardomains[str(v)]) for v in m.vars]

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

        
        if value.dtype in continuous_types:
            area = integrate.quad(pdfx, lower, upper, epsabs = 1e-8)[0]
        else: 
            area = np.sum(map(pdfx, np.arange(lower, upper + 1)))

        assert_almost_equal(area, 1, err_msg = str(pt))


def check_dlogp(model, value, domains): 

    domains = [d[1:-1] for d in domains]
    bij = DictToArrayBijection(IdxMap(model.cont_vars), model.test_point)

    dlp = model.dlogpc()
    dlogp = bij.mapf(model.dlogpc())
    
    lp = model.logpc
    logp = bij.mapf(model.logpc)
    ndlogp = Gradient(logp)

    for a in its.product(*domains):
        pt = clean_point(dict( (str(var), val) for var,val in zip(model.vars, a)))

        #print lp.f.maker.fgraph.inputs, dlp.f.maker.fgraph.inputs 

        pt = bij.map(pt)
