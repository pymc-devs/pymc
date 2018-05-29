import pymc3 as pm
import numpy as np
import scipy.stats as sp
from .checks import close_to
from .models import simple_model, mv_simple
import theano
from theano import tensor as tt


tol = 2.0**-11

def test_logp():
    start, model, (mu, sig) = simple_model()
    lp = model.fastlogp
    lp(start)
    close_to(lp(start), sp.norm.logpdf(start['x'], mu, sig).sum(), tol)


def test_dlogp():
    start, model, (mu, sig) = simple_model()
    dlogp = model.fastdlogp()
    close_to(dlogp(start), -(start['x'] - mu) / sig**2, 1. / sig**2 / 100.)


def test_dlogp2():
    start, model, (_, sig) = mv_simple()
    H = np.linalg.inv(sig)
    d2logp = model.fastd2logp()
    close_to(d2logp(start), H, np.abs(H / 100.))


def test_deterministic():
    with pm.Model() as model:
        x = pm.Normal('x', 0, 1)
        y = pm.Deterministic('y', x**2)

    assert model.y == y
    assert model['y'] == y


def test_deterministic_name_handling():
    with pm.Model() as model:
        x = pm.Normal('x', 0, 1)
        a = tt.constant(0)
        a.name = 'a'
        b = tt.constant(0)
        b.name = 'b'
        wa = pm.Deterministic('wa', a)
        wb = pm.Deterministic('wb', b)
        c = wa * x + wb
        c.name = 'c'
        wc = pm.Deterministic('wc', c)

    # Test __class__ of the Determinisitc variables
    assert(not pm.util.MetaNameWrapped.is_name_wrapped_instance(a))
    assert(not pm.util.MetaNameWrapped.is_name_wrapped_instance(b))
    assert(not pm.util.MetaNameWrapped.is_name_wrapped_instance(c))
    assert(pm.util.MetaNameWrapped.is_name_wrapped_instance(wa))
    assert(pm.util.MetaNameWrapped.is_name_wrapped_instance(wb))
    assert(pm.util.MetaNameWrapped.is_name_wrapped_instance(wc))

    # Test name handling difference between theano variables and Deterministics
    assert(a.name=='b')
    assert(b.name=='b')
    assert(c.name=='c')
    assert(wa.name=='wa')
    assert(wb.name=='wb')
    assert(wc.name=='wc')
    assert(str(wa)==str(a))
    assert(str(wb)==str(a))
    assert(str(wc)==str(c))
    assert(hash(wa)==hash(wa.get_theano_instance()))
    assert(hash(wb)==hash(wb.get_theano_instance()))
    assert(hash(wc)==hash(wc.get_theano_instance()))
    assert(a==b)
    assert(wa!=wb)
    assert(wa.equals(wb))

    # Assert that the Apply nodes look the same to theano and that the outputs
    # point to the wrapped variables
    c_graph = tt.printing.debugprint(c, file='str')
    wc_graph = tt.printing.debugprint(wc, file='str') 
    assert(c_graph==wc_graph)
    assert(all([wc==oo for oo in [o for o in wc.owner.outputs if str(wc)==str(o)]]))

def get_inputs(x):
    if hasattr(x, 'owner') and x.owner is not None:
        out = []
        for i in x.owner.inputs:
            out.extend(get_inputs(i))
        return out
    else:
        if not isinstance(x, tt.TensorConstant):
            return [(x, x.name, type(x))]
        else:
            return []


def test_mapping():
    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1)
        sd = pm.Gamma('sd', 1, 1)
        y = pm.Normal('y', mu, sd, observed=np.array([.1, .5]))
    lp = model.fastlogp
    lparray = model.logp_array
    point = model.test_point
    parray = model.bijection.map(point)
    assert lp(point) == lparray(parray)

    randarray = np.random.randn(*parray.shape)
    randpoint = model.bijection.rmap(randarray)
    assert lp(randpoint) == lparray(randarray)



