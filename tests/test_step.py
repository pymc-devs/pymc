from checks import *
from models import simple_model
from theano.tensor import constant
from scipy.stats.mstats import moment

def check_stat(name, trace, var, stat, value, bound):
    s = stat(trace[var], axis =0)
    assert all(np.abs(s - value) < bound), "stats out of bounds : " + repr(s) + ", " + repr(value) + ", " + repr(bound)


def test_step_continuous():

    mu = np.array([-.1,.5, 1.1])
    p = np.array([
        [2. , 0 ,  0],
        [.05 , .1,  0],
        [1. ,-0.05,5.5]])

    tau = np.dot(p,p.T) 

    start = {'x' : np.array([.1, 1., .8])}
    model = pm.Model(start)
    Var = model.Var

    x = Var('x', pm.MvNormal(constant(mu),constant(tau)), 3)

    H = tau

    hmc = pm.hmc_step(model, model.vars,  H)
    mh = pm.metropolis_step(model, model.vars , np.linalg.inv(H), scaling = .25)
    compound = pm.compound_step([hmc, mh])

    steps = [mh, hmc, compound]

    unc = np.diag(tau)**-.5
    check = [('x', np.mean, mu, unc/10.)]

    for st in steps:
        for (var, stat, val, bound) in check:
            np.random.seed(1)
            h, _, _ = sample(8000, st, start)

            #return h
            yield check_stat,repr(st), h, var, stat, val, bound  



        

