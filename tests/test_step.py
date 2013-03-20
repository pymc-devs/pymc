from checks import *
from models import simple_model, mv_simple
from theano.tensor import constant
from scipy.stats.mstats import moment

def check_stat(name, trace, var, stat, value, bound):
    s = stat(trace[var], axis =0)
    close_to(s, value, bound)


def test_step_continuous():
    start, model, (mu, C) = mv_simple()
    H = np.linalg.inv(C)
    

    hmc = pm.HamiltonianStep(model, model.vars,  H)
    mh = pm.Metropolis(model, model.vars , C, scaling = .25)
    compound = pm.Compound([hmc, mh])

    steps = [mh, hmc, compound]

    unc = np.diag(C)**.5
    check = [('x', np.mean, mu, unc/10.),
             ('x', np.std , unc, unc/10.)]

    for st in steps:
        for (var, stat, val, bound) in check:
            np.random.seed(1)
            h, _, _ = sample(8000, st, start)

            yield check_stat,repr(st), h, var, stat, val, bound  



        

