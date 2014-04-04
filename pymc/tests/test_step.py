from .checks import *
from .models import simple_model, mv_simple, mv_simple_discrete
from theano.tensor import constant
from scipy.stats.mstats import moment


def check_stat(name, trace, var, stat, value, bound):
    s = stat(trace[var][2000:], axis=0)
    close_to(s, value, bound)


def test_step_continuous():
    start, model, (mu, C) = mv_simple()

    with model:
        hmc = pm.HamiltonianMC(scaling=C, is_cov=True)
        mh = pm.Metropolis(S=C,
                           proposal_dist=pm.MultivariateNormalProposal)
        slicer = pm.Slice()
        nuts = pm.NUTS(scaling=C, is_cov=True)
        compound = pm.CompoundStep([hmc, mh])


    steps = [mh, hmc, slicer, nuts, compound]

    unc = np.diag(C) ** .5
    check = [('x', np.mean, mu, unc / 10.),
             ('x', np.std, unc, unc / 10.)]

    for st in steps:
        h = sample(8000, st, start, model=model, random_seed=1)
        for (var, stat, val, bound) in check:
            yield check_stat, repr(st), h, var, stat, val, bound

def test_step_discrete():
    start, model, (mu, C) = mv_simple_discrete()

    with model:
        mh = pm.Metropolis(S=C,
                           proposal_dist=pm.MultivariateNormalProposal)
        slicer = pm.Slice()


    steps = [mh]

    unc = np.diag(C) ** .5
    check = [('x', np.mean, mu, unc / 10.),
             ('x', np.std, unc, unc / 10.)]

    for st in steps:
        h = sample(20000, st, start, model=model, random_seed=1)

        for (var, stat, val, bound) in check:
            yield check_stat, repr(st), h, var, stat, val, bound
