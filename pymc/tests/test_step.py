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
        single_component_slicer = pm.SingleComponentSlice()
        compound = pm.CompoundStep([hmc, mh])

    steps = [mh, hmc, slicer, nuts, single_component_slicer, compound]

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


@unittest.skip("Test is failing, probably related to https://github.com/pymc-devs/pymc/issues/358")
def test_single_component_metropolis():
    start, model, (mu, tau) = simple_model()

    with model:
        single_component_metropolis = pm.SingleComponentMetropolis()

    check = [('x', np.mean, mu, tau / 10.),
             ('x', np.std, tau, tau / 10.)]

    h = sample(30000, single_component_metropolis, start, model=model, random_seed=1)[-10000:]

    for (var, stat, val, bound) in check:
        yield check_stat, 'single_component_metropolis', h, var, stat, val, bound
