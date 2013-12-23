#from ..plots import *
from pymc.plots import *
from pymc import psample, Slice, Metropolis, find_hessian, sample


def test_plots():

    # Test single trace
    from pymc.examples import arbitrary_stochastic as asmod

    with asmod.model as model:

        start = model.test_point
        h = find_hessian(start)
        step = Metropolis(model.vars, h)
        trace = sample(3000, step, start)

        forestplot(trace)

        autocorrplot(trace)


def test_multichain_plots():

    from pymc.examples import disaster_model as dm

    with dm.model as model:
        # Run sampler
        step1 = Slice([dm.early_mean, dm.late_mean])
        step2 = Metropolis([dm.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = psample(1000, [step1, step2], start, threads=2)

    forestplot(ptrace, vars=['early_mean', 'late_mean'])

    autocorrplot(ptrace, vars=['switchpoint'])
