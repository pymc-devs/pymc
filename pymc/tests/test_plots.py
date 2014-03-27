import matplotlib
matplotlib.use('Agg', warn=False)

import numpy as np 
from checks import close_to

import pymc.plots
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

        traceplot(trace)
        forestplot(trace)

        autocorrplot(trace)

def test_plots_multidimensional():

    # Test single trace
    from models import multidimensional_model


    start, model, _ = multidimensional_model()
    with model as model:
        h = np.diag(find_hessian(start))
        step = Metropolis(model.vars, h)
        trace = sample(3000, step, start)

        traceplot(trace)
        #forestplot(trace)
        #autocorrplot(trace)


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

def test_make_2d(): 

    a = np.arange(4)
    close_to(pymc.plots.make_2d(a), a[:,None], 0)

    n = 7
    a = np.arange(n*4*5).reshape((n,4,5))
    res = pymc.plots.make_2d(a)

    assert res.shape == (n,20)
    close_to(a[:,0,0], res[:,0], 0)
    close_to(a[:,3,2], res[:,2*4+3], 0)
