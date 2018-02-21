import matplotlib
matplotlib.use('Agg', warn=False)  # noqa

import numpy as np
import pymc3 as pm
from .checks import close_to

from .models import multidimensional_model, simple_categorical
from ..plots import traceplot, forestplot, autocorrplot, plot_posterior, energyplot, densityplot, pairplot
from ..plots.utils import make_2d
from ..step_methods import Slice, Metropolis
from ..sampling import sample
from ..tuning.scaling import find_hessian
from .test_examples import build_disaster_model
from pymc3.examples import arbitrary_stochastic as asmod
import theano
import pytest


def test_plots():
    # Test single trace
    with asmod.build_model() as model:
        start = model.test_point
        h = find_hessian(start)
        step = Metropolis(model.vars, h)
        trace = sample(3000, tune=0, step=step, start=start, chains=1)

    traceplot(trace)
    forestplot(trace)
    plot_posterior(trace)
    autocorrplot(trace)
    energyplot(trace)
    densityplot(trace) 

def test_energyplot():
    with asmod.build_model():
        trace = sample(cores=1)

    energyplot(trace)
    energyplot(trace, shade=0.5, alpha=0)
    energyplot(trace, kind='hist')


def test_plots_categorical():
    # Test single trace
    start, model, _ = simple_categorical()
    with asmod.build_model() as model:
        start = model.test_point
        h = find_hessian(start)
        step = Metropolis(model.vars, h)
        trace = sample(3000, tune=0, step=step, start=start, chains=1)

    traceplot(trace)


def test_plots_multidimensional():
    # Test multiple trace
    start, model, _ = multidimensional_model()
    with model:
        h = np.diag(find_hessian(start))
        step = Metropolis(model.vars, h)
        trace = sample(3000, tune=0, step=step, start=start)
    
    traceplot(trace)
    plot_posterior(trace)
    forestplot(trace)
    densityplot(trace)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on GPU due to cores=2")
def test_multichain_plots():
    model = build_disaster_model()
    with model:
        # Run sampler
        step1 = Slice([model.early_mean_log__, model.late_mean_log__])
        step2 = Metropolis([model.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = sample(1000, tune=0, step=[step1, step2], start=start, cores=2)

    forestplot(ptrace, varnames=['early_mean', 'late_mean'])
    autocorrplot(ptrace, varnames=['switchpoint'])
    plot_posterior(ptrace)


def test_make_2d():
    a = np.arange(4)
    close_to(make_2d(a), a[:, None], 0)

    n = 7
    a = np.arange(n * 4 * 5).reshape((n, 4, 5))
    res = make_2d(a)

    assert res.shape == (n, 20)
    close_to(a[:, 0, 0], res[:, 0], 0)
    close_to(a[:, 3, 2], res[:, 2 * 4 + 3], 0)


def test_plots_transformed():
    with pm.Model():
        pm.Uniform('x', 0, 1)
        step = pm.Metropolis()
        trace = pm.sample(100, tune=0, step=step, chains=1)

    assert traceplot(trace).shape == (1, 2)
    assert traceplot(trace, plot_transformed=True).shape == (2, 2)
    assert autocorrplot(trace).shape == (1, 1)
    assert autocorrplot(trace, plot_transformed=True).shape == (2, 1)
    assert plot_posterior(trace).numCols == 1
    assert plot_posterior(trace, plot_transformed=True).shape == (2, )

    with pm.Model():
        pm.Uniform('x', 0, 1)
        step = pm.Metropolis()
        trace = pm.sample(100, tune=0, step=step, chains=2)

    assert traceplot(trace).shape == (1, 2)
    assert traceplot(trace, plot_transformed=True).shape == (2, 2)
    assert autocorrplot(trace).shape == (1, 2)
    assert autocorrplot(trace, plot_transformed=True).shape == (2, 2)
    assert plot_posterior(trace).numCols == 1
    assert plot_posterior(trace, plot_transformed=True).shape == (2, )

def test_pairplot():
    with pm.Model() as model:
        a = pm.Normal('a', shape=2)
        c = pm.HalfNormal('c', shape=2)
        b = pm.Normal('b', a, c, shape=2)
        d = pm.Normal('d', 100, 1)
        trace = pm.sample(1000)

    pairplot(trace)
    pairplot(trace, hexbin=True, plot_transformed=True)
    pairplot(trace, sub_varnames=['a_0', 'c_0', 'b_1'])
    