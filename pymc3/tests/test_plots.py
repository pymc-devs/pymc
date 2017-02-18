import matplotlib
matplotlib.use('Agg', warn=False)

import numpy as np
import pymc3 as pm
from .checks import close_to

from .models import multidimensional_model, simple_categorical
from ..plots import traceplot, forestplot, autocorrplot, plot_posterior, make_2d
from ..step_methods import Slice, Metropolis
from ..sampling import sample
from ..tuning.scaling import find_hessian
from .test_examples import build_disaster_model
from pymc3.examples import arbitrary_stochastic as asmod


def test_plots():
    # Test single trace
    with asmod.build_model() as model:
        start = model.test_point
        h = find_hessian(start)
        step = Metropolis(model.vars, h)
        trace = sample(3000, step=step, start=start)

        traceplot(trace)
        forestplot(trace)
        plot_posterior(trace)
        autocorrplot(trace)


def test_plots_categorical():
    # Test single trace
    start, model, _ = simple_categorical()
    with asmod.build_model() as model:
        start = model.test_point
        h = find_hessian(start)
        step = Metropolis(model.vars, h)
        trace = sample(3000, step=step, start=start)

        traceplot(trace)


def test_plots_multidimensional():
    # Test single trace
    start, model, _ = multidimensional_model()
    with model:
        h = np.diag(find_hessian(start))
        step = Metropolis(model.vars, h)
        trace = sample(3000, step=step, start=start)

        traceplot(trace)
        plot_posterior(trace)


def test_multichain_plots():
    model = build_disaster_model()
    with model:
        # Run sampler
        step1 = Slice([model.early_mean_log_, model.late_mean_log_])
        step2 = Metropolis([model.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = sample(1000, step=[step1, step2], start=start, njobs=2)

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
    with pm.Model() as model:
        pm.Uniform('x', 0, 1)
        step = pm.Metropolis()
        trace = pm.sample(100, step=step)

    assert traceplot(trace).shape == (1, 2)
    assert traceplot(trace, plot_transformed=True).shape == (2, 2)
    assert autocorrplot(trace).shape == (1, 1)
    assert autocorrplot(trace, plot_transformed=True).shape == (2, 1)
    assert plot_posterior(trace).shape == (1, )
    assert plot_posterior(trace, plot_transformed=True).shape == (2, )
