import matplotlib
matplotlib.use('Agg', warn=False)

import numpy as np
from .checks import close_to

from .models import multidimensional_model
from ..plots import traceplot, forestplot, autocorrplot, make_2d
from ..step_methods import Slice, Metropolis
from ..sampling import sample
from ..tuning.scaling import find_hessian
from pymc3.examples import disaster_model as dm, arbitrary_stochastic as asmod


def test_plots():
    # Test single trace
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
    start, model, _ = multidimensional_model()
    with model as model:
        h = np.diag(find_hessian(start))
        step = Metropolis(model.vars, h)
        trace = sample(3000, step, start)

        traceplot(trace)
        # forestplot(trace)
        # autocorrplot(trace)


def test_multichain_plots():
    with dm.model:
        # Run sampler
        step1 = Slice([dm.early_mean, dm.late_mean])
        step2 = Metropolis([dm.switchpoint])
        start = {'early_mean': 2., 'late_mean': 3., 'switchpoint': 50}
        ptrace = sample(1000, [step1, step2], start, njobs=2)

    forestplot(ptrace, varnames=['early_mean', 'late_mean'])
    autocorrplot(ptrace, varnames=['switchpoint'])


def test_make_2d():
    a = np.arange(4)
    close_to(make_2d(a), a[:, None], 0)

    n = 7
    a = np.arange(n * 4 * 5).reshape((n, 4, 5))
    res = make_2d(a)

    assert res.shape == (n, 20)
    close_to(a[:, 0, 0], res[:, 0], 0)
    close_to(a[:, 3, 2], res[:, 2 * 4 + 3], 0)
