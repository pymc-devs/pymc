import numpy as np

from pymc3.blocking import DictToArrayBijection
from . import models
from pymc3.step_methods.hmc.base_hmc import BaseHMC
import pymc3
from pymc3.theanof import floatX
from .checks import close_to
from .helpers import select_by_precision


def test_leapfrog_reversible():
    n = 3
    np.random.seed(42)
    start, model, _ = models.non_normal(n)
    size = model.ndim
    scaling = floatX(np.random.randn(size) ** 2)
    step = BaseHMC(vars=model.vars, model=model, scaling=scaling)
    step.integrator._logp_dlogp_func.set_extra_values({})
    p = floatX(step.potential.random())
    q = floatX(np.random.randn(size))
    start = step.integrator.compute_state(p, q)
    precision = select_by_precision(float64=1E-8, float32=1E-4)
    for epsilon in [.01, .1, 1.2]:
        for n_steps in [1, 2, 3, 4, 20]:
            state = start
            for _ in range(n_steps):
                state = step.integrator.step(epsilon, state)
            for _ in range(n_steps):
                state = step.integrator.step(-epsilon, state)
            close_to(state.q, start.q, precision, str((n_steps, epsilon)))
            close_to(state.p, start.p, precision, str((n_steps, epsilon)))


def test_nuts_tuning():
    model = pymc3.Model()
    with model:
        pymc3.Normal("mu", mu=0, sd=1)
        step = pymc3.NUTS()
        trace = pymc3.sample(10, step=step, tune=5, progressbar=False)

    assert not step.tune
    assert np.all(trace['step_size'][5:] == trace['step_size'][5])
