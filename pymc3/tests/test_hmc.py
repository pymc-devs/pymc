import numpy as np

from pymc3.blocking import DictToArrayBijection
from . import models
from pymc3.step_methods.hmc.base_hmc import BaseHMC
import pymc3
from pymc3.theanof import floatX
from .checks import close_to
from .helpers import select_by_precision
import pytest
import theano


def test_leapfrog_reversible():
    n = 3
    start, model, _ = models.non_normal(n)
    step = BaseHMC(vars=model.vars, model=model)
    bij = DictToArrayBijection(step.ordering, start)
    q0 = bij.map(start)
    p0 = floatX(np.ones(n) * .05)
    precision = select_by_precision(float64=1E-8, float32=1E-4)
    for epsilon in [.01, .1, 1.2]:
        for n_steps in [1, 2, 3, 4, 20]:

            q, p = q0, p0
            q, p, _ = step.leapfrog(q, p, floatX(np.array(epsilon)), np.array(n_steps, dtype='int32'))
            q, p, _ = step.leapfrog(q, -p, floatX(np.array(epsilon)), np.array(n_steps, dtype='int32'))
            close_to(q, q0, precision, str((n_steps, epsilon)))
            close_to(-p, p0, precision, str((n_steps, epsilon)))

@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
def test_leapfrog_reversible_single():
    n = 3
    start, model, _ = models.non_normal(n)

    integrators = ['leapfrog', 'two-stage', 'three-stage']
    steps = [BaseHMC(vars=model.vars, model=model, integrator=method, use_single_leapfrog=True)
             for method in integrators]
    for method, step in zip(integrators, steps):
        bij = DictToArrayBijection(step.ordering, start)
        q0 = bij.map(start)
        p0 = floatX(np.ones(n) * .05)
        precision = select_by_precision(float64=1E-8, float32=1E-5)
        for epsilon in [0.01, 0.1, 1.2]:
            for n_steps in [1, 2, 3, 4, 20]:
                dlogp0 = step.dlogp(q0)

                q, p = q0, p0
                dlogp = dlogp0

                energy = step.compute_energy(q, p)
                for _ in range(n_steps):
                    q, p, v, dlogp, _ = step.leapfrog(q, p, dlogp, floatX(np.array(epsilon)))
                p = -p
                for _ in range(n_steps):
                    q, p, v, dlogp, _ = step.leapfrog(q, p, dlogp, floatX(np.array(epsilon)))

                close_to(q, q0, precision, str(('q', method, n_steps, epsilon)))
                close_to(-p, p0, precision, str(('p', method, n_steps, epsilon)))


def test_nuts_tuning():
    model = pymc3.Model()
    with model:
        pymc3.Normal("mu", mu=0, sd=1)
        step = pymc3.NUTS()
        trace = pymc3.sample(10, step=step, tune=5, progressbar=False)

    assert not step.tune
    assert np.all(trace['step_size'][5:] == trace['step_size'][5])
