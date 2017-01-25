import numpy as np

from pymc3.blocking import DictToArrayBijection
from . import models
from pymc3.step_methods.hmc.base_hmc import BaseHMC
import pymc3
from .checks import close_to


def test_leapfrog_reversible():
    n = 3
    start, model, _ = models.non_normal(n)
    step = BaseHMC(vars=model.vars, model=model)
    bij = DictToArrayBijection(step.ordering, start)
    q0 = bij.map(start)
    p0 = np.ones(n) * .05

    for epsilon in [.01, .1, 1.2]:
        for n_steps in [1, 2, 3, 4, 20]:

            q, p = q0, p0
            q, p, _ = step.leapfrog(q, p, np.array(epsilon), np.array(n_steps, dtype='int32'))
            q, p, _ = step.leapfrog(q, -p, np.array(epsilon), np.array(n_steps, dtype='int32'))

            close_to(q, q0, 1e-8, str((n_steps, epsilon)))
            close_to(-p, p0, 1e-8, str((n_steps, epsilon)))


def test_nuts_tuning():
    model = pymc3.Model()
    with model:
        mu = pymc3.Normal("mu", mu=0, sd=1)
        step = pymc3.NUTS()
        trace = pymc3.sample(10, step=step, tune=5, progressbar=False)
    assert not step.tune
