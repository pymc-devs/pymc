import numpy as np
import numpy.testing as npt

from . import models
from pymc3.step_methods.hmc.base_hmc import BaseHMC
from pymc3.exceptions import SamplingError
import pymc3
import pytest
import logging
from pymc3.theanof import floatX
logger = logging.getLogger('pymc3')

def test_leapfrog_reversible():
    n = 3
    np.random.seed(42)
    start, model, _ = models.non_normal(n)
    size = model.ndim
    scaling = floatX(np.random.rand(size))
    step = BaseHMC(vars=model.vars, model=model, scaling=scaling)
    step.integrator._logp_dlogp_func.set_extra_values({})
    p = floatX(step.potential.random())
    q = floatX(np.random.randn(size))
    start = step.integrator.compute_state(p, q)
    for epsilon in [.01, .1]:
        for n_steps in [1, 2, 3, 4, 20]:
            state = start
            for _ in range(n_steps):
                state = step.integrator.step(epsilon, state)
            for _ in range(n_steps):
                state = step.integrator.step(-epsilon, state)
            npt.assert_allclose(state.q, start.q, rtol=1e-5)
            npt.assert_allclose(state.p, start.p, rtol=1e-5)


def test_nuts_tuning():
    model = pymc3.Model()
    with model:
        pymc3.Normal("mu", mu=0, sigma=1)
        step = pymc3.NUTS()
        trace = pymc3.sample(10, step=step, tune=5, progressbar=False, chains=1)

    assert not step.tune
    assert np.all(trace['step_size'][5:] == trace['step_size'][5])

def test_nuts_error_reporting(caplog):
    model = pymc3.Model()
    with caplog.at_level(logging.CRITICAL) and pytest.raises(SamplingError):
        with model:
            pymc3.HalfNormal('a', sigma=1, transform=None, testval=-1)
            pymc3.HalfNormal('b', sigma=1, transform=None)
            trace = pymc3.sample(init='adapt_diag', chains=1)
        assert "Bad initial energy, check any log  probabilities that are inf or -inf: a        -inf\nb" in caplog.text

