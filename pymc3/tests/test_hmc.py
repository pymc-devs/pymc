#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging

import numpy as np
import numpy.testing as npt

import pymc3

from pymc3.step_methods.hmc.base_hmc import BaseHMC
from pymc3.tests import models
from pymc3.theanof import floatX

logger = logging.getLogger("pymc3")


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
    for epsilon in [0.01, 0.1]:
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
        trace = pymc3.sample(
            10, step=step, tune=5, progressbar=False, chains=1, return_inferencedata=False
        )

    assert not step.tune
    assert np.all(trace["step_size"][5:] == trace["step_size"][5])
