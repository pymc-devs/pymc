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

import pymc

from pymc.aesaraf import floatX
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.step_methods.hmc.base_hmc import BaseHMC
from pymc.tests import models

logger = logging.getLogger("pymc")


def test_leapfrog_reversible():
    n = 3
    np.random.seed(42)
    start, model, _ = models.non_normal(n)
    size = sum(start[n.name].size for n in model.value_vars)
    scaling = floatX(np.random.rand(size))

    class HMC(BaseHMC):
        def _hamiltonian_step(self, *args, **kwargs):
            pass

    step = HMC(vars=model.value_vars, model=model, scaling=scaling)

    step.integrator._logp_dlogp_func.set_extra_values({})
    astart = DictToArrayBijection.map(start)
    p = RaveledVars(floatX(step.potential.random()), astart.point_map_info)
    q = RaveledVars(floatX(np.random.randn(size)), astart.point_map_info)
    start = step.integrator.compute_state(p, q)
    for epsilon in [0.01, 0.1]:
        for n_steps in [1, 2, 3, 4, 20]:
            state = start
            for _ in range(n_steps):
                state = step.integrator.step(epsilon, state)
            for _ in range(n_steps):
                state = step.integrator.step(-epsilon, state)
            npt.assert_allclose(state.q.data, start.q.data, rtol=1e-5)
            npt.assert_allclose(state.p.data, start.p.data, rtol=1e-5)


def test_nuts_tuning():
    with pymc.Model():
        pymc.Normal("mu", mu=0, sigma=1)
        step = pymc.NUTS()
        idata = pymc.sample(
            10, step=step, tune=5, discard_tuned_samples=False, progressbar=False, chains=1
        )

    assert not step.tune
    ss_tuned = idata.warmup_sample_stats["step_size"][0, -1]
    ss_posterior = idata.sample_stats["step_size"][0, :]
    np.testing.assert_array_equal(ss_posterior, ss_tuned)
