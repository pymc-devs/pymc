#   Copyright 2024 The PyMC Developers
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

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.pytensorf import floatX
from pymc.step_methods.hmc import HamiltonianMC
from pymc.step_methods.hmc.base_hmc import BaseHMC
from tests import models
from tests.helpers import RVsAssignmentStepsTester, StepMethodTester


class TestStepHamiltonianMC(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (lambda C, _: HamiltonianMC(scaling=C, is_cov=True, blocked=False), 1000),
            (lambda C, _: HamiltonianMC(scaling=C, is_cov=True), 1000),
        ],
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentHamiltonianMC(RVsAssignmentStepsTester):
    @pytest.mark.parametrize("step, step_kwargs", [(HamiltonianMC, {})])
    def test_continuous_steps(self, step, step_kwargs):
        self.continuous_steps(step, step_kwargs)


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

    astart = DictToArrayBijection.map(start)
    p = RaveledVars(floatX(step.potential.random()), astart.point_map_info)
    q = floatX(np.random.randn(size))
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
    with pm.Model():
        pm.Normal("mu", mu=0, sigma=1)
        step = pm.NUTS()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            idata = pm.sample(
                10, step=step, tune=5, discard_tuned_samples=False, progressbar=False, chains=1
            )

    assert not step.tune
    ss_tuned = idata.warmup_sample_stats["step_size"][0, -1]
    ss_posterior = idata.sample_stats["step_size"][0, :]
    np.testing.assert_array_equal(ss_posterior, ss_tuned)
