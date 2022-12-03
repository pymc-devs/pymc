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

import pytensor
import pytest

import pymc as pm

from pymc.step_methods import (
    NUTS,
    CompoundStep,
    DEMetropolis,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc.tests.helpers import StepMethodTester, fast_unstable_sampling_mode
from pymc.tests.models import simple_2model_continuous


class TestCompoundStep:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS, DEMetropolis)

    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    assert isinstance(sampler(blocked=False), CompoundStep)

    def test_blocked(self):
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    sampler_instance = sampler(blocked=True)
                    assert not isinstance(sampler_instance, CompoundStep)
                    assert isinstance(sampler_instance, sampler)

    def test_name(self):
        with pm.Model() as m:
            c1 = pm.HalfNormal("c1")
            c2 = pm.HalfNormal("c2")

            step1 = NUTS([c1])
            step2 = Slice([c2])
            step = CompoundStep([step1, step2])
        assert step.name == "Compound[nuts, slice]"


class TestStepCompound(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (
                lambda C, _: CompoundStep(
                    [
                        HamiltonianMC(scaling=C, is_cov=True),
                        HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                    ]
                ),
                1000,
            ),
        ],
        ids=str,
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentCompound:
    def test_compound_step(self):
        with pm.Model() as m:
            c1 = pm.HalfNormal("c1")
            c2 = pm.HalfNormal("c2")

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                step1 = NUTS([c1])
                step2 = NUTS([c2])
                step = CompoundStep([step1, step2])
            assert {m.rvs_to_values[c1], m.rvs_to_values[c2]} == set(step.vars)
