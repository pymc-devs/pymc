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

import pytest

from pymc.step_methods.hmc import HamiltonianMC
from pymc.tests.helpers import RVsAssignmentStepsTester, StepMethodTester


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
