#   Copyright 2023 The PyMC Developers
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

from pymc.step_methods.slicer import Slice
from pymc.tests import sampler_fixtures as sf
from pymc.tests.helpers import RVsAssignmentStepsTester, StepMethodTester


class TestSliceUniform(sf.SliceFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 4
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05


class TestStepSlicer(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (lambda *_: Slice(), 2000),
            (lambda *_: Slice(blocked=True), 2000),
        ],
        ids=str,
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentSlicer(RVsAssignmentStepsTester):
    @pytest.mark.parametrize("step, step_kwargs", [(Slice, {})])
    def test_continuous_steps(self, step, step_kwargs):
        self.continuous_steps(step, step_kwargs)
