#   Copyright 2024 - present The PyMC Developers
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

import pymc as pm


@pytest.mark.parametrize(
    "distribution, lower, upper, mass, fixed_params, fixed_stats",
    [
        (pm.Gamma, 0.1, 0.4, None, {}, None),
        (pm.StudentT, 0.1, 0.4, 0.7, {"nu": 7}, None),
        (pm.Beta, 0, 0.7, 0.9, {}, ("mode", 0.25)),
    ],
)
def test_find_constrained_prior(
    distribution,
    lower,
    upper,
    mass,
    fixed_params,
    fixed_stats,
):
    pm.find_constrained_prior(
        distribution,
        lower=lower,
        upper=upper,
        mass=mass,
        fixed_params=fixed_params,
        fixed_stat=fixed_stats,
    )
