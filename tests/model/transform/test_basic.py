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
from pymc import Data, Model
from pymc.distributions import Normal
from pymc.model.transform.basic import (
    prune_vars_detached_from_observed,
)


def test_prune_vars_detached_from_observed():
    with Model() as m:
        obs_data = Data("obs_data", 0)
        a0 = Data("a0", 0)
        a1 = Normal("a1", a0)
        a2 = Normal("a2", a1)
        Normal("obs", a2, observed=obs_data)

        d0 = Data("d0", 0)
        d1 = Normal("d1", d0)

    assert set(m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs", "d0", "d1"}
    pruned_m = prune_vars_detached_from_observed(m)
    assert set(pruned_m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs"}
