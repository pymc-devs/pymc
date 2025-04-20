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
import numpy as np

import pymc as pm

from pymc.model.transform.basic import prune_vars_detached_from_observed, remove_minibatched_nodes


def test_prune_vars_detached_from_observed():
    with pm.Model() as m:
        obs_data = pm.Data("obs_data", 0)
        a0 = pm.Data("a0", 0)
        a1 = pm.Normal("a1", a0)
        a2 = pm.Normal("a2", a1)
        pm.Normal("obs", a2, observed=obs_data)

        d0 = pm.Data("d0", 0)
        d1 = pm.Normal("d1", d0)

    assert set(m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs", "d0", "d1"}
    pruned_m = prune_vars_detached_from_observed(m)
    assert set(pruned_m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs"}


def test_remove_minibatches():
    data_size = 100
    data = np.zeros((data_size,))
    batch_size = 10
    with pm.Model(coords={"d": range(5)}) as m1:
        mb = pm.Minibatch(data, batch_size=batch_size)
        mu = pm.Normal("mu", dims="d")
        x = pm.Normal("x")
        y = pm.Normal("y", x, observed=mb, total_size=100)

    m2 = remove_minibatched_nodes(m1)
    assert m1.y.shape[0].eval() == batch_size
    assert m2.y.shape[0].eval() == data_size
    assert m1.coords == m2.coords
    assert m1.dim_lengths["d"].eval() == m2.dim_lengths["d"].eval()
