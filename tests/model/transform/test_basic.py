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

from pymc.model.transform.basic import (
    model_to_minibatch,
    prune_vars_detached_from_observed,
    remove_minibatched_nodes,
)
from pymc.testing import assert_equivalent_models


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


def test_model_to_minibatch():
    data_size = 100
    n_features = 4

    obs_data_np = np.zeros((data_size,))
    X_data_np = np.random.normal(size=(data_size, n_features))

    with pm.Model(coords={"feature": range(n_features), "data_dim": range(data_size)}) as m1:
        obs_data = pm.Data("obs_data", obs_data_np, dims=["data_dim"])
        X_data = pm.Data("X_data", X_data_np, dims=["data_dim", "feature"])
        beta = pm.Normal("beta", dims="feature")

        mu = X_data @ beta

        y = pm.Normal("y", mu=mu, sigma=1, observed=obs_data, dims="data_dim")

    with pm.Model(
        coords={"feature": range(n_features), "data_dim": range(data_size)}
    ) as reference_model:
        obs_data = pm.Data("obs_data", obs_data_np, dims=["data_dim"])
        X_data = pm.Data("X_data", X_data_np, dims=["data_dim", "feature"])
        minibatch_obs_data, minibatch_X_data = pm.Minibatch(obs_data, X_data, batch_size=10)
        beta = pm.Normal("beta", dims="feature")
        mu = minibatch_X_data @ beta
        y = pm.Normal(
            "y",
            mu=mu,
            sigma=1,
            observed=minibatch_obs_data,
            dims="data_dim",
            total_size=(obs_data.shape[0], ...),
        )

    m2 = model_to_minibatch(m1, batch_size=10)
    assert_equivalent_models(m2, reference_model)


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
