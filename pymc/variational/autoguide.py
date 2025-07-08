#   Copyright 2025 - present The PyMC Developers
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
import pytensor.tensor as pt

from pytensor import Variable, graph_replace
from pytensor.graph import vectorize_graph

import pymc as pm

from pymc.model.core import Model

ModelVariable = Variable | str


def AutoDiagonalNormal(model):
    coords = model.coords
    free_rvs = model.free_RVs
    draws = pt.tensor("draws", shape=(), dtype="int64")

    with Model(coords=coords) as guide_model:
        for rv in free_rvs:
            loc = pt.tensor(f"{rv.name}_loc", shape=rv.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=rv.type.shape)
            z = pm.Normal(
                f"{rv.name}_z",
                mu=0,
                sigma=1,
                shape=(draws, *rv.type.shape),
                transform=model.rvs_to_transforms[rv],
            )
            pm.Deterministic(
                rv.name, loc + scale * z, dims=model.named_vars_to_dims.get(rv.name, None)
            )

    return guide_model


def AutoFullRankNormal(model):
    # TODO: Broken

    coords = model.coords
    free_rvs = model.free_RVs
    draws = pt.tensor("draws", shape=(), dtype="int64")

    rv_sizes = [np.prod(rv.type.shape) for rv in free_rvs]
    total_size = np.sum(rv_sizes)
    tril_size = total_size * (total_size + 1) // 2

    locs = [pt.tensor(f"{rv.name}_loc", shape=rv.type.shape) for rv in free_rvs]
    packed_L = pt.tensor("L", shape=(tril_size,), dtype="float64")
    L = pm.expand_packed_triangular(packed_L)

    with Model(coords=coords) as guide_model:
        z = pm.MvNormal(
            "z", mu=np.zeros(total_size), cov=np.eye(total_size), size=(draws, total_size)
        )
        params = pt.concatenate([loc.ravel() for loc in locs]) + L @ z

        cursor = 0

        for rv, size in zip(free_rvs, rv_sizes):
            pm.Deterministic(
                rv.name,
                params[cursor : cursor + size].reshape(rv.type.shape),
                dims=model.named_vars_to_dims.get(rv.name, None),
            )
            cursor += size

    return guide_model


def get_logp_logq(model, guide_model):
    inputs_to_guide_rvs = {
        model_value_var: guide_model[rv.name]
        for rv, model_value_var in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }

    logp = vectorize_graph(model.logp(), inputs_to_guide_rvs)
    logq = graph_replace(guide_model.logp(), guide_model.values_to_rvs)

    return logp, logq
