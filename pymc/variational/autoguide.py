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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.graph.basic import Variable
from pytensor.graph.features import ShapeFeature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors, explicit_graph_inputs
from pytensor.tensor import TensorVariable
from pytensor.tensor.basic import infer_shape_db

from pymc.distributions import MvNormal, Normal
from pymc.math import expand_packed_triangular
from pymc.model.core import Deterministic, Model


@dataclass(frozen=True)
class AutoGuideModel:
    model: Model
    draws: TensorVariable

    @property
    def params(self) -> tuple[Variable]:
        draws = self.draws
        return tuple(
            var for var in explicit_graph_inputs(self.model.named_vars) if var is not draws
        )


def get_symbolic_rv_shapes(
    rvs: Sequence[Variable], raise_if_rvs_in_graph: bool = True
) -> tuple[TensorVariable]:
    # TODO: Move me to pytensorf, this is needed often

    rv_shapes = [rv.shape for rv in rvs]
    shape_fg = FunctionGraph(outputs=rv_shapes, features=[ShapeFeature()], clone=True)
    with config.change_flags(optdb__max_use_ratio=10, cxx=""):
        infer_shape_db.default_query.rewrite(shape_fg)
    rv_shapes = shape_fg.outputs

    if raise_if_rvs_in_graph and (overlap := (set(rvs) & set(ancestors(rv_shapes)))):
        raise ValueError(f"rv_shapes still depend the following rvs {overlap}")

    return tuple(rv_shapes)


def AutoDiagonalNormal(model) -> AutoGuideModel:
    coords = model.coords
    free_rvs = model.free_RVs
    draws = pt.tensor("draws", shape=(), dtype="int64")

    free_rv_shapes = dict(zip(free_rvs, get_symbolic_rv_shapes(free_rvs)))

    with Model(coords=coords) as guide_model:
        for rv in free_rvs:
            loc = pt.tensor(f"{rv.name}_loc", shape=rv.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=rv.type.shape)
            z = Normal(
                f"{rv.name}_z",
                mu=0,
                sigma=1,
                # TODO: Don't require draws in guide_model, automate this as a "vectorize_model"
                shape=(draws, *free_rv_shapes[rv]),
                # TODO: What are we trying to do here with transform
                transform=model.rvs_to_transforms[rv],
            )
            Deterministic(
                rv.name, loc + scale * z, dims=model.named_vars_to_dims.get(rv.name, None)
            )

    return AutoGuideModel(guide_model, draws)


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
    L = expand_packed_triangular(packed_L)

    with Model(coords=coords) as guide_model:
        z = MvNormal("z", mu=np.zeros(total_size), cov=np.eye(total_size), size=(draws, total_size))
        params = pt.concatenate([loc.ravel() for loc in locs]) + L @ z

        cursor = 0

        for rv, size in zip(free_rvs, rv_sizes):
            Deterministic(
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
