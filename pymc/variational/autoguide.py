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
from typing import Protocol

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.basic import infer_shape_db
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.rewriting.shape import ShapeFeature

from pymc.distributions import Normal
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.basic import conditional_logp
from pymc.model.core import Deterministic, Model
from pymc.pytensorf import compile, rewrite_pregrad


def vectorize_random_graph(
    graph: Sequence[TensorVariable], batch_draws: TensorLike
) -> tuple[TensorVariable]:
    # Find the root random nodes
    rvs = tuple(
        var
        for var in ancestors(graph)
        if (
            var.owner is not None
            and isinstance(var.owner.op, RandomVariable | SymbolicRandomVariable)
        )
    )
    rvs_set = set(rvs)
    root_rvs = tuple(rv for rv in rvs if not (set(rv.owner.inputs) & rvs_set))

    # Vectorize graph by vectorizing root RVs
    batch_draws = pt.as_tensor(batch_draws, dtype=int)
    vectorized_replacements = {
        root_rv: change_dist_size(root_rv, new_size=batch_draws, expand=True)
        for root_rv in root_rvs
    }
    return vectorize_graph(graph, replace=vectorized_replacements)


@dataclass(frozen=True)
class AutoGuideModel:
    model: Model
    params_init_values: dict[Variable, np.ndarray]

    @property
    def params(self) -> tuple[Variable]:
        return tuple(self.params_init_values.keys())

    def stochastic_logq(self):
        """Returns a graph representing the logp of the guide model, evaluated under draws from its random variables."""
        # This allows arbitrary
        logp_terms = conditional_logp(
            {rv: rv for rv in self.model.deterministics},
            warn_rvs=False,
        )
        return pt.sum([logp_term.sum() for logp_term in logp_terms.values()])


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

    free_rv_shapes = dict(zip(free_rvs, get_symbolic_rv_shapes(free_rvs)))
    params_init_values = {}

    with Model(coords=coords) as guide_model:
        for rv in free_rvs:
            loc = pt.tensor(f"{rv.name}_loc", shape=rv.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=rv.type.shape)
            # TODO: Make these customizable
            params_init_values[loc] = pt.random.uniform(-1, 1, size=free_rv_shapes[rv]).eval()
            params_init_values[scale] = pt.full(free_rv_shapes[rv], 0.1).eval()

            z = Normal(
                f"{rv.name}_z",
                mu=0,
                sigma=1,
                shape=free_rv_shapes[rv],
            )
            Deterministic(
                rv.name,
                loc + pt.softplus(scale) * z,
                dims=model.named_vars_to_dims.get(rv.name, None),
            )

    return AutoGuideModel(guide_model, params_init_values)


def get_logp_logq(model: Model, guide: AutoGuideModel):
    inputs_to_guide_rvs = {
        model_value_var: guide.model[rv.name]
        for rv, model_value_var in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }

    logp = graph_replace(model.logp(), inputs_to_guide_rvs)
    logq = guide.stochastic_logq()

    return logp, logq


def advi_objective(logp: TensorVariable, logq: TensorVariable):
    negative_elbo = logq - logp
    return negative_elbo


class TrainingFn(Protocol):
    def __call__(self, draws: int, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_training_fn(model: Model, guide: AutoGuideModel, **compile_kwargs) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params

    logp, logq = get_logp_logq(model, guide)

    scalar_negative_elbo = advi_objective(logp, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    negative_elbo_grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=params)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_loss_dloss = compile(
        inputs=[draws, *params], outputs=[negative_elbo, *negative_elbo_grads], **compile_kwargs
    )

    return f_loss_dloss
