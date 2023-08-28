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
from typing import List, Sequence, Union

from pytensor import Variable
from pytensor.graph import ancestors

from pymc import Model
from pymc.model.fgraph import (
    ModelObservedRV,
    ModelVar,
    fgraph_from_model,
    model_from_fgraph,
)

ModelVariable = Union[Variable, str]


def prune_vars_detached_from_observed(model: Model) -> Model:
    """Prune model variables that are not related to any observed variable in the Model."""

    # Potentials are ambiguous as whether they correspond to likelihood or prior terms,
    # We simply raise for now
    if model.potentials:
        raise NotImplementedError("Pruning not implemented for models with Potentials")

    fgraph, _ = fgraph_from_model(model, inlined_views=True)
    observed_vars = (
        out
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelObservedRV)
        for out in node.outputs
    )
    ancestor_nodes = {var.owner for var in ancestors(observed_vars)}
    nodes_to_remove = {
        node
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelVar) and node not in ancestor_nodes
    }
    for node_to_remove in nodes_to_remove:
        fgraph.remove_node(node_to_remove)
    return model_from_fgraph(fgraph)


def parse_vars(model: Model, vars: Union[ModelVariable, Sequence[ModelVariable]]) -> List[Variable]:
    if not isinstance(vars, (list, tuple)):
        vars = (vars,)
    return [model[var] if isinstance(var, str) else var for var in vars]
