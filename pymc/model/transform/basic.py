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
from collections.abc import Sequence

from pytensor import Variable, clone_replace
from pytensor.graph import ancestors
from pytensor.graph.fg import FunctionGraph

from pymc.data import MinibatchOp
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelObservedRV,
    ModelVar,
    fgraph_from_model,
    model_from_fgraph,
)

ModelVariable = Variable | str


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
    return model_from_fgraph(fgraph, mutate_fgraph=True)


def parse_vars(model: Model, vars: ModelVariable | Sequence[ModelVariable]) -> list[Variable]:
    if isinstance(vars, list | tuple):
        vars_seq = vars
    else:
        vars_seq = (vars,)
    return [model[var] if isinstance(var, str) else var for var in vars_seq]


def remove_minibatched_nodes(model: Model) -> Model:
    """Remove all uses of pm.Minibatch in the Model."""
    fgraph, _ = fgraph_from_model(model)

    replacements = {}
    for var in fgraph.apply_nodes:
        if isinstance(var.op, MinibatchOp):
            for inp, out in zip(var.inputs, var.outputs):
                replacements[out] = inp

    old_outs, old_coords, old_dim_lengths = fgraph.outputs, fgraph._coords, fgraph._dim_lengths  # type: ignore[attr-defined]
    # Using `rebuild_strict=False` means all coords, names, and dim information is lost
    # So we need to restore it from the old fgraph
    new_outs = clone_replace(old_outs, replacements, rebuild_strict=False)  # type: ignore[arg-type]
    for old_out, new_out in zip(old_outs, new_outs):
        new_out.name = old_out.name
    fgraph = FunctionGraph(outputs=new_outs, clone=False)
    fgraph._coords = old_coords  # type: ignore[attr-defined]
    fgraph._dim_lengths = old_dim_lengths  # type: ignore[attr-defined]
    return model_from_fgraph(fgraph, mutate_fgraph=True)
