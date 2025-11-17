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

from pymc.data import Minibatch, MinibatchOp
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelObservedRV,
    ModelVar,
    fgraph_from_model,
    model_from_fgraph,
)
from pymc.pytensorf import toposort_replace

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


def model_to_minibatch(
    model: Model, *, batch_size: int, minibatch_vars: list[str] | None = None
) -> Model:
    """Replace all Data containers with pm.Minibatch, and add total_size to all observed RVs."""
    from pymc.variational.minibatch_rv import create_minibatch_rv

    if minibatch_vars is None:
        original_minibatch_vars = [
            variable
            for variable in model.data_vars
            if (variable.type.ndim > 0) and (variable.type.shape[0] is None)
        ]
    else:
        original_minibatch_vars = parse_vars(model, minibatch_vars)
        for variable in original_minibatch_vars:
            if variable.type.ndim == 0:
                raise ValueError(
                    f"Cannot minibatch {variable.name} because it is a scalar variable."
                )
            if variable.type.shape[0] is not None:
                raise ValueError(
                    f"Cannot minibatch {variable.name} because its first dimension is static "
                    f"(size={variable.type.shape[0]})."
                )

    # TODO: Validate that this graph is actually valid to minibatch. Example: linear regression with sigma fixed
    #  shape, but mu from data --> y cannot be minibatched because of sigma.

    fgraph, memo = fgraph_from_model(model, inlined_views=True)

    pre_minibatch_vars = [memo[var] for var in original_minibatch_vars]
    minibatch_vars = Minibatch(*pre_minibatch_vars, batch_size=batch_size)

    # Replace uses of the specified data variables with Minibatch variables
    # We need a two-step clone because FunctionGraph can only mutate one variable at a time
    # and when there are multiple vars to minibatch you end up replacing the same variable twice recursively
    # exampre: out = x + y
    # goal: replace (x, y) by (Minibatch(x, y).0, Minibatch(x, y).1)]
    # replace x first we get: out = Minibatch(x, y).0 + y
    # then replace y we get: out = Minibatch(x, Minibatch(...).1).0 + Minibatch(x, y).1
    # The second replacement of y ends up creating a circular dependency
    pre_minibatch_var_to_dummy = tuple((var, var.type()) for var in pre_minibatch_vars)
    dummy_to_minibatch_var = tuple(
        (dummy, minibatch_var)
        for (_, dummy), minibatch_var in zip(pre_minibatch_var_to_dummy, minibatch_vars)
    )

    # Furthermore, we only want to replace uses of the data variables (x, y), but not the data variables themselves,
    # So we use an intermediate FunctionGraph that doesn't contain the data variables as outputs
    other_model_vars = [out for out in fgraph.outputs if out not in pre_minibatch_vars]
    minibatch_fgraph = FunctionGraph(outputs=other_model_vars, clone=False)
    minibatch_fgraph._coords = fgraph._coords  # type: ignore[attr-defined]
    minibatch_fgraph._dim_lengths = fgraph._dim_lengths  # type: ignore[attr-defined]
    toposort_replace(minibatch_fgraph, pre_minibatch_var_to_dummy)
    toposort_replace(minibatch_fgraph, dummy_to_minibatch_var)

    # Then replace all observed RVs that depend on the minibatch variables with MinibatchRVs
    dependent_replacements = {}
    total_size = (pre_minibatch_vars[0].owner.inputs[0].shape[0], ...)
    vars_to_minibatch_set = set(pre_minibatch_vars)
    for model_var in minibatch_fgraph.outputs:
        if not (set(ancestors([model_var])) & vars_to_minibatch_set):
            continue
        if not isinstance(model_var.owner.op, ModelObservedRV):
            raise ValueError(
                "Minibatching only supports observed RVs depending on minibatched variables. "
                f"Found dependent unobserved variable: {model_var.name}."
            )
        # TODO: If vars_to_minibatch had a leading dim, we should check that the dependent RVs also has that same dim
        # And conversely other variables do not have that dim
        observed_rv = model_var.owner.inputs[0]
        dependent_replacements[observed_rv] = create_minibatch_rv(
            observed_rv, total_size=total_size
        )

    toposort_replace(minibatch_fgraph, tuple(dependent_replacements.items()))

    # Finally reintroduce the original data variable outputs
    for pre_minibatch_var in pre_minibatch_vars:
        minibatch_fgraph.add_output(pre_minibatch_var)

    return model_from_fgraph(minibatch_fgraph, mutate_fgraph=True)


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
