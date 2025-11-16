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
    model: Model, *, batch_size: int, vars_to_minibatch: list[str] | None = None
) -> Model:
    """Replace all Data containers with pm.Minibatch, and add total_size to all observed RVs."""
    from pymc.variational.minibatch_rv import create_minibatch_rv

    if vars_to_minibatch is None:
        vars_to_minibatch = [
            variable
            for variable in model.data_vars
            if (variable.type.ndim > 0) and (variable.type.shape[0] is None)
        ]
    else:
        vars_to_minibatch = parse_vars(model, vars_to_minibatch)
        for variable in vars_to_minibatch:
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

    cloned_vars_to_minibatch = [memo[var] for var in vars_to_minibatch]
    minibatch_vars = Minibatch(*cloned_vars_to_minibatch, batch_size=batch_size)

    var_to_dummy = {
        var: var.type()  # model_named(minibatch_var, *extract_dims(var))
        for var, minibatch_var in zip(cloned_vars_to_minibatch, minibatch_vars)
    }
    dummy_to_minibatch = {
        var_to_dummy[var]: minibatch_var
        for var, minibatch_var in zip(cloned_vars_to_minibatch, minibatch_vars)
    }
    total_size = (cloned_vars_to_minibatch[0].owner.inputs[0].shape[0], ...)

    # TODO: If vars_to_minibatch had a leading dim, we should check that the dependent RVs also has that same dim
    #  (or just do this all in xtensor)
    vtm_set = set(vars_to_minibatch)

    # TODO: Handle potentials, free_RVs, etc

    # Create a temporary fgraph that does not include as outputs any of the variables that will be minibatched. This
    # ensures the results of this function match the outputs from a model constructed using the pm.Minibatch API.
    tmp_fgraph = FunctionGraph(
        outputs=[out for out in fgraph.outputs if out not in var_to_dummy.keys()], clone=False
    )

    # All variables that will be minibatched are first replaced by dummy variables, to avoid infinite recursion during
    # rewrites. The issue is that the Minibatch Op we will introduce depends on the original input variables (to get
    # the shapes). That's fine in the final output, but during the intermediate rewrites this creates a circulatr
    # dependency.
    dummy_replacements = tuple(var_to_dummy.items())
    toposort_replace(tmp_fgraph, dummy_replacements)

    # Now we can replace the dummy variables with the actual Minibatch variables.
    replacements = tuple(dummy_to_minibatch.items())
    toposort_replace(tmp_fgraph, replacements)

    # The last step is to replace all RVs that depend on the minibatched variables with MinibatchRVs that are aware
    # of the total_size.  Importantly, all of the toposort_replace calls above modify fgraph in place, so the
    # model.rvs_to_values[original_rv] will already have been modified to depend on the Minibatch variables -- only
    # the outer RVs need to be replaced here.
    dependent_replacements = {}

    for original_rv in model.observed_RVs:
        original_value_var = model.rvs_to_values[original_rv]

        if not (set(ancestors([original_rv, original_value_var])) & vtm_set):
            continue

        rv = memo[original_rv].owner.inputs[0]
        dependent_replacements[rv] = create_minibatch_rv(rv, total_size=total_size)

    toposort_replace(fgraph, tuple(dependent_replacements.items()))

    # FIXME: The fgraph is being rebuilt here to clean up the clients. It is not clear why they are getting messed up
    #  in the first place (pytensor bug, or something wrong in the above manipulations?)
    new_fgraph = FunctionGraph(outputs=fgraph.outputs)
    new_fgraph._coords = fgraph._coords  # type: ignore[attr-defined]
    new_fgraph._dim_lengths = fgraph._dim_lengths  # type: ignore[attr-defined]

    return model_from_fgraph(new_fgraph, mutate_fgraph=True)


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
