#   Copyright 2024 The PyMC Developers
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
from pytensor import clone_replace
from pytensor.compile import SharedVariable
from pytensor.graph import FunctionGraph
from pytensor.tensor import constant

from pymc import Model
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model, model_from_fgraph


def freeze_dims_and_data(model: Model) -> Model:
    """Recreate a Model with fixed RV dimensions and Data values.

    The dimensions of the pre-existing RVs will no longer follow changes to the coordinates.
    Likewise, it will not be possible to update pre-existing Data in the new model.

    Note that any new RVs and Data created after calling this function will still be "unfrozen".

    This transformation may allow more performant sampling, or compiling model functions to backends that
    are more restrictive about dynamic shapes such as JAX.
    """
    fg, memo = fgraph_from_model(model)

    # Replace mutable dim lengths and data by constants
    frozen_vars = {
        memo[dim_length]: constant(
            dim_length.get_value(), name=dim_length.name, dtype=dim_length.type.dtype
        )
        for dim_length in model.dim_lengths.values()
        if isinstance(dim_length, SharedVariable)
    }
    frozen_vars |= {
        memo[data_var].owner.inputs[0]: constant(
            data_var.get_value(), name=data_var.name, dtype=data_var.type.dtype
        )
        for data_var in model.named_vars.values()
        if isinstance(data_var, SharedVariable)
    }

    old_outs, coords = fg.outputs, fg._coords  # type: ignore
    # Rebuild strict will force the recreation of RV nodes with updated static types
    new_outs = clone_replace(old_outs, replace=frozen_vars, rebuild_strict=False)  # type: ignore
    for old_out, new_out in zip(old_outs, new_outs):
        new_out.name = old_out.name
    fg = FunctionGraph(outputs=new_outs, clone=False)
    fg._coords = coords  # type: ignore

    # Recreate value variables from new RVs to propagate static types to logp graphs
    replacements = {}
    for node in fg.apply_nodes:
        if not isinstance(node.op, ModelFreeRV):
            continue
        rv, old_value, *dims = node.inputs
        if dims is None:
            continue
        transform = node.op.transform
        if transform is None:
            new_value = rv.type()
        else:
            new_value = transform.forward(rv, *rv.owner.inputs).type()  # type: ignore
        new_value.name = old_value.name
        replacements[old_value] = new_value
    fg.replace_all(tuple(replacements.items()), import_missing=True)

    return model_from_fgraph(fg)


__all__ = ("freeze_dims_and_data",)
