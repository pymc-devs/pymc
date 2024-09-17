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
from collections.abc import Sequence

from pytensor import clone_replace
from pytensor.compile import SharedVariable
from pytensor.graph import FunctionGraph
from pytensor.tensor import constant
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.variable import TensorConstant

from pymc import Model
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model, model_from_fgraph


def _constant_from_shared(shared: SharedVariable) -> TensorConstant:
    assert isinstance(shared, TensorSharedVariable)
    return constant(shared.get_value(), name=shared.name, dtype=shared.type.dtype)


def freeze_dims_and_data(
    model: Model, dims: Sequence[str] | None = None, data: Sequence[str] | None = None
) -> Model:
    """Recreate a Model with fixed RV dimensions and Data values.

    The dimensions of the pre-existing RVs will no longer follow changes to the coordinates.
    Likewise, it will not be possible to update pre-existing Data in the new model.

    Note that any new RVs and Data created after calling this function will still be "unfrozen".

    This transformation may allow more performant sampling, or compiling model functions to backends that
    are more restrictive about dynamic shapes such as JAX.

    Parameters
    ----------
    model : Model
        The model where to freeze dims and data.
    dims : Sequence of str, optional
        The dimensions to freeze.
        If None, all dimensions are frozen. Pass an empty list to avoid freezing any dimension.
    data : Sequence of str, optional
        The data to freeze.
        If None, all data are frozen. Pass an empty list to avoid freezing any data.

    Returns
    -------
    Model
        A new model with the specified dimensions and data frozen.
    """
    fg, memo = fgraph_from_model(model)

    if dims is None:
        dims = tuple(model.dim_lengths.keys())
    if data is None:
        data = tuple(model.named_vars.keys())

    # Replace mutable dim lengths and data by constants
    frozen_replacements = {
        memo[dim_length]: _constant_from_shared(dim_length)
        for dim_length in (model.dim_lengths[dim_name] for dim_name in dims)
        if isinstance(dim_length, SharedVariable)
    }
    frozen_replacements |= {
        memo[datum].owner.inputs[0]: _constant_from_shared(datum)
        for datum in (model.named_vars[datum_name] for datum_name in data)
        if isinstance(datum, SharedVariable)
    }

    old_outs, old_coords, old_dim_lenghts = fg.outputs, fg._coords, fg._dim_lengths  # type: ignore[attr-defined]
    # Rebuild strict will force the recreation of RV nodes with updated static types
    new_outs = clone_replace(old_outs, replace=frozen_replacements, rebuild_strict=False)  # type: ignore[arg-type]
    for old_out, new_out in zip(old_outs, new_outs):
        new_out.name = old_out.name
    fg = FunctionGraph(outputs=new_outs, clone=False)
    fg._coords = old_coords  # type: ignore[attr-defined]
    fg._dim_lengths = {  # type: ignore[attr-defined]
        dim: frozen_replacements.get(dim_length, dim_length)
        for dim, dim_length in old_dim_lenghts.items()
    }

    # Recreate value variables from new RVs to propagate static types to logp graphs
    replacements = {}
    for node in fg.apply_nodes:
        if not isinstance(node.op, ModelFreeRV):
            continue
        rv, old_value, *_ = node.inputs
        transform = node.op.transform
        if transform is None:
            new_value = rv.type()
        else:
            new_value = transform.forward(rv, *rv.owner.inputs).type()  # type: ignore[arg-type]
        new_value.name = old_value.name
        replacements[old_value] = new_value
    fg.replace_all(tuple(replacements.items()), import_missing=True)

    return model_from_fgraph(fg, mutate_fgraph=True)


__all__ = ("freeze_dims_and_data",)
