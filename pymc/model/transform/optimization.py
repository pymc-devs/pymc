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
from typing import cast

from pytensor.compile import SharedVariable
from pytensor.graph import Constant, FunctionGraph, Variable
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import ancestors

from pymc.model.core import FrozenModel, Model
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model, model_from_fgraph


def _constant_from_shared(shared: SharedVariable) -> Constant:
    return shared.type.constant_type(type=shared.type, data=shared.get_value(), name=shared.name)


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


    Examples
    --------
    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        from pymc.model.transform import freeze_dims_and_data

        with pm.Model() as m:
            x = pm.Data("x", [0, 1, 2] * 1000)
            y = pm.Normal("y", mu=pt.unique(x).mean())

        # pt.unique(x).mean() has to be computed in every logp function evaluation
        print("Logp eval time (1000x): ", m.profile(m.logp()).fct_call_time)

        # pt.uniqe(x).mean() is cached in the logp function
        frozen_m = freeze_dims_and_data(m)
        print("Logp eval time (1000x): ", frozen_m.profile(frozen_m.logp()).fct_call_time)

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


def freeze_model(model: Model) -> FrozenModel:
    """Return a frozen copy of the model that caches its compiled functions.

    On the frozen model, compiled functions (``compile_fn``, ``logp_dlogp_function``,
    ``initial_point``, and the forward-sampling function used by
    ``sample_prior_predictive`` / ``sample_posterior_predictive``) are compiled once and
    reused across calls, so e.g. batched posterior predictive over changing ``pm.set_data``
    values, or repeated ``pm.sample``, do not recompile. Seeding is re-applied on every
    call, so cached functions stay reproducible.

    To keep the cache valid the frozen model cannot be mutated: graph-mutating methods
    (``register_rv``, ``add_coord``, ``set_initval``, ...) raise, and the dims and data
    that any free variable depends on are frozen to constants as in
    :func:`freeze_dims_and_data`. Data (and dims) that only Deterministics and observed
    variables depend on remain updatable through ``pm.set_data`` — values and shapes are
    runtime inputs of the cached functions, so updates and resizes take effect without
    recompilation.

    Functions with random variables compiled to backends that detach their RNGs at compile
    time (JAX, MLX, PyTorch) cannot be reseeded and are compiled fresh on each call.

    Constant and strategy-string initial values are preserved on the frozen model;
    symbolic initial values are not supported.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc.model.transform.optimization import freeze_model

        with pm.Model() as m:
            x = pm.Data("x", [0.0, 1.0, 2.0])
            beta = pm.Normal("beta")
            pm.Normal("y", mu=beta * x, observed=[1.0, 2.0, 3.0], shape=x.shape)
            idata = pm.sample()

        with freeze_model(m):
            for x_batch in x_batches:
                pm.set_data({"x": x_batch})
                # Compiles on the first call only
                pm.sample_posterior_predictive(idata, predictions=True)
    """
    free_rv_ancestors = set(ancestors(model.free_RVs))
    frozen_dims = [
        name
        for name, length in model.dim_lengths.items()
        if isinstance(length, SharedVariable) and length in free_rv_ancestors
    ]
    frozen_data = [
        name
        for name, var in model.named_vars.items()
        if isinstance(var, SharedVariable) and var in free_rv_ancestors
    ]

    # freeze_dims_and_data (via fgraph_from_model) does not carry initial values through the
    # fgraph round-trip and rejects models that have them. Non-symbolic initial values are
    # model metadata, not graph structure, so clear them for the round-trip and transplant
    # them onto the frozen model afterwards, matched by variable name. Symbolic initial
    # values reference variables of the original graph and cannot be transplanted.
    initial_values = {}
    for rv, initval in model.rvs_to_initial_values.items():
        if initval is None:
            continue
        if isinstance(initval, Variable) and not isinstance(initval, Constant):
            raise NotImplementedError(
                f"Cannot freeze model: {rv.name} has a symbolic initial value. "
                "Only None, strategy strings and constant initial values are supported."
            )
        initial_values[rv.name] = initval
    saved_initial_values = dict(model.rvs_to_initial_values)
    try:
        for rv in model.rvs_to_initial_values:
            model.rvs_to_initial_values[rv] = None
        frozen_model = freeze_dims_and_data(model, dims=frozen_dims, data=frozen_data)
    finally:
        model.rvs_to_initial_values.update(saved_initial_values)

    for name, initval in initial_values.items():
        frozen_model.set_initval(frozen_model[name], initval)

    frozen_model.__class__ = FrozenModel  # type: ignore[assignment]
    return cast(FrozenModel, frozen_model)


__all__ = ("freeze_dims_and_data", "freeze_model")
