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
import copy

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile import SharedVariable
from pytensor.compile.builders import OpFromGraph, construct_nominal_fgraph
from pytensor.graph import Constant, FunctionGraph, Variable
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.scalar import Cast
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.type import TensorType

from pymc.model.core import FrozenModel, Model
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model, model_from_fgraph


def _constant_from_shared(shared: SharedVariable) -> Constant:
    return shared.type.constant_type(type=shared.type, data=shared.get_value(), name=shared.name)


def _extract_initial_values(model: Model) -> dict[str, np.ndarray | Variable | str]:
    """Return the model's non-default initial values, keyed by variable name.

    Symbolic initial values reference variables of the model's graph, which the fgraph
    round-trip clones, so they cannot be transplanted onto the rebuilt model and are
    rejected.
    """
    initial_values = {}
    for rv, initval in model.rvs_to_initial_values.items():
        if initval is None:
            continue
        if isinstance(initval, Variable) and not isinstance(initval, Constant):
            raise NotImplementedError(
                f"{rv.name} has a symbolic initial value, which cannot be transplanted onto "
                "the transformed model. Only None, strategy strings and constant initial "
                "values are supported."
            )
        initial_values[rv.name] = initval
    return initial_values


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

    Notes
    -----
    Constant and strategy-string initial values are preserved on the new model. Symbolic
    initial values (which reference variables of the original graph) are not supported.

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
    # fgraph_from_model does not carry initial values through the round-trip and rejects
    # models that have them. Preserve them here: clear them for the round-trip and transplant
    # them back onto the new model (matched by variable name) below.
    initial_values = _extract_initial_values(model)
    saved_initial_values = dict(model.rvs_to_initial_values)
    try:
        for rv in model.rvs_to_initial_values:
            model.rvs_to_initial_values[rv] = None
        fg, memo = fgraph_from_model(model)
    finally:
        model.rvs_to_initial_values.update(saved_initial_values)

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

    new_model = model_from_fgraph(fg, mutate_fgraph=True)
    for name, initval in initial_values.items():
        new_model.set_initval(new_model[name], initval)
    return new_model


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
    frozen_model = freeze_dims_and_data(model, dims=frozen_dims, data=frozen_data)

    # Retype the rebuilt model in place as a FrozenModel. This is the standard idiom for
    # converting an instance to a sibling class: both are pure-Python subclasses of
    # BaseModel with the same instance layout, so only the method resolution changes
    # (mutators become unavailable, compiled functions become cached).
    frozen_model.__class__ = FrozenModel  # type: ignore[assignment]
    return cast(FrozenModel, frozen_model)


def _cast_root(var: Variable, from_dtype: str, to_dtype: str) -> Variable:
    """Return a `to_dtype` clone of a root variable (constant, shared or input)."""
    if getattr(var.type, "dtype", None) != from_dtype:
        return var
    if isinstance(var, Constant):
        return pt.constant(var.data.astype(to_dtype), name=var.name)
    if isinstance(var, SharedVariable):
        return pytensor.shared(
            var.get_value(borrow=False).astype(to_dtype), name=var.name, shape=var.type.shape
        )
    return var.type.clone(dtype=to_dtype)(name=var.name)


def _restore_static_shape(new: Variable, old: Variable) -> Variable:
    if isinstance(new.type, TensorType) and new.type.shape != old.type.shape:
        new = pt.specify_shape(new, old.type.shape)
        new.name = old.name
    return new


def _cast_graph_floats(
    outputs: Sequence[Variable], from_dtype: str, to_dtype: str
) -> tuple[list[Variable], dict[Variable, Variable]]:
    """Clone the graph of `outputs`, casting every `from_dtype` variable to `to_dtype`.

    Returns the converted outputs and a memo mapping old to new variables.
    """
    memo: dict[Variable, Variable] = {}

    def mapped(var):
        if var not in memo:
            memo[var] = _cast_root(var, from_dtype, to_dtype)
        return memo[var]

    for node in io_toposort([], outputs):
        op, new_inputs = node.op, [mapped(var) for var in node.inputs]
        if (
            isinstance(op, Elemwise)
            and isinstance(op.scalar_op, Cast)
            and op.scalar_op.o_type.dtype == from_dtype
        ):
            # Redirect explicit casts (e.g. `x.astype("float64")`)
            new_outputs = [pt.cast(new_inputs[0], to_dtype)]
        elif isinstance(op, OpFromGraph):
            # Convert the inner graph of e.g. SymbolicRandomVariables recursively.
            # Static shapes frozen in the inner graph cannot be re-inferred from the
            # inner inputs when nodes are rebuilt, so they are restored explicitly.
            inner_outs, inner_memo = _cast_graph_floats(op.inner_outputs, from_dtype, to_dtype)
            inner_outs = [
                _restore_static_shape(new, old)
                for new, old in zip(inner_outs, op.inner_outputs, strict=True)
            ]
            inner_ins = [
                inner_memo.get(i, _cast_root(i, from_dtype, to_dtype)) for i in op.inner_inputs
            ]
            new_op = copy.copy(op)
            new_op.fgraph = construct_nominal_fgraph(inner_ins, inner_outs).freeze()
            new_op.input_types = [i.type for i in inner_ins]
            new_op.output_types = [o.type for o in inner_outs]
            # Drop gradient caches computed for the old inner graph
            new_op._lop_op_cache = {}
            new_op._rop_op_cache = None
            new_op._frozen_lop = None
            new_op._frozen_rop = None
            new_outputs = new_op.make_node(*new_inputs).outputs
        elif getattr(op, "dtype", None) == from_dtype:
            # Ops with a fixed output dtype: RandomVariables, reductions, ARange, ...
            new_op = copy.copy(op)
            new_op.dtype = to_dtype
            new_outputs = new_op.make_node(*new_inputs).outputs
        else:
            new_outputs = op.make_node(*new_inputs).outputs
        for old, new in zip(node.outputs, new_outputs, strict=True):
            new.name = old.name
            memo[old] = new

    return [mapped(out) for out in outputs], memo


def _cast_model_floats(model: Model, from_dtype: str, to_dtype: str) -> Model:
    initial_values = _extract_initial_values(model)
    saved_initial_values = dict(model.rvs_to_initial_values)
    try:
        for rv in model.rvs_to_initial_values:
            model.rvs_to_initial_values[rv] = None
        fg, _ = fgraph_from_model(model)
    finally:
        model.rvs_to_initial_values.update(saved_initial_values)

    new_outputs, memo = _cast_graph_floats(fg.outputs, from_dtype, to_dtype)
    new_fg = FunctionGraph(outputs=new_outputs, clone=False)
    new_fg._coords = fg._coords  # type: ignore[attr-defined]
    new_fg._dim_lengths = {  # type: ignore[attr-defined]
        dim: memo.get(length, length)
        for dim, length in fg._dim_lengths.items()  # type: ignore[attr-defined]
    }

    new_model = model_from_fgraph(new_fg, mutate_fgraph=True)
    for name, initval in initial_values.items():
        if isinstance(initval, np.ndarray) and initval.dtype.kind == "f":
            initval = initval.astype(to_dtype)
        new_model.set_initval(new_model[name], initval)
    return new_model


def model_to_float32(model: Model) -> Model:
    """Recreate a Model with all float64 variables and data cast to float32.

    Every float64 variable is converted: data (constants and `pm.Data`), free and
    observed RVs (including the inner graphs of symbolic RVs like `ZeroSumNormal`),
    value variables, Deterministics and Potentials. Integer, boolean and RNG
    variables are unaffected. Explicit `.astype("float64")` casts are redirected
    to float32.

    This can substantially speed up sampling on CPUs (via SIMD vectorization and
    halved memory traffic) and especially on GPUs, at the cost of precision.

    Compile and sample under ``floatX="float32"``, otherwise constants introduced
    when building logp graphs will upcast intermediate computations back to float64:

    .. code-block:: python

        import pymc as pm
        import pytensor

        from pymc.model.transform.optimization import model_to_float32

        with pm.Model() as m:
            x = pm.Data("x", [0.0, 1.0, 2.0])
            beta = pm.Normal("beta")
            pm.Normal("y", mu=beta * x, sigma=1.0, observed=[1.0, 2.0, 3.0])

        with pytensor.config.change_flags(floatX="float32"):
            with model_to_float32(m):
                idata = pm.sample()

    Notes
    -----
    ``pm.set_data`` on the new model expects float32 arrays.

    Constant and strategy-string initial values are preserved (arrays are cast);
    symbolic initial values are not supported.
    """
    return _cast_model_floats(model, "float64", "float32")


def model_to_float64(model: Model) -> Model:
    """Recreate a Model with all float32 variables and data cast to float64.

    The inverse of :func:`model_to_float32`. See its docstring for details.
    """
    return _cast_model_floats(model, "float32", "float64")


__all__ = ("freeze_dims_and_data", "freeze_model", "model_to_float32", "model_to_float64")
