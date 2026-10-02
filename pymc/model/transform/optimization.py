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
from pytensor.compile.builders import construct_nominal_fgraph
from pytensor.graph import Constant, FunctionGraph, Op, Variable
from pytensor.graph.op import HasInnerGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.scalar import Cast, discrete_dtypes
from pytensor.scan.op import Scan
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.type import TensorType

from pymc.logprob.transforms import Transform
from pymc.model.core import FrozenModel, Model
from pymc.model.fgraph import (
    ModelFreeRV,
    ModelValuedVar,
    fgraph_from_model,
    model_from_fgraph,
)


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


def _fgraph_and_initial_values(model: Model):
    """Return `fgraph_from_model(model)` and the initial values it drops (and rejects)."""
    initial_values = _extract_initial_values(model)
    saved_initial_values = dict(model.rvs_to_initial_values)
    try:
        for rv in model.rvs_to_initial_values:
            model.rvs_to_initial_values[rv] = None
        fg, memo = fgraph_from_model(model)
    finally:
        model.rvs_to_initial_values.update(saved_initial_values)
    return fg, memo, initial_values


def _model_fgraph_like(
    fg: FunctionGraph, outputs: Sequence[Variable], replacements: dict
) -> FunctionGraph:
    """Build a model fgraph from `outputs`, with the coords and (replaced) dim lengths of `fg`."""
    new_fg = FunctionGraph(outputs=outputs, clone=False)
    new_fg._coords = fg._coords  # type: ignore[attr-defined]
    new_fg._dim_lengths = {  # type: ignore[attr-defined]
        dim: replacements.get(dim_length, dim_length)
        for dim, dim_length in fg._dim_lengths.items()  # type: ignore[attr-defined]
    }
    return new_fg


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
    fg, memo, initial_values = _fgraph_and_initial_values(model)

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

    # Rebuild strict will force the recreation of RV nodes with updated static types
    new_outs = clone_replace(fg.outputs, replace=frozen_replacements, rebuild_strict=False)  # type: ignore[arg-type]
    fg = _model_fgraph_like(fg, new_outs, frozen_replacements)

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


def _is_dtype(dtype, ref_dtype: str) -> bool:
    """Whether `dtype` (a dtype-like or alias such as "float" or "floatX") is `ref_dtype`."""
    if dtype == "floatX":
        dtype = pytensor.config.floatX
    try:
        return dtype is not None and np.dtype(dtype).name == ref_dtype
    except TypeError:
        return False


def _cast_root(var: Variable, from_dtype: str, to_dtype: str) -> Variable:
    """Return a `to_dtype` clone of a root variable (constant, shared or input)."""
    if getattr(var.type, "dtype", None) != from_dtype:
        return var
    new_type = var.type.clone(dtype=to_dtype)
    if isinstance(var, Constant):
        return new_type.make_constant(var.data.astype(to_dtype), name=var.name)
    if isinstance(var, SharedVariable):
        value = var.get_value(borrow=False).astype(to_dtype)
        return type(var)(type=new_type, value=value, strict=False, name=var.name)
    return new_type(name=var.name)


def _restore_static_shape(new: Variable, old: Variable) -> Variable:
    if isinstance(new.type, TensorType) and new.type.shape != old.type.shape:
        new = pt.specify_shape(new, old.type.shape)
        new.name = old.name
    return new


def _cast_op(op: Op, from_dtype: str, to_dtype: str) -> Op:
    """Return `op` with the `from_dtype` baked into its attributes or inner graph cast."""
    # Fixed output dtypes (RandomVariables, reductions, ARange, ...), constants and core ops
    changes: dict = {
        attr: to_dtype
        for attr in ("dtype", "acc_dtype")
        if _is_dtype(getattr(op, attr, None), from_dtype)
    }
    for attr, value in vars(op).items():
        new_value: Variable | Op
        if isinstance(value, Constant):
            new_value = _cast_root(value, from_dtype, to_dtype)
        elif attr == "core_op" and isinstance(value, Op):
            new_value = _cast_op(value, from_dtype, to_dtype)
        else:
            continue
        if new_value is not value:
            changes[attr] = new_value

    if isinstance(op, HasInnerGraph) and any(
        getattr(var.type, "dtype", None) == from_dtype for var in op.fgraph.variables
    ):
        inner_outs, memo = _cast_graph_floats(op.inner_outputs, from_dtype, to_dtype)
        # Static shapes frozen in the inner graph cannot be re-inferred from the inner inputs
        inner_outs = [
            _restore_static_shape(new, old)
            for new, old in zip(inner_outs, op.inner_outputs, strict=True)
        ]
        inner_ins = [memo.get(i, _cast_root(i, from_dtype, to_dtype)) for i in op.inner_inputs]
        if isinstance(op, Scan):
            kwargs = ("mode", "truncate_gradient", "name", "profile", "allow_gc", "strict")
            op = Scan(inner_ins, inner_outs, op.info, **{k: getattr(op, k) for k in kwargs})
        else:
            op = op.clone_with_inner_graph(construct_nominal_fgraph(inner_ins, inner_outs))  # type: ignore[attr-defined]
    elif changes:
        op = copy.copy(op)
    vars(op).update(changes)
    return op


class _CastedTransform(Transform):
    """Wrap a transform whose graphs embed constants of a different float dtype.

    Such constants are not reachable from the model graph, so the graphs the wrapped
    transform builds are converted with `_cast_graph_floats` on every call.
    """

    def __init__(self, transform: Transform, from_dtype: str, to_dtype: str):
        self.transform = transform
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        # Keep the name: value variable names derive from it
        self.name = transform.name  # type: ignore[attr-defined]

    def _converted(self, method: str, value, *inputs):
        out = getattr(self.transform, method)(value, *inputs)
        (out,), _ = _cast_graph_floats([out], self.from_dtype, self.to_dtype, (value, *inputs))
        return out

    def forward(self, value, *inputs):
        return self._converted("forward", value, *inputs)

    def backward(self, value, *inputs):
        return self._converted("backward", value, *inputs)

    def log_jac_det(self, value, *inputs):
        return self._converted("log_jac_det", value, *inputs)


def _transform_keeps_dtype(transform: Transform, rv: Variable, value: Variable, dtype: str) -> bool:
    """Whether the transform's graphs on `rv`/`value` stay in `dtype`.

    Probed under ``floatX=dtype``, the setting the converted model is meant to be
    compiled under, so only transforms that embed foreign-dtype constants get wrapped.
    """
    inputs = rv.owner.inputs  # type: ignore[union-attr]
    with pytensor.config.change_flags(floatX=dtype):
        outs = (
            transform.forward(rv, *inputs),  # type: ignore[arg-type]
            transform.backward(value, *inputs),  # type: ignore[arg-type]
            transform.log_jac_det(value, *inputs),  # type: ignore[arg-type]
        )
    return all(out.type.dtype == dtype for out in outs)  # type: ignore[union-attr]


def _cast_graph_floats(
    outputs: Sequence[Variable],
    from_dtype: str,
    to_dtype: str,
    frozen: Sequence[Variable] = (),
) -> tuple[list[Variable], dict[Variable, Variable]]:
    """Clone the graph of `outputs`, casting every `from_dtype` variable to `to_dtype`.

    The `frozen` variables are kept as they are, and the graph above them is not visited.
    Returns the converted outputs and a memo mapping old to new variables.
    """
    memo: dict[Variable, Variable] = {var: var for var in frozen}

    def mapped(var):
        if var not in memo:
            memo[var] = _cast_root(var, from_dtype, to_dtype)
        return memo[var]

    def stale(new_outputs):
        return any(getattr(out.type, "dtype", None) == from_dtype for out in new_outputs)

    for node in io_toposort(frozen, outputs):
        op, new_inputs = node.op, [mapped(var) for var in node.inputs]
        if (
            isinstance(op, Elemwise)
            and isinstance(op.scalar_op, Cast)
            and op.scalar_op.o_type.dtype == from_dtype
        ):
            # Redirect explicit casts (e.g. `x.astype("float64")`)
            new_outputs = [pt.cast(new_inputs[0], to_dtype)]
        else:
            op = _cast_op(op, from_dtype, to_dtype)
            if isinstance(op, ModelValuedVar) and (transform := op.transform) is not None:
                # Transform objects travel with the op and may embed constants of the old
                # dtype in the value-space graphs (logp, initial point); wrap them if so.
                rv, value = new_inputs
                if not _transform_keeps_dtype(transform, rv, value, to_dtype):
                    op = copy.copy(op)
                    op.transform = _CastedTransform(transform, from_dtype, to_dtype)
            new_outputs = op.make_node(*new_inputs).outputs
            if stale(new_outputs):
                # Discrete inputs upcast `to_dtype` back to `from_dtype` (e.g. int32 * float32)
                new_inputs = [
                    i.astype(to_dtype) if getattr(i.type, "dtype", None) in discrete_dtypes else i
                    for i in new_inputs
                ]
                new_outputs = op.make_node(*new_inputs).outputs
        if stale(new_outputs):
            name = next(out.name for out in outputs if node.outputs[0] in ancestors([out]))
            raise NotImplementedError(
                f"Cannot convert {node.outputs[0]} (in the graph of {name}) to {to_dtype}: "
                f"{op} outputs {from_dtype} regardless of its inputs."
            )
        for old, new in zip(node.outputs, new_outputs, strict=True):
            new.name = old.name
            memo[old] = new

    return [mapped(out) for out in outputs], memo


def _cast_model_floats(model: Model, from_dtype: str, to_dtype: str) -> Model:
    fg, _, initial_values = _fgraph_and_initial_values(model)
    new_outputs, _ = _cast_graph_floats(fg.outputs, from_dtype, to_dtype)
    # Dim lengths are integers, so there is nothing to replace
    new_model = model_from_fgraph(_model_fgraph_like(fg, new_outputs, {}), mutate_fgraph=True)
    for name, initval in initial_values.items():
        if isinstance(initval, np.ndarray) and initval.dtype.kind == "f":
            initval = initval.astype(to_dtype)
        new_model.set_initval(new_model[name], initval)
    return new_model


def model_to_float32(model: Model) -> Model:
    """Recreate a Model with all float64 variables and data cast to float32.

    Every float64 variable is converted: data (constants and `pm.Data`), free and
    observed RVs (including the inner graphs of symbolic and scan-based RVs like
    `ZeroSumNormal` or `AR`), value variables, Deterministics and Potentials. Integer,
    boolean and RNG variables are unaffected. Explicit `.astype("float64")` casts are
    redirected to float32.

    This can speed up sampling at the cost of precision — most on GPUs and for
    compute-bound models; on CPU backends gains depend on how memory- and
    BLAS-bound the model's logp is.

    The new model must be compiled and sampled under ``floatX="float32"``. Under the
    default ``floatX`` initial points are float64 and `pm.sample` raises:

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

    Raises
    ------
    NotImplementedError
        If an operation outputs float64 regardless of the dtype of its inputs.

    Notes
    -----
    Only the model graph is converted. Graphs built from it afterwards (logp, initial
    point) are as float32 as those of a model created under ``floatX="float32"``, e.g.
    the logp of `LKJCholeskyCov` still has float64 terms.

    ``pm.set_data`` casts new values to ``floatX``, so it also needs ``floatX="float32"``.

    Constant and strategy-string initial values are preserved (arrays are cast);
    symbolic initial values are not supported.
    """
    return _cast_model_floats(model, "float64", "float32")


def model_to_float64(model: Model) -> Model:
    """Recreate a Model with all float32 variables and data cast to float64.

    The counterpart of :func:`model_to_float32`, see its docstring for details. It is not
    an exact inverse: data rounded to float32 stays rounded.
    """
    return _cast_model_floats(model, "float32", "float64")


__all__ = ("freeze_dims_and_data", "freeze_model", "model_to_float32", "model_to_float64")
