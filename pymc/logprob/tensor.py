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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np

from pytensor import tensor as pt
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import Cast
from pytensor.tensor import TensorVariable
from pytensor.tensor.basic import Alloc, Join, MakeVector, ScalarFromTensor, Split
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.rewriting import (
    local_dimshuffle_rv_lift,
)
from pytensor.tensor.reshape import JoinDims, SplitDims, join_dims, split_dims
from pytensor.tensor.rewriting.basic import elemwise_of

from pymc.logprob.abstract import (
    MeasurableOp,
    _icdf,
    _icdf_helper,
    _logcdf,
    _logcdf_helper,
    logprob_query,
    valued_rv,
)
from pymc.logprob.censoring import ROUNDING_OPS
from pymc.logprob.mixture import rv_pull_down
from pymc.logprob.query import (
    UnsupportedObservation,
    bind_value,
    contains_random,
    density_ndim,
    infer_density_ndim,
    infer_support_axes,
    other_query_uses,
    output_queries,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)
from pymc.logprob.rewriting import (
    measurable_ir_rewrites_db,
)
from pymc.logprob.utils import (
    filter_measurable_variables,
)
from pymc.pytensorf import resolve_shapes


@rewrite_logprob_query.register(Alloc)
def rewrite_broadcast_logprob(op, fgraph, query, **kwargs):
    """Log-probability expression for (statically-)broadcasted RV.

    The probability is the same as the base RV, if no broadcasting had happened.
    The broadcast dimensions are degenerate copies of the base entries, so they are
    consumed like support dimensions and disappear from the logp:

    ``logp(broadcast_to(normal(size=(3, 1)), (2, 3, 4)), zeros((2, 3, 4))) == logp(normal(size=(3,)), zeros((3,)))``

    And zero if the value couldn't have possibly originated via broadcasting:

    ``logp(broadcast_to(normal(size=(1,)), (3,)), [1, 2, 3]) == -np.inf``

    The consistency check is elementwise over the base variable's batch dimensions,
    so entries that were not broadcast from each other keep their own logp.
    """
    output, value = query_parts(query)
    rv, *shape = output.owner.inputs

    n_new_dims = len(shape) - rv.ndim
    assert n_new_dims >= 0

    # Enumerate broadcasted dims
    expanded_dims = tuple(range(n_new_dims))
    broadcast_dims = tuple(
        i + n_new_dims
        for i, (v_bcast, rv_bcast) in enumerate(
            zip(value.broadcastable[n_new_dims:], rv.broadcastable)
        )
        if (not v_bcast) and rv_bcast
    )

    # "Unbroadcast" value via indexing.
    # All entries in the broadcasted dimensions should be the same, so we simply select
    # the first of each. Broadcast dims are re-inserted with expand_dims (rather than
    # sliced with `0:1`) so they are statically known to be broadcastable.
    indices = []
    for i in range(value.ndim):
        # Remove expanded and broadcasted (but not expanded) dims
        if i in expanded_dims or i in broadcast_dims:
            indices.append(0)
        else:
            indices.append(slice(None))

    unbroadcast_value = value[tuple(indices)]
    # The base variable still carries the broadcast dims (with length 1); they are
    # re-inserted with expand_dims so they are statically known to be broadcastable
    rv_value = unbroadcast_value
    if broadcast_dims:
        rv_value = pt.expand_dims(rv_value, tuple(d - n_new_dims for d in broadcast_dims))
    logp = logprob_query(bind_value(fgraph, rv, rv_value))

    # The broadcast dims are consumed like support dims and disappear from the logp
    core_ndim = rv_value.ndim - logp.ndim
    squeeze_axes = tuple(d - n_new_dims for d in broadcast_dims if (d - n_new_dims) < logp.ndim)
    if squeeze_axes:
        logp = pt.squeeze(logp, axis=squeeze_axes)

    # Check that dependent values were indeed identical, by comparing with a re-broadcasted value.
    # The check is reduced only over the expanded/broadcast dimensions (and any support
    # dimensions consumed by the base logp), so unrelated batch entries do not
    # contaminate each other.
    # Note: This could fail due to float-precision issues.
    # If that proves to be a problem we should switch to `pt.allclose`
    valid_value = pt.broadcast_to(rv_value, shape)
    core_dims = tuple(range(value.ndim - core_ndim, value.ndim))
    reduced_dims = tuple(sorted({*expanded_dims, *broadcast_dims, *core_dims}))
    check = pt.all(pt.eq(value, valid_value), axis=reduced_dims)

    return [pt.switch(check, logp, -np.inf)]


@node_rewriter([elemwise_of(Cast)])
def find_measurable_casts(fgraph, node) -> list[TensorVariable] | None:
    r"""Find measurable casts that do not discretize the base variable."""
    if isinstance(node.op, MeasurableOp):
        return None

    [base_var] = node.inputs
    if not filter_measurable_variables([base_var]):
        return None

    if base_var.dtype.startswith("float") and node.outputs[0].dtype.startswith("int"):
        if not (
            isinstance(base_var.owner_op, Elemwise)
            and isinstance(base_var.owner.op.scalar_op, ROUNDING_OPS)
        ):
            return [pt.cast(pt.trunc(base_var), node.outputs[0].dtype)]
    return None


@_logcdf.register(Cast)
def cast_logcdf(op, value, base_var):
    if base_var.type.dtype.startswith(("int", "uint", "bool")):
        # For a discrete base variable, P(cast(X) <= 1.5) = P(X <= 1)
        value = pt.floor(value)
    return _logcdf_helper(base_var, value)


@_icdf.register(Cast)
def cast_icdf(op, value, base_var):
    return pt.cast(_icdf_helper(base_var, value), op.o_type.dtype)


@_logcdf.register(ScalarFromTensor)
@_logcdf.register(SpecifyAssumptions)
@_logcdf.register(DeepCopyOp)
def identity_logcdf(op, value, base_var):
    return _logcdf_helper(base_var, value)


@_icdf.register(ScalarFromTensor)
@_icdf.register(SpecifyAssumptions)
@_icdf.register(DeepCopyOp)
def identity_icdf(op, value, base_var):
    return _icdf_helper(base_var, value)


measurable_ir_rewrites_db.register(
    "find_measurable_casts", find_measurable_casts, "basic", "tensor"
)

# Note: pymc-extras (<0.5) registers an equivalent rewrite under the name
# "find_measurable_value_identities"; the names must not collide
measurable_ir_rewrites_db.register("dimshuffle_lift", local_dimshuffle_rv_lift, "basic", "tensor")


@infer_density_ndim.register(Join)
def join_density_ndim(op, var):
    ranks = {density_ndim(inp) for inp in var.owner.inputs}
    if len(ranks) != 1:
        raise ValueError("Joined logps have different number of dimensions")
    # Event factors are scalars, but concatenating them still produces a vector.
    return max(1, ranks.pop())


@infer_density_ndim.register(JoinDims)
def join_dims_density_ndim(op, var):
    base = var.owner.inputs[0]
    batch_axes = set(range(base.ndim)) - set(support_axes(base))
    joined_batch_axes = batch_axes & set(op.axis_range)
    return len(batch_axes) - len(joined_batch_axes) + bool(joined_batch_axes)


@infer_support_axes.register(Join)
def measure_join(op, var):
    metas = [support_axes(inp) for inp in var.owner.inputs]
    if len({density_ndim(inp) for inp in var.owner.inputs}) != 1:
        raise ValueError("Joined logps have different number of dimensions")
    first = metas[0]
    if any(meta != first for meta in metas):
        raise UnsupportedObservation("Joined variables have different support axes")
    axes = first
    if axes is not None and op.axis % var.ndim in axes:
        axes = None
    return axes


@infer_support_axes.register(MakeVector)
def measure_vector(op, var):
    if any(density_ndim(inp) != 0 for inp in var.owner.inputs):
        raise UnsupportedObservation("MakeVector requires scalar density terms")
    return ()


@infer_support_axes.register(Split)
def measure_split(op, var):
    return support_axes(var.owner.inputs[0])


@infer_support_axes.register(DimShuffle)
def measure_dimshuffle(op, var):
    meta = support_axes(var.owner.inputs[0])
    if meta is None:
        raise UnsupportedObservation("Cannot remove or reinterpret these support axes")
    axes = tuple(i for i, axis in enumerate(op.new_order) if axis in meta)
    return axes


@infer_support_axes.register(JoinDims)
def measure_join_dims(op, var):
    axes = support_axes(var.owner.inputs[0])
    if axes is None:
        raise UnsupportedObservation("Cannot merge axes with an unknown event layout")
    region = set(op.axis_range)
    if region & set(axes) and region - set(axes):
        return None
    result = {
        axis if axis < op.start_axis else axis - op.n_axes + 1
        for axis in axes
        if axis not in region
    }
    if region & set(axes):
        result.add(op.start_axis)
    return tuple(sorted(result))


@infer_support_axes.register(SplitDims)
def measure_split_dims(op, var):
    base = var.owner.inputs[0]
    axes = support_axes(base)
    if axes is None:
        raise UnsupportedObservation("Cannot split axes with an unknown event layout")
    n_axes = var.ndim - base.ndim + 1
    result = [axis if axis < op.axis else axis + n_axes - 1 for axis in axes if axis != op.axis]
    if op.axis in axes:
        result.extend(range(op.axis, op.axis + n_axes))
    return tuple(sorted(result))


@rewrite_logprob_query.register(Join)
def rewrite_join_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    components = list(rv.owner.inputs)
    axis = op.axis % rv.ndim
    lengths = resolve_shapes([component.shape[axis] for component in components], fgraph=fgraph)
    if any(contains_random(length) for length in lengths):
        raise UnsupportedObservation("Join lengths still depend on unvalued random variables")
    rv, value = query_parts(query)
    components = list(rv.owner.inputs)
    values = pt.split(value, lengths, n_splits=len(components), axis=axis)
    terms = [
        logprob_query(bind_value(fgraph, component, val))
        for component, val in zip(components, values)
    ]
    axes = support_axes(components[0])
    density_axis = axis - sum(a < axis for a in axes) if axes is not None else axis
    density_axis = min(density_axis, max(0, terms[0].ndim - 1))
    return [pt.concatenate([pt.atleast_1d(term) for term in terms], axis=density_axis)]


@rewrite_logprob_query.register(MakeVector)
def rewrite_vector_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    return [
        pt.stack(
            [
                logprob_query(bind_value(fgraph, inp, value[i]))
                for i, inp in enumerate(list(rv.owner.inputs))
            ]
        )
    ]


@rewrite_logprob_query.register(Split)
def rewrite_split_logprob(op, fgraph, query, **kwargs):
    rv, _ = query_parts(query)
    producer = rv.owner
    queries = output_queries(fgraph, producer)
    base, lengths = producer.inputs
    axis = op.axis % base.ndim
    meta = support_axes(base)
    if len(queries) != len(producer.outputs):
        missing = [out for out in producer.outputs if out not in queries]
        if other_query_uses(fgraph, missing, set(queries.values())):
            return None
        if meta is None or axis in meta:
            raise UnsupportedObservation("Partial Split: no marginalization rule is implemented")
        offsets = pt.concatenate([pt.zeros(1, dtype=lengths.dtype), pt.cumsum(lengths)])
        replacements = {}
        for out, pending in queries.items():
            index = (slice(None),) * axis + (slice(offsets[out.index], offsets[out.index + 1]),)
            marginal = rv_pull_down(base[index])
            if not isinstance(marginal.owner_op, RandomVariable):
                raise UnsupportedObservation(
                    "Partial Split: no marginalization rule is implemented"
                )
            replacements[pending.outputs[0]] = logprob_query(
                valued_rv(marginal, query_parts(pending)[1])
            )
        return replacements
    if meta is None:
        raise UnsupportedObservation("Splitting an event needs support-axis metadata")
    value = pt.concatenate([query_parts(queries[out])[1] for out in producer.outputs], axis=axis)
    term = logprob_query(bind_value(fgraph, base, value))
    if axis in meta:
        # Preserve the allocation of a joint event density by component size.
        weights = lengths / pt.sum(lengths)
        terms = [term * weights[i] for i in range(len(queries))]
    else:
        density_axis = axis - sum(a < axis for a in meta)
        terms = pt.split(term, lengths, n_splits=len(queries), axis=density_axis)
    return {queries[out].outputs[0]: term for out, term in zip(producer.outputs, terms)}


@rewrite_logprob_query.register(DimShuffle)
def rewrite_dimshuffle_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    (base,) = rv.owner.inputs
    meta = support_axes(base)
    undo = [i for i, axis in enumerate(op.new_order) if axis != "x"]
    for axis in op.drop:
        undo.insert(axis, "x")
    value = value.dimshuffle(undo)
    shuffle = list(op.shuffle)
    for axis in op.drop:
        shuffle.insert(axis, axis)
    value = value.dimshuffle([shuffle.index(i) for i in range(len(shuffle))])
    term = logprob_query(bind_value(fgraph, base, value))
    batch_axes = [axis for axis in range(base.ndim) if axis not in meta]
    order = [
        "x" if axis == "x" else batch_axes.index(axis)
        for axis in op.new_order
        if axis == "x" or axis in batch_axes
    ]
    return [term.dimshuffle(order)]


@rewrite_logprob_query.register(JoinDims)
def rewrite_join_dims_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    (base,) = rv.owner.inputs
    axes = support_axes(base)
    shapes = resolve_shapes([base.shape[i] for i in op.axis_range], fgraph=fgraph)
    backward = split_dims(value, shape=shapes, axis=op.start_axis)
    term = logprob_query(bind_value(fgraph, base, backward))
    batch_axes = [axis for axis in range(base.ndim) if axis not in axes]
    n_axes = sum(axis in op.axis_range for axis in batch_axes)
    if n_axes == 0 and op.n_axes != 0:
        return [term]
    start = sum(axis < op.start_axis for axis in batch_axes)
    return [join_dims(term, start_axis=start, n_axes=n_axes)]


@rewrite_logprob_query.register(SplitDims)
def rewrite_split_dims_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    base, shape = rv.owner.inputs
    axes = support_axes(base)
    n_axes = value.ndim - base.ndim + 1
    backward = join_dims(value, start_axis=op.axis, n_axes=n_axes)
    term = logprob_query(bind_value(fgraph, base, backward))
    if op.axis in axes:
        return [term]
    axis = op.axis - sum(axis < op.axis for axis in axes)
    return [split_dims(term, shape=shape, axis=axis)]


@infer_support_axes.register(Alloc)
def measure_broadcast(op, var):
    base = var.owner.inputs[0]
    offset = var.ndim - base.ndim
    axes = {axis + offset for axis in support_axes(base)}
    axes.update(range(offset))
    axes.update(
        i + offset
        for i, (a, b) in enumerate(zip(base.broadcastable, var.broadcastable[offset:]))
        if a and not b
    )
    return tuple(sorted(axes))


@infer_support_axes.register(ScalarFromTensor)
@infer_support_axes.register(SpecifyAssumptions)
@infer_support_axes.register(DeepCopyOp)
def measure_identity(op, var):
    return support_axes(var.owner.inputs[0])


@rewrite_logprob_query.register(ScalarFromTensor)
@rewrite_logprob_query.register(SpecifyAssumptions)
@rewrite_logprob_query.register(DeepCopyOp)
def rewrite_identity_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    # Assumptions describe the RV, not arbitrary query points outside its support.
    return [logprob_query(bind_value(fgraph, rv.owner.inputs[0], pt.as_tensor_variable(value)))]


@rewrite_logprob_query.register(Cast)
def rewrite_cast_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    base = rv.owner.inputs[0]
    kinds = {"b": 0, "u": 1, "i": 1, "f": 2}
    in_kind = kinds.get(np.dtype(base.dtype).kind)
    out_kind = kinds.get(np.dtype(rv.dtype).kind)
    if in_kind is None or out_kind is None:
        return None
    if out_kind < in_kind and not (
        np.dtype(rv.dtype).kind == "i"
        and isinstance(base.owner_op, Elemwise)
        and isinstance(base.owner.op.scalar_op, ROUNDING_OPS)
    ):
        return None
    # Preserve the query point: casting 1.5 back to an integer would hide an impossible value.
    return [logprob_query(bind_value(fgraph, base, value))]
