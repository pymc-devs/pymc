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
import typing

import numpy as np

from pytensor import tensor as pt
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.fg import FunctionGraph
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
    MeasurableElemwise,
    MeasurableOp,
    ValuedRV,
    _icdf,
    _icdf_helper,
    _logcdf,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
    promised_valued_rv,
)
from pymc.logprob.censoring import MeasurableRound
from pymc.logprob.rewriting import (
    assume_valued_outputs,
    early_measurable_ir_rewrites_db,
    measurable_ir_rewrites_db,
    remove_promised_valued_rvs,
)
from pymc.logprob.utils import (
    check_potential_measurability,
    dirac_delta,
    filter_measurable_variables,
    get_related_valued_nodes,
    replace_rvs_by_values,
)
from pymc.pytensorf import constant_fold, get_symbolic_rv_shapes


class MeasurableMakeVector(MeasurableOp, MakeVector):
    """A placeholder used to specify a log-likelihood for a make_vector sub-graph."""


@_logprob.register(MeasurableMakeVector)
def logprob_make_vector(op, values, *base_rvs, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableMakeVector`."""
    # TODO: Sort out this circular dependency issue

    (value,) = values

    base_rvs = remove_promised_valued_rvs(base_rvs)

    base_rvs_to_values = {base_rv: value[i] for i, base_rv in enumerate(base_rvs)}
    for i, (base_rv, value) in enumerate(base_rvs_to_values.items()):
        base_rv.name = f"base_rv[{i}]"
        value.name = f"value[{i}]"

    logps = [_logprob_helper(base_rv, value) for base_rv, value in base_rvs_to_values.items()]

    # If the stacked variables depend on each other, we have to replace them by the respective values
    logps = replace_rvs_by_values(logps, rvs_to_values=base_rvs_to_values)

    return pt.stack(logps)


class MeasurableJoin(MeasurableOp, Join):
    """A placeholder used to specify a log-likelihood for a join sub-graph."""


@_logprob.register(MeasurableJoin)
def logprob_join(op, values, *base_rvs, **kwargs):
    """Compute the log-likelihood graph for a `Join`."""
    (value,) = values
    axis = op.axis

    base_rvs = remove_promised_valued_rvs(base_rvs)

    base_rv_shapes = [base_var.shape[axis] for base_var in base_rvs]

    # We don't need the graph to be constant, just to have RandomVariables removed
    base_rv_shapes = constant_fold(base_rv_shapes, raise_not_constant=False)

    split_values = pt.split(
        value,
        splits_size=base_rv_shapes,
        n_splits=len(base_rvs),
        axis=axis,
    )

    base_rvs_to_split_values = dict(zip(base_rvs, split_values))
    logps = [
        _logprob_helper(base_var, split_value)
        for base_var, split_value in base_rvs_to_split_values.items()
    ]

    if len({logp.ndim for logp in logps}) != 1:
        raise ValueError(
            "Joined logps have different number of dimensions, this can happen when "
            "joining univariate and multivariate distributions",
        )

    # If the stacked variables depend on each other, we have to replace them by the respective values
    logps = replace_rvs_by_values(logps, rvs_to_values=base_rvs_to_split_values)

    # Adjust for multivariate logp fewer dimensions to the right
    axis = min(axis, logps[0].ndim - 1)
    join_logprob = pt.concatenate(
        [pt.atleast_1d(logp) for logp in logps],
        axis=axis,
    )

    return join_logprob


@node_rewriter([MakeVector, Join])
def find_measurable_stacks(fgraph, node) -> list[TensorVariable] | None:
    r"""Find `Joins`\s and `MakeVector`\s for which a `logprob` can be computed."""
    from pymc.pytensorf import toposort_replace

    if isinstance(node.op, MeasurableOp):
        return None

    is_join = isinstance(node.op, Join)

    base_vars = node.inputs

    # Allow mixing potentially measurable inputs with deterministic ones.
    new_base_vars: list[TensorVariable] = []
    has_measurable = False
    for base_var in base_vars:
        if check_potential_measurability([base_var]):
            has_measurable = True
            new_base_vars.append(base_var)
        else:
            # `Op.__call__` is typed as returning `Variable | list[Variable]`, so mypy can't infer this is a TensorVariable
            new_base_vars.append(dirac_delta(base_var))  # type: ignore[arg-type]
    if not has_measurable:
        return None
    base_vars = new_base_vars
    base_vars = assume_valued_outputs(base_vars)
    if not all(var.owner and isinstance(var.owner.op, MeasurableOp) for var in base_vars):
        return None

    # Each base var will be "valued" by the logprob method, so other rewrites shouldn't mess with it
    # and potentially break interdependencies. For this reason, this rewrite should be applied early in
    # the IR construction
    replacements = [(base_var, promised_valued_rv(base_var)) for base_var in base_vars]
    temp_fgraph = FunctionGraph(outputs=base_vars, clone=False)
    toposort_replace(temp_fgraph, replacements)  # type: ignore[arg-type]
    new_base_vars = temp_fgraph.outputs  # type: ignore[assignment]

    if is_join:
        measurable_stack = MeasurableJoin(axis=node.op.axis)(*new_base_vars)
    else:
        measurable_stack = MeasurableMakeVector(node.op.dtype)(*new_base_vars)
    assert isinstance(measurable_stack, TensorVariable)

    return [measurable_stack]


class MeasurableSplit(MeasurableOp, Split):
    """A placeholder used to specify a log-likelihood for a split sub-graph."""


@node_rewriter([Split])
def find_measurable_splits(fgraph, node) -> list[TensorVariable] | None:
    if isinstance(node.op, MeasurableOp):
        return None

    x, splits = node.inputs
    if not filter_measurable_variables([x]):
        return None

    return MeasurableSplit(node.op.len_splits, node.op.axis).make_node(x, splits).outputs


@_logprob.register(MeasurableSplit)
def logprob_split(op: MeasurableSplit, values, x, splits, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableSplit`."""
    if len(values) != op.len_splits:
        # TODO: Don't rewrite the split in the first place if not all parts are linked to value variables
        # This also allows handling some cases where not all splits are used
        raise ValueError("Split logp requires the number of values to match the number of splits")

    axis = op.axis

    # Reverse the effects of split on the value variable
    join_value = pt.join(axis, *values)

    join_logp = _logprob_helper(x, join_value)

    reduced_dims = join_value.ndim - join_logp.ndim

    if reduced_dims and axis >= join_logp.ndim:
        # If the axis is over a dimension that was reduced in the logp (multivariate logp),
        # We cannot split it into distinct entries. The mapping between values-densities breaks.
        # We return the weighted logp by the split sizes. This is a good solution as any?
        split_weights = splits / pt.sum(splits)
        return [join_logp * split_weights[i] for i in range(typing.cast(int, op.len_splits))]

    # Otherwise we can split the logp as if the split were over batched dimensions
    return pt.split(
        join_logp,
        splits_size=splits,
        n_splits=op.len_splits,
        axis=axis,
    )


class MeasurableDimShuffle(MeasurableOp, DimShuffle):
    """A placeholder used to specify a log-likelihood for a dimshuffle sub-graph."""

    def __str__(self):
        return f"Measurable{super().__str__()}"


@_logprob.register(MeasurableDimShuffle)
def logprob_dimshuffle(op: MeasurableDimShuffle, values, base_var, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableDimShuffle`."""
    (value,) = values

    # Reverse the effects of dimshuffle on the value variable
    # First, drop any augmented dimensions and reinsert any dropped dimensions
    undo_ds: list[int | str] = [i for i, o in enumerate(op.new_order) if o != "x"]
    dropped_dims = tuple(op.drop)
    for dropped_dim in dropped_dims:
        undo_ds.insert(dropped_dim, "x")
    value = value.dimshuffle(undo_ds)

    # Then, unshuffle remaining dims
    original_shuffle = list(op.shuffle)
    for dropped_dim in dropped_dims:
        original_shuffle.insert(dropped_dim, dropped_dim)
    undo_ds = [original_shuffle.index(i) for i in range(len(original_shuffle))]
    value = value.dimshuffle(undo_ds)

    raw_logp = _logprob_helper(base_var, value)

    # Re-apply original dimshuffle, ignoring any support dimensions consumed by
    # the logprob function. This assumes that support dimensions are always in
    # the rightmost positions, and all we need to do is to discard the highest
    # indexes in the original dimshuffle order. Otherwise, there is no way of
    # knowing which dimensions were consumed by the logprob function.
    redo_ds = [o for o in op.new_order if o == "x" or o < raw_logp.ndim]
    return raw_logp.dimshuffle(redo_ds)


def _elemwise_root(var: TensorVariable) -> TensorVariable | None:
    """Walk through dimension-preserving measurable operations to the root variable."""
    from pymc.distributions.distribution import SymbolicRandomVariable
    from pymc.logprob.transforms import MeasurableTransform

    if isinstance(var.owner.op, RandomVariable | SymbolicRandomVariable):
        return var
    elif isinstance(var.owner.op, MeasurableTransform):
        return _elemwise_root(var.owner.inputs[var.owner.op.measurable_input_idx])
    else:
        return None


def _elemwise_univariate_chain(fgraph, node) -> bool:
    # Check whether only Elemwise operations connect a base univariate RV to the valued node through var.
    inp = node.inputs[0]
    [out] = node.outputs

    # Check that the root is a univariate distribution linked by only elemwise operations
    root = _elemwise_root(inp)
    if root is None:
        return False
    elif root.owner.op.ndim_supp != 0:
        # This is still fine if the variable is directly valued
        return any(get_related_valued_nodes(fgraph, node))

    def elemwise_leaf(var: TensorVariable, clients=fgraph.clients) -> bool:
        var_clients = clients[var]
        if len(var_clients) != 1:
            return False
        [(client, _)] = var_clients
        if isinstance(client.op, ValuedRV):
            return True
        elif isinstance(client.op, Elemwise) and len(client.outputs) == 1:
            return elemwise_leaf(client.outputs[0])
        else:
            return False

    # Check that the path to the valued node consists only of elemwise operations
    return elemwise_leaf(out)


@node_rewriter([DimShuffle])
def find_measurable_dimshuffles(fgraph, node) -> list[TensorVariable] | None:
    r"""Find `Dimshuffle`\s for which a `logprob` can be computed."""
    if isinstance(node.op, MeasurableOp):
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    # In cases where DimShuffle transposes dimensions, we only apply this rewrite when only Elemwise
    # operations separate it from the valued node. Further transformations likely need to know where
    # the support axes are for a correct implementation (and thus assume they are the rightmost axes).
    # TODO: When we include the support axis as meta information in each  intermediate MeasurableVariable,
    #  we can lift this restriction (see https://github.com/pymc-devs/pymc/issues/6360)
    if tuple(node.op.shuffle) != tuple(sorted(node.op.shuffle)) and not _elemwise_univariate_chain(
        fgraph, node
    ):
        return None

    base_var = node.inputs[0]

    measurable_dimshuffle = MeasurableDimShuffle(
        input_ndim=node.op.input_ndim, new_order=node.op.new_order
    )(base_var)
    assert isinstance(measurable_dimshuffle, TensorVariable)

    return [measurable_dimshuffle]


class MeasurableBroadcast(MeasurableOp, Alloc):
    """A placeholder used to specify a log-likelihood for a broadcast sub-graph."""


@_logprob.register(MeasurableBroadcast)
def broadcast_logprob(op, values, rv, *shape, **kwargs):
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
    [value] = values

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
    logp = _logprob_helper(rv, rv_value)

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

    return pt.switch(check, logp, -np.inf)


@node_rewriter([Alloc])
def find_measurable_broadcast(fgraph, node):
    r"""Find measurable broadcasts ``broadcast_to(rv, shape)``."""
    if isinstance(node.op, MeasurableOp):
        return None

    base_rv, *shape = node.inputs

    if not filter_measurable_variables([base_rv]):
        return None

    if check_potential_measurability(shape):
        return None

    # The broadcast dimensions are degenerate copies of the base variable's entries.
    # Without meta information about the support axes, other rewrites would treat the
    # copies as independent entries (e.g., counting the jacobian of a transform once
    # per copy), so the broadcast is only claimed when directly valued, where no other
    # rewrite needs to reason about the returned logp.
    # TODO: When we include the support axis as meta information in each intermediate
    #  MeasurableVariable, we can lift this restriction (see https://github.com/pymc-devs/pymc/issues/6360)
    if not any(get_related_valued_nodes(fgraph, node)):
        return None

    return [MeasurableBroadcast()(base_rv, *shape)]


class MeasurableCast(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a cast sub-graph."""

    valid_scalar_types = (Cast,)


@node_rewriter([elemwise_of(Cast)])
def find_measurable_casts(fgraph, node) -> list[TensorVariable] | None:
    r"""Find measurable casts that do not discretize the base variable."""
    if isinstance(node.op, MeasurableOp):
        return None

    [base_var] = node.inputs
    if not filter_measurable_variables([base_var]):
        return None

    [out] = node.outputs
    # bool < integer < float; casting to a lower kind discretizes the base variable,
    # which is not a measure-preserving identity
    kind_order = {"b": 0, "u": 1, "i": 1, "f": 2}
    in_dtype = np.dtype(base_var.type.dtype)
    out_dtype = np.dtype(out.type.dtype)
    in_kind = kind_order.get(in_dtype.kind)
    out_kind = kind_order.get(out_dtype.kind)
    if in_kind is None or out_kind is None:
        # Kinds we can't order, such as complex
        return None
    if out_kind < in_kind:
        # A float -> signed int cast rounds towards zero, so it is a `trunc` composed
        # with a cast that merely relabels the dtype. Introduce the `trunc` explicitly
        # and let `find_measurable_roundings` claim it; the relabelling cast is then
        # measure-preserving and claimed on a later pass.
        # The other narrowing casts are not truncations: unsigned ints wrap around for
        # negative values (-2.7 -> 254 for uint8), and bool tests `x != 0`, which
        # collapses the support onto two points instead of partitioning it.
        if in_dtype.kind != "f" or out_dtype.kind != "i":
            return None
        # Rewriting to the `trunc` the user could have written themselves leaves the
        # judgement of whether the base may be truncated to `find_measurable_roundings`,
        # which declines the bases whose support it cannot establish, rather than
        # duplicating that reasoning here. Once the `trunc` is in place the cast only
        # relabels the dtype and is claimed as measure-preserving below; skipping it for
        # an already rounded base is what brings the rewrite to a fixpoint.
        if not isinstance(base_var.owner.op, MeasurableRound):
            return [pt.cast(pt.trunc(base_var), out_dtype.name)]

    if in_kind < kind_order["f"] and out_kind == kind_order["f"]:
        # Casting a discrete variable to float hides its discreteness from other
        # rewrites, which classify variables by dtype (e.g., a continuous jacobian
        # would wrongly be applied to scalings of the cast variable).
        # Only claim the cast when it is directly valued.
        if not all(
            client != "output" and isinstance(client.op, ValuedRV)
            for client, _ in fgraph.clients[out]
        ):
            return None

    return [MeasurableCast(scalar_op=node.op.scalar_op)(base_var)]  # type: ignore[list-item]


@_logprob.register(MeasurableCast)
def cast_logprob(op, values, base_var, **kwargs):
    [value] = values
    # The cast is measure-preserving; the value is passed through as is.
    # Casting it back could silently map impossible values to possible ones
    # (e.g., 1.5 -> 1 for an integer base variable).
    return _logprob_helper(base_var, value)


@_logcdf.register(MeasurableCast)
def cast_logcdf(op, value, base_var):
    if base_var.type.dtype.startswith(("int", "uint", "bool")):
        # For a discrete base variable, P(cast(X) <= 1.5) = P(X <= 1)
        value = pt.floor(value)
    return _logcdf_helper(base_var, value)


@_icdf.register(MeasurableCast)
def cast_icdf(op, value, base_var):
    return pt.cast(_icdf_helper(base_var, value), op.scalar_op.o_type.dtype)


class MeasurableScalarFromTensor(MeasurableOp, ScalarFromTensor):
    """ScalarFromTensor of a measurable variable."""


class MeasurableSpecifyAssumptions(MeasurableOp, SpecifyAssumptions):
    """SpecifyAssumptions of a measurable variable."""


class MeasurableDeepCopyOp(MeasurableOp, DeepCopyOp):
    """DeepCopyOp of a measurable variable."""


@node_rewriter([ScalarFromTensor, SpecifyAssumptions, DeepCopyOp])
def find_measurable_identity_ops(fgraph, node) -> list | None:
    r"""Find identity-like operations that leave the value of a measurable variable untouched."""
    if isinstance(node.op, MeasurableOp):
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    [base_var] = node.inputs
    new_op: MeasurableOp
    if isinstance(node.op, ScalarFromTensor):
        new_op = MeasurableScalarFromTensor()
    elif isinstance(node.op, SpecifyAssumptions):
        new_op = MeasurableSpecifyAssumptions(dict(node.op.assumptions))
    else:
        new_op = MeasurableDeepCopyOp()
    return [new_op(base_var)]


@_logprob.register(MeasurableScalarFromTensor)
@_logprob.register(MeasurableSpecifyAssumptions)
@_logprob.register(MeasurableDeepCopyOp)
def identity_logprob(op, values, base_var, **kwargs):
    # The operation does not change the value, so the measure passes through.
    # Note that the assumptions of a SpecifyAssumptions are NOT stamped onto the value:
    # the value is an arbitrary query point (e.g., a sampler proposal), and a density
    # rewritten under trusted assumptions could return a finite logp where the honest
    # density would return -inf, just like a skipped support check.
    [value] = values
    return _logprob_helper(base_var, value)


@_logcdf.register(MeasurableScalarFromTensor)
@_logcdf.register(MeasurableSpecifyAssumptions)
@_logcdf.register(MeasurableDeepCopyOp)
def identity_logcdf(op, value, base_var):
    return _logcdf_helper(base_var, value)


@_icdf.register(MeasurableScalarFromTensor)
@_icdf.register(MeasurableSpecifyAssumptions)
@_icdf.register(MeasurableDeepCopyOp)
def identity_icdf(op, value, base_var):
    return _icdf_helper(base_var, value)


class MeasurableJoinDims(MeasurableOp, JoinDims):
    """A placeholder used to specify a log-likelihood for a join_dims sub-graph."""


class MeasurableSplitDims(MeasurableOp, SplitDims):
    """A placeholder used to specify a log-likelihood for a split_dims sub-graph."""


@node_rewriter([JoinDims, SplitDims])
def find_measurable_join_split_dims(fgraph, node) -> list[TensorVariable] | None:
    r"""Find `JoinDims` and `SplitDims` for which a `logprob` can be computed."""
    if isinstance(node.op, MeasurableOp):
        return None

    base_var, *other_inputs = node.inputs

    if not filter_measurable_variables([base_var]):
        return None

    if check_potential_measurability(other_inputs):
        return None

    if isinstance(node.op, JoinDims):
        # A join that straddles the batch/support boundary merges batch and support
        # axes into a single one, breaking the assumption that support axes are the
        # rightmost positions, which other rewrites rely on. Such joins are only valid
        # when directly valued. Joins contained in the batch axes (or in the support
        # axes, which simply become fewer rightmost axes) preserve the assumption and
        # need no restriction; splitting never straddles, since it only inflates a
        # single axis.
        # TODO: When we include the support axis as meta information in each intermediate
        #  MeasurableVariable, the root walk becomes unnecessary (see https://github.com/pymc-devs/pymc/issues/6360)
        start_axis, n_axes = node.op.start_axis, node.op.n_axes
        root = _elemwise_root(base_var)
        if root is None:
            straddles = True
        else:
            batch_ndim = base_var.type.ndim - root.owner.op.ndim_supp
            straddles = start_axis < batch_ndim < start_axis + n_axes
        if straddles and not any(get_related_valued_nodes(fgraph, node)):
            return None
        measurable_op: MeasurableOp = MeasurableJoinDims(start_axis, n_axes)
    else:
        measurable_op = MeasurableSplitDims(node.op.axis)

    return [measurable_op(base_var, *other_inputs)]  # type: ignore[operator]


@_logprob.register(MeasurableJoinDims)
def logprob_join_dims(op, values, base_var, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableJoinDims`."""
    (value,) = values

    [base_shape] = get_symbolic_rv_shapes([base_var])
    unjoined_shape = [base_shape[i] for i in op.axis_range]
    unjoined_value = split_dims(value, shape=unjoined_shape, axis=op.start_axis)

    raw_logp = _logprob_helper(base_var, unjoined_value)

    # Re-join the value dimensions, ignoring any support dimensions consumed by the
    # logprob function (assumed to be the rightmost positions). A join lying entirely
    # within consumed support dimensions leaves nothing to re-join, but a length-zero
    # join (expand_dims) of remaining dimensions is still re-applied.
    if op.start_axis > raw_logp.ndim or (op.n_axes > 0 and op.start_axis >= raw_logp.ndim):
        return raw_logp
    return join_dims(raw_logp, op.start_axis, min(op.n_axes, raw_logp.ndim - op.start_axis))


@_logprob.register(MeasurableSplitDims)
def logprob_split_dims(op, values, base_var, shape, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableSplitDims`."""
    (value,) = values

    n_axes = value.type.ndim - base_var.type.ndim + 1
    joined_value = join_dims(value, start_axis=op.axis, n_axes=n_axes)

    raw_logp = _logprob_helper(base_var, joined_value)

    # Re-split the value dimensions, unless the split dimension was a support dimension
    # consumed by the logprob function (assumed to be the rightmost positions)
    if op.axis >= raw_logp.ndim:
        return raw_logp
    return split_dims(raw_logp, shape=shape, axis=op.axis)


measurable_ir_rewrites_db.register(
    "find_measurable_casts", find_measurable_casts, "basic", "tensor"
)

# Note: pymc-extras (<0.5) registers an equivalent rewrite under the name
# "find_measurable_value_identities"; the names must not collide
measurable_ir_rewrites_db.register(
    "find_measurable_identity_ops", find_measurable_identity_ops, "basic", "tensor"
)

measurable_ir_rewrites_db.register(
    "find_measurable_join_split_dims", find_measurable_join_split_dims, "basic", "tensor"
)


measurable_ir_rewrites_db.register("dimshuffle_lift", local_dimshuffle_rv_lift, "basic", "tensor")


# We register this later than `dimshuffle_lift` so that it is only applied as a fallback
measurable_ir_rewrites_db.register(
    "find_measurable_dimshuffles", find_measurable_dimshuffles, "basic", "tensor"
)

early_measurable_ir_rewrites_db.register(
    "find_measurable_stacks",
    find_measurable_stacks,
    "basic",
    "tensor",
)

measurable_ir_rewrites_db.register(
    "find_measurable_splits",
    find_measurable_splits,
    "basic",
    "tensor",
)

measurable_ir_rewrites_db.register(
    "find_measurable_broadcast",
    find_measurable_broadcast,
    "basic",
    "tensor",
)
