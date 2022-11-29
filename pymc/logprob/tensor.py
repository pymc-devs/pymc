from typing import List, Optional, Union

import aesara
from aesara import tensor as at
from aesara.graph.op import compute_test_value
from aesara.graph.rewriting.basic import node_rewriter
from aesara.tensor.basic import Join, MakeVector
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.rewriting import local_dimshuffle_rv_lift, local_rv_size_lift

from aeppl.abstract import MeasurableVariable, assign_custom_measurable_outputs
from aeppl.logprob import _logprob, logprob
from aeppl.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


@node_rewriter([BroadcastTo])
def naive_bcast_rv_lift(fgraph, node):
    """Lift a ``BroadcastTo`` through a ``RandomVariable`` ``Op``.

    XXX: This implementation simply broadcasts the ``RandomVariable``'s
    parameters, which won't always work (e.g. multivariate distributions).

    TODO: Instead, it should use ``RandomVariable.ndim_supp``--and the like--to
    determine which dimensions of each parameter need to be broadcasted.
    Also, this doesn't need to remove ``size`` to perform the lifting, like it
    currently does.
    """

    if not (
        isinstance(node.op, BroadcastTo)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, RandomVariable)
    ):
        return None  # pragma: no cover

    bcast_shape = node.inputs[1:]

    rv_var = node.inputs[0]
    rv_node = rv_var.owner

    if hasattr(fgraph, "dont_touch_vars") and rv_var in fgraph.dont_touch_vars:
        return None  # pragma: no cover

    # Do not replace RV if it is associated with a value variable
    rv_map_feature: Optional[PreserveRVMappings] = getattr(
        fgraph, "preserve_rv_mappings", None
    )
    if rv_map_feature is not None and rv_var in rv_map_feature.rv_values:
        return None

    if not bcast_shape:
        # The `BroadcastTo` is broadcasting a scalar to a scalar (i.e. doing nothing)
        assert rv_var.ndim == 0
        return [rv_var]

    size_lift_res = local_rv_size_lift.transform(fgraph, rv_node)
    if size_lift_res is None:
        lifted_node = rv_node
    else:
        _, lifted_rv = size_lift_res
        lifted_node = lifted_rv.owner

    rng, size, dtype, *dist_params = lifted_node.inputs

    new_dist_params = [
        at.broadcast_to(
            param,
            at.broadcast_shape(
                tuple(param.shape), tuple(bcast_shape), arrays_are_shapes=True
            ),
        )
        for param in dist_params
    ]
    bcasted_node = lifted_node.op.make_node(rng, size, dtype, *new_dist_params)

    if aesara.config.compute_test_value != "off":
        compute_test_value(bcasted_node)

    return [bcasted_node.outputs[1]]


class MeasurableMakeVector(MakeVector):
    """A placeholder used to specify a log-likelihood for a cumsum sub-graph."""


MeasurableVariable.register(MeasurableMakeVector)


@_logprob.register(MeasurableMakeVector)
def logprob_make_vector(op, values, *base_vars, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableMakeVector`."""
    (value,) = values

    return at.stack(
        [logprob(base_var, value[i]) for i, base_var in enumerate(base_vars)]
    )


class MeasurableJoin(Join):
    """A placeholder used to specify a log-likelihood for a join sub-graph."""


MeasurableVariable.register(MeasurableJoin)


@_logprob.register(MeasurableJoin)
def logprob_join(op, values, axis, *base_vars, **kwargs):
    """Compute the log-likelihood graph for a `Join`."""
    (value,) = values

    split_values = at.split(
        value,
        splits_size=[base_var.shape[axis] for base_var in base_vars],
        n_splits=len(base_vars),
        axis=axis,
    )

    logps = [
        logprob(base_var, split_value)
        for base_var, split_value in zip(base_vars, split_values)
    ]

    if len(set(logp.ndim for logp in logps)) != 1:
        raise ValueError(
            "Joined logps have different number of dimensions, this can happen when "
            "joining univariate and multivariate distributions",
        )

    base_vars_ndim_supp = split_values[0].ndim - logps[0].ndim
    join_logprob = at.concatenate(
        [
            at.atleast_1d(logprob(base_var, split_value))
            for base_var, split_value in zip(base_vars, split_values)
        ],
        axis=axis - base_vars_ndim_supp,
    )

    return join_logprob


@node_rewriter([MakeVector, Join])
def find_measurable_stacks(
    fgraph, node
) -> Optional[List[Union[MeasurableMakeVector, MeasurableJoin]]]:
    r"""Finds `Joins`\s and `MakeVector`\s for which a `logprob` can be computed."""

    if isinstance(node.op, (MeasurableMakeVector, MeasurableJoin)):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(
        fgraph, "preserve_rv_mappings", None
    )

    if rv_map_feature is None:
        return None  # pragma: no cover

    stack_out = node.outputs[0]

    is_join = isinstance(node.op, Join)

    if is_join:
        axis, *base_vars = node.inputs
    else:
        base_vars = node.inputs

    if not all(
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and base_var not in rv_map_feature.rv_values
        for base_var in base_vars
    ):
        return None  # pragma: no cover

    # Make base_vars unmeasurable
    base_vars = [
        assign_custom_measurable_outputs(base_var.owner) for base_var in base_vars
    ]

    if is_join:
        measurable_stack = MeasurableJoin()(axis, *base_vars)
    else:
        measurable_stack = MeasurableMakeVector(node.op.dtype)(*base_vars)

    measurable_stack.name = stack_out.name

    return [measurable_stack]


class MeasurableDimShuffle(DimShuffle):
    """A placeholder used to specify a log-likelihood for a dimshuffle sub-graph."""

    # Need to get the absolute path of `c_func_file`, otherwise it tries to
    # find it locally and fails when a new `Op` is initialized
    c_func_file = DimShuffle.get_path(DimShuffle.c_func_file)


MeasurableVariable.register(MeasurableDimShuffle)


@_logprob.register(MeasurableDimShuffle)
def logprob_dimshuffle(op, values, base_var, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableDimShuffle`."""
    (value,) = values

    # Reverse the effects of dimshuffle on the value variable
    # First, drop any augmented dimensions and reinsert any dropped dimensions
    undo_ds: List[Union[int, str]] = [i for i, o in enumerate(op.new_order) if o != "x"]
    dropped_dims = tuple(sorted(set(op.transposition) - set(op.shuffle)))
    for dropped_dim in dropped_dims:
        undo_ds.insert(dropped_dim, "x")
    value = value.dimshuffle(undo_ds)

    # Then, unshuffle remaining dims
    original_shuffle = list(op.shuffle)
    for dropped_dim in dropped_dims:
        original_shuffle.insert(dropped_dim, dropped_dim)
    undo_ds = [original_shuffle.index(i) for i in range(len(original_shuffle))]
    value = value.dimshuffle(undo_ds)

    raw_logp = logprob(base_var, value)

    # Re-apply original dimshuffle, ignoring any support dimensions consumed by
    # the logprob function. This assumes that support dimensions are always in
    # the rightmost positions, and all we need to do is to discard the highest
    # indexes in the original dimshuffle order. Otherwise, there is no way of
    # knowing which dimensions were consumed by the logprob function.
    redo_ds = [o for o in op.new_order if o == "x" or o < raw_logp.ndim]
    return raw_logp.dimshuffle(redo_ds)


@node_rewriter([DimShuffle])
def find_measurable_dimshuffles(fgraph, node) -> Optional[List[MeasurableDimShuffle]]:
    r"""Finds `Dimshuffle`\s for which a `logprob` can be computed."""

    if isinstance(node.op, MeasurableDimShuffle):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(
        fgraph, "preserve_rv_mappings", None
    )

    if rv_map_feature is None:
        return None  # pragma: no cover

    base_var = node.inputs[0]

    # We can only apply this rewrite directly to `RandomVariable`s, as those are
    # the only `Op`s for which we always know the support axis. Other measurable
    # variables can have arbitrary support axes (e.g., if they contain separate
    # `MeasurableDimShuffle`s). Most measurable variables with `DimShuffle`s
    # should still be supported as long as the `DimShuffle`s can be merged/
    # lifted towards the base RandomVariable.
    # TODO: If we include the support axis as meta information in each
    # intermediate MeasurableVariable, we can lift this restriction.
    if not (
        base_var.owner
        and isinstance(base_var.owner.op, RandomVariable)
        and base_var not in rv_map_feature.rv_values
    ):
        return None  # pragma: no cover

    # Make base_vars unmeasurable
    base_var = assign_custom_measurable_outputs(base_var.owner)

    measurable_dimshuffle = MeasurableDimShuffle(
        node.op.input_broadcastable, node.op.new_order
    )(base_var)
    measurable_dimshuffle.name = node.outputs[0].name

    return [measurable_dimshuffle]


measurable_ir_rewrites_db.register(
    "dimshuffle_lift", local_dimshuffle_rv_lift, -5, "basic", "tensor"
)


# We register this later than `dimshuffle_lift` so that it is only applied as a fallback
measurable_ir_rewrites_db.register(
    "find_measurable_dimshuffles", find_measurable_dimshuffles, 0, "basic", "tensor"
)


measurable_ir_rewrites_db.register(
    "broadcast_to_lift", naive_bcast_rv_lift, -5, "basic", "tensor"
)


measurable_ir_rewrites_db.register(
    "find_measurable_stacks",
    find_measurable_stacks,
    0,
    "basic",
    "tensor",
)
