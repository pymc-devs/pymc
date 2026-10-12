#   Copyright 2024 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
"""Materialize density metadata on intermediate tensor operations."""

from functools import cache

from pytensor.graph import Apply
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.type import RandomType

from pymc.logprob.abstract import (
    DensityMetadata,
    LogprobQuery,
    MeasurableOp,
    MeasurableTensorOp,
    ValuedRV,
)
from pymc.logprob.query import (
    UnsupportedObservation,
    contains_random,
    infer_density_ndim,
    infer_support_axes,
    rewrite_logprob_query,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db


@cache
def measurable_type(native_type):
    return type(
        f"Measurable{native_type.__name__}",
        (MeasurableTensorOp, native_type),
        {
            "native_type": native_type,
            "__props__": ("base_op", "metadata"),
            "__module__": __name__,
        },
    )


def native_op(op):
    return op.base_op


def leads_to_observation(fgraph, node):
    """Whether a node is on a source path, stopping at density/value expressions."""
    pending = list(node.outputs)
    seen = set()
    while pending:
        var = pending.pop()
        if var in seen:
            continue
        seen.add(var)
        for client, index in fgraph.clients.get(var, ()):
            if isinstance(client.op, ValuedRV):
                if index == 0:
                    return True
            elif not isinstance(client.op, LogprobQuery):
                pending.extend(client.outputs)
    return False


def measurable_node(node):
    """Recognize an already normalized native node and construct its annotated Op."""
    native = node.op
    if isinstance(native, ValuedRV | LogprobQuery | MeasurableOp):
        return node
    selection = None
    if isinstance(native, Elemwise):
        from pytensor.scalar import ScalarOp

        from pymc.logprob.transforms import select_transform

        rule = rewrite_logprob_query.dispatch(type(native.scalar_op))
        if rule is rewrite_logprob_query.dispatch(ScalarOp):
            selection = select_transform(native.scalar_op, node.inputs)
            if selection is None:
                return node
    if infer_support_axes.dispatch(type(native)) is infer_support_axes.dispatch(object):
        return node
    axes, ranks, recognized = [], [], []
    for out in node.outputs:
        if isinstance(out.type, RandomType):
            axes.append(None)
            ranks.append(None)
            recognized.append(False)
            continue
        random = contains_random(out)
        if not random:
            axes.append(())
            ranks.append(out.ndim)
            recognized.append(False)
            continue
        try:
            support = infer_support_axes(native, out)
            # Compute the default rank from this freshly inferred support layout.
            rank_rule = infer_density_ndim.dispatch(type(native))
            if rank_rule is infer_density_ndim.dispatch(object):
                rank = None if support is None else out.ndim - len(support)
            else:
                rank = rank_rule(native, out)
        except UnsupportedObservation:
            support, rank = None, None
        axes.append(None if support is None else tuple(axis - out.ndim for axis in support))
        ranks.append(rank)
        recognized.append(bool(random) if rank is not None else None)
    # An alternative Op owns final facts for all of its stochastic outputs.
    # Do not freeze a multi-output node while another output still needs a
    # conditioning value to establish its density layout.
    if not any(recognized) or any(state is None for state in recognized):
        return node
    transform = None
    measured_input = None
    if selection is not None:
        base, transform, _ = selection
        measured_input = node.inputs.index(base)
    metadata = DensityMetadata(
        tuple(axes),
        tuple(ranks),
        tuple(recognized),
        transform,
        measured_input,
        tuple(getattr(out, "ndim", None) for out in node.outputs),
    )
    measurable = measurable_type(type(native))(native, metadata)
    # Native make_node has already normalized dtype and axes. Construction of
    # the alternative Op owns the metadata and cannot invoke a native clone.
    new_node = Apply(measurable, list(node.inputs), [out.type() for out in node.outputs])
    return new_node


@node_rewriter(None)
def find_measurable_tensor(fgraph, node):
    op = node.op
    if isinstance(op, ValuedRV | LogprobQuery):
        return None
    if isinstance(op, MeasurableOp):
        return None
    if any(isinstance(n.op, ValuedRV) for n in fgraph.apply_nodes) and not leads_to_observation(
        fgraph, node
    ):
        return None
    new_node = measurable_node(node)
    if new_node.op == op and all(
        new.type == old.type for new, old in zip(new_node.outputs, node.outputs, strict=True)
    ):
        return None
    return new_node.outputs


@node_rewriter(None)
def remove_measurable_tensor(fgraph, node):
    if isinstance(node.op, MeasurableTensorOp):
        return native_op(node.op).make_node(*node.inputs).outputs
    return None


measurable_ir_rewrites_db.register("find_measurable_tensor", find_measurable_tensor, "basic")
