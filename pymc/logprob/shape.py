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
import numpy as np
import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Alloc

from pymc.logprob.abstract import MeasurableOp, _logprob, _logprob_helper
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    check_potential_measurability,
    filter_measurable_variables,
    get_related_valued_nodes,
)


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


measurable_ir_rewrites_db.register(
    "find_measurable_broadcast",
    find_measurable_broadcast,
    "basic",
    "shape",
)
