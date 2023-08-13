#   Copyright 2023 The PyMC Developers
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
from typing import Optional

import numpy as np
import pytensor.tensor as pt

from pytensor.graph import node_rewriter
from pytensor.tensor.extra_ops import BroadcastTo

from pymc.logprob.abstract import MeasurableVariable, _logprob, _logprob_helper
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


class MeasurableBroadcast(BroadcastTo):
    pass


MeasurableVariable.register(MeasurableBroadcast)


measurable_broadcast = MeasurableBroadcast()


@_logprob.register(MeasurableBroadcast)
def broadcast_logprob(op, values, rv, *shape, **kwargs):
    """Log-probability expression for (statically-)broadcasted RV

    The probability is the same as the base RV, if no broadcasting had happened:

    ``logp(broadcast_to(normal(size=(3, 1)), (2, 3, 4)), zeros((2, 3, 4))) == logp(normal(size=(3, 1)), zeros((3, 1)))``

    And zero if the value couldn't have possibly originated via broadcasting:

    ``logp(broadcast_to(normal(size=(1,)), (3,)), [1, 2, 3]) == [-np.inf]``

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
    # All entries in the broadcasted dimensions should be the same, so we simply select the first of each.
    indices = []
    for i in range(value.ndim):
        # Remove expanded dims
        if i in expanded_dims:
            indices.append(0)
        # Keep first entry of broadcasted (but not expanded) dims
        elif i in broadcast_dims:
            indices.append(slice(0, 1))
        else:
            indices.append(slice(None))

    unbroadcast_value = value[tuple(indices)]
    logp = _logprob_helper(rv, unbroadcast_value)

    # Check that dependent values were indeed identical, by comparing with a re-broadcasted value
    valid_value = pt.broadcast_to(unbroadcast_value, shape)
    # Note: This could fail due to float-precision issues.
    # If that proves to be a problem we should switch to `pt.allclose`
    check = pt.all(pt.eq(value, valid_value))
    logp = pt.switch(check, logp, -np.inf)

    # Reintroduce expanded_dims in the returned logp
    if n_new_dims > 0:
        logp = pt.shape_padleft(logp, n_new_dims)

    return logp


@node_rewriter([BroadcastTo])
def find_measurable_broadcast(fgraph, node):
    r"""Finds `BroadcastTo`\s for which a `logprob` can be computed."""

    if isinstance(node.op, MeasurableBroadcast):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    base_rv, *shape = node.inputs

    if not rv_map_feature.request_measurable([base_rv]):
        return None

    new_rv = measurable_broadcast.make_node(base_rv, *shape).default_output()

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_broadcast",
    find_measurable_broadcast,
    "basic",
    "shape",
)
