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
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor import eq
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneTypeT

from pymc.logprob.abstract import MeasurableOp, _logprob, _logprob_helper
from pymc.logprob.checks import MeasurableCheckAndRaise
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    check_potential_measurability,
    dirac_delta,
    filter_measurable_variables,
)


class MeasurableSetSubtensor(IncSubtensor, MeasurableOp):
    """Measurable SetSubtensor Op."""

    def __str__(self):
        return f"Measurable{super().__str__()}"


class MeasurableAdvancedSetSubtensor(AdvancedIncSubtensor, MeasurableOp):
    """Measurable AdvancedSetSubtensor Op."""

    def __str__(self):
        return f"Measurable{super().__str__()}"


set_subtensor_does_not_broadcast = MeasurableCheckAndRaise(
    exc_type=NotImplementedError,
    msg="Measurable SetSubtensor not supported when set value is broadcasted.",
)


@node_rewriter(tracks=[IncSubtensor, AdvancedIncSubtensor1, AdvancedIncSubtensor])
def find_measurable_set_subtensor(fgraph, node) -> list | None:
    """Find `SetSubtensor` for which a `logprob` can be computed."""
    if isinstance(node.op, MeasurableOp):
        return None

    if not node.op.set_instead_of_inc:
        return None

    x, y, *idx_elements = node.inputs

    measurable_inputs = filter_measurable_variables([x, y])

    if y not in measurable_inputs:
        return None

    if x not in measurable_inputs:
        # x is potentially measurable, wait for it's logprob IR to be inferred
        if check_potential_measurability([x]):
            return None
        # x has no link to measurable variables, so it's value should be constant
        else:
            x = dirac_delta(x, rtol=0, atol=0)

    if check_potential_measurability(idx_elements):
        return None

    measurable_class: type[MeasurableSetSubtensor | MeasurableAdvancedSetSubtensor]
    if isinstance(node.op, IncSubtensor):
        measurable_class = MeasurableSetSubtensor
        idx = indices_from_subtensor(idx_elements, node.op.idx_list)
    else:
        measurable_class = MeasurableAdvancedSetSubtensor
        idx = tuple(idx_elements)

    # Check that y is not certainly broadcasted.
    indexed_block = x[idx]
    missing_y_dims = indexed_block.type.ndim - y.type.ndim
    y_bcast = [True] * missing_y_dims + list(y.type.broadcastable)
    if any(
        y_dim_bcast and indexed_block_dim_len not in (None, 1)
        for y_dim_bcast, indexed_block_dim_len in zip(
            y_bcast, indexed_block.type.shape, strict=True
        )
    ):
        return None

    measurable_set_subtensor = measurable_class(**node.op._props_dict())(x, y, *idx_elements)

    # Often with indexing we don't know the static shape of the indexed block.
    # And, what's more, the indexing operations actually support runtime broadcasting.
    # As the logp is not valid under broadcasting, we have to add a runtime check.
    # This will hopefully be removed during shape inference when not violated.
    potential_broadcasted_dims = [
        i
        for i, (y_bcast_dim, indexed_block_dim_len) in enumerate(
            zip(y_bcast, indexed_block.type.shape)
        )
        if y_bcast_dim and indexed_block_dim_len is None
    ]
    if potential_broadcasted_dims:
        indexed_block_shape = tuple(indexed_block.shape)
        measurable_set_subtensor = set_subtensor_does_not_broadcast(
            measurable_set_subtensor,
            *(eq(indexed_block_shape[i], 1) for i in potential_broadcasted_dims),
        )

    return [measurable_set_subtensor]


measurable_ir_rewrites_db.register(
    find_measurable_set_subtensor.__name__,
    find_measurable_set_subtensor,
    "basic",
    "set_subtensor",
)


def indexed_dims(idx) -> list[int | None]:
    """Return the indices of the dimensions of the indexed tensor that are being indexed."""
    dims: list[int | None] = []
    idx_counter = 0
    for idx_elem in idx:
        if isinstance(idx_elem, Variable) and isinstance(idx_elem.type, NoneTypeT):
            # None in indexes correspond to newaxis, and don't map to any existing dimension
            dims.append(None)

        elif (
            isinstance(idx_elem, Variable)
            and isinstance(idx_elem.type, TensorType)
            and idx_elem.type.dtype == "bool"
        ):
            # Boolean indexes map to as many dimensions as the mask has
            for i in range(idx_elem.type.ndim):
                dims.append(idx_counter)
                idx_counter += 1
        else:
            dims.append(idx_counter)
            idx_counter += 1

    return dims


@_logprob.register(MeasurableSetSubtensor)
@_logprob.register(MeasurableAdvancedSetSubtensor)
def logprob_setsubtensor(op, values, x, y, *idx_elements, **kwargs):
    """Compute the log-likelihood graph for a `SetSubtensor`.

    For a generative graph like:
        o = zeros(2)
        x = o[0].set(X)
        y = x[1].set(Y)

    The log-likelihood graph is:
        logp(y, value) = (
            logp(x, value)
            [1].set(logp(y, value[1]))
        )

    Unrolling the logp(x, value) gives:
        logp(y, value) = (
            DiracDelta(zeros(2), value)  # Irrelevant if all entries are set
            [0].set(logp(x, value[0]))
            [1].set(logp(y, value[1]))
        )
    """
    [value] = values
    if isinstance(op, MeasurableSetSubtensor):
        # For basic indexing we have to recreate the index from the input list
        idx = indices_from_subtensor(idx_elements, op.idx_list)
    else:
        # For advanced indexing we can use the idx_elements directly
        idx = tuple(idx_elements)

    x_logp = _logprob_helper(x, value)
    y_logp = _logprob_helper(y, value[idx])

    y_ndim_supp = x[idx].type.ndim - y_logp.type.ndim
    x_ndim_supp = x.type.ndim - x_logp.type.ndim
    ndim_supp = max(y_ndim_supp, x_ndim_supp)
    if ndim_supp > 0:
        # Multivariate logp only valid if we are not doing indexing along the reduced dimensions
        # Otherwise we don't know if successive writings are overlapping or not
        core_dims = set(range(x.type.ndim)[-ndim_supp:])
        if set(indexed_dims(idx)) & core_dims:
            # When we have IR meta-info about support_ndim, we can fail at the rewriting stage
            raise NotImplementedError(
                "Indexing along core dimensions of multivariate SetSubtensor not supported"
            )

        ndim_supp_diff = y_ndim_supp - x_ndim_supp
        if ndim_supp_diff > 0:
            # In this case y_logp will have fewer dimensions than x_logp after indexing, so we need to reduce x before indexing.
            x_logp = x_logp.sum(axis=tuple(range(-ndim_supp_diff, 0)))
        elif ndim_supp_diff < 0:
            # In this case x_logp will have fewer dimensions than y_logp after indexing, so we need to reduce y before indexing.
            y_logp = y_logp.sum(axis=tuple(range(ndim_supp_diff, 0)))

    out_logp = x_logp[idx].set(y_logp)
    return out_logp
