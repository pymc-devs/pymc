from typing import List, Optional

import aesara.tensor as at
from aesara.graph.rewriting.basic import node_rewriter
from aesara.tensor.extra_ops import CumOp

from aeppl.abstract import MeasurableVariable, assign_custom_measurable_outputs
from aeppl.logprob import _logprob, logprob
from aeppl.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


class MeasurableCumsum(CumOp):
    """A placeholder used to specify a log-likelihood for a cumsum sub-graph."""


MeasurableVariable.register(MeasurableCumsum)


@_logprob.register(MeasurableCumsum)
def logprob_cumsum(op, values, base_rv, **kwargs):
    """Compute the log-likelihood graph for a `Cumsum`."""
    (value,) = values

    value_diff = at.diff(value, axis=op.axis)
    value_diff = at.concatenate(
        (
            # Take first element of axis and add a broadcastable dimension so
            # that it can be concatenated with the rest of value_diff
            at.shape_padaxis(
                at.take(value, 0, axis=op.axis),
                axis=op.axis,
            ),
            value_diff,
        ),
        axis=op.axis,
    )

    cumsum_logp = logprob(base_rv, value_diff)

    return cumsum_logp


@node_rewriter([CumOp])
def find_measurable_cumsums(fgraph, node) -> Optional[List[MeasurableCumsum]]:
    r"""Finds `Cumsums`\s for which a `logprob` can be computed."""

    if not (isinstance(node.op, CumOp) and node.op.mode == "add"):
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableCumsum):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(
        fgraph, "preserve_rv_mappings", None
    )

    if rv_map_feature is None:
        return None  # pragma: no cover

    rv = node.outputs[0]

    base_rv = node.inputs[0]
    if not (
        base_rv.owner
        and isinstance(base_rv.owner.op, MeasurableVariable)
        and base_rv not in rv_map_feature.rv_values
    ):
        return None  # pragma: no cover

    # Check that cumsum does not mix dimensions
    if base_rv.ndim > 1 and node.op.axis is None:
        return None

    new_op = MeasurableCumsum(axis=node.op.axis or 0, mode="add")
    # Make base_var unmeasurable
    unmeasurable_base_rv = assign_custom_measurable_outputs(base_rv.owner)
    new_rv = new_op.make_node(unmeasurable_base_rv).default_output()
    new_rv.name = rv.name

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_cumsums",
    find_measurable_cumsums,
    0,
    "basic",
    "cumsum",
)
