from typing import List, Optional

import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Node
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import node_rewriter
from aesara.scalar.basic import Clip
from aesara.scalar.basic import clip as scalar_clip
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.var import TensorConstant

from aeppl.abstract import MeasurableVariable, assign_custom_measurable_outputs
from aeppl.logprob import CheckParameterValue, _logcdf, _logprob
from aeppl.rewriting import measurable_ir_rewrites_db


class MeasurableClip(Elemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""


MeasurableVariable.register(MeasurableClip)


@node_rewriter(tracks=[Elemwise])
def find_measurable_clips(
    fgraph: FunctionGraph, node: Node
) -> Optional[List[MeasurableClip]]:
    # TODO: Canonicalize x[x>ub] = ub -> clip(x, x, ub)

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableClip):
        return None  # pragma: no cover

    if not (isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, Clip)):
        return None

    clipped_var = node.outputs[0]
    base_var, lower_bound, upper_bound = node.inputs

    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and base_var not in rv_map_feature.rv_values
    ):
        return None

    # Replace bounds by `+-inf` if `y = clip(x, x, ?)` or `y=clip(x, ?, x)`
    # This is used in `clip_logprob` to generate a more succint logprob graph
    # for one-sided clipped random variables
    lower_bound = lower_bound if (lower_bound is not base_var) else at.constant(-np.inf)
    upper_bound = upper_bound if (upper_bound is not base_var) else at.constant(np.inf)

    clipped_op = MeasurableClip(scalar_clip)
    # Make base_var unmeasurable
    unmeasurable_base_var = assign_custom_measurable_outputs(base_var.owner)
    clipped_rv_node = clipped_op.make_node(
        unmeasurable_base_var, lower_bound, upper_bound
    )
    clipped_rv = clipped_rv_node.outputs[0]

    clipped_rv.name = clipped_var.name

    return [clipped_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_clips",
    find_measurable_clips,
    0,
    "basic",
    "censoring",
)


@_logprob.register(MeasurableClip)
def clip_logprob(op, values, base_rv, lower_bound, upper_bound, **kwargs):
    r"""Logprob of a clipped censored distribution

    The probability is given by
    .. math::
        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower, dist) & \text{for } x = lower, \\
            \text{P}(x, dist) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper, dist) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    """
    (value,) = values

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logprob = _logprob(base_rv_op, (value,), *base_rv_inputs, **kwargs)
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    is_lower_bounded, is_upper_bounded = False, False
    if not (
        isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))
    ):
        is_upper_bounded = True

        logccdf = at.log1mexp(logcdf)
        # For right clipped discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv.dtype.startswith("int"):
            logccdf = at.logaddexp(logccdf, logprob)

        logprob = at.switch(
            at.eq(value, upper_bound),
            logccdf,
            at.switch(at.gt(value, upper_bound), -np.inf, logprob),
        )
    if not (
        isinstance(lower_bound, TensorConstant)
        and np.all(np.isneginf(lower_bound.value))
    ):
        is_lower_bounded = True
        logprob = at.switch(
            at.eq(value, lower_bound),
            logcdf,
            at.switch(at.lt(value, lower_bound), -np.inf, logprob),
        )

    if is_lower_bounded and is_upper_bounded:
        logprob = CheckParameterValue("lower_bound <= upper_bound")(
            logprob, at.all(at.le(lower_bound, upper_bound))
        )

    return logprob
