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
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import (
    Ceil,
    Clip,
    Floor,
    Maximum,
    Minimum,
    RoundHalfAwayFromZero,
    RoundHalfToEven,
    Trunc,
)
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.rewriting.basic import elemwise_of
from pytensor.tensor.variable import TensorConstant

from pymc.logprob.abstract import (
    _icdf,
    _icdf_helper,
    _logccdf_helper,
    _logcdf,
    _logcdf_helper,
    logprob_query,
)
from pymc.logprob.query import (
    bind_value,
    contains_random,
    density_sources,
    other_query_uses,
    query_parts,
    rewrite_logprob_query,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    CheckParameterValue,
    filter_measurable_variables,
)

ROUNDING_OPS = (RoundHalfToEven, RoundHalfAwayFromZero, Floor, Ceil, Trunc)


@node_rewriter([elemwise_of(Clip)])
def find_measurable_clips(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    # TODO: Canonicalize x[x>ub] = ub -> clip(x, x, ub)

    lower_bound: Variable | None
    upper_bound: Variable | None
    base_var, lower_bound, upper_bound = node.inputs

    # Clipping a constant by measurable bounds is an order statistic, not censoring
    if not filter_measurable_variables([base_var]):
        return None

    # A bound that is the clipped variable itself declares one-sided clipping
    # (`y = clip(x, x, ub)`), as do literal +-inf constants
    if lower_bound is base_var or (
        isinstance(lower_bound, TensorConstant) and np.all(np.isneginf(lower_bound.value))
    ):
        lower_bound = None
    if upper_bound is base_var or (
        isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))
    ):
        upper_bound = None

    # Fuse clips of clips into a single clip, so that e.g. two-sided censoring written
    # as maximum(minimum(x, ub), lb) becomes one node. Mass pooled at an inner bound
    # either stays there when it lies inside the outer interval, or is re-pooled at
    # the outer bound, so the bounds combine with maximum/minimum (constant bounds
    # fold; crossed bounds are caught by the CheckParameterValue in the logprob)
    if isinstance(base_var.owner_op, Elemwise) and isinstance(base_var.owner.op.scalar_op, Clip):
        inner_base, inner_lower, inner_upper = base_var.owner.inputs
        if inner_lower is not inner_base:
            lower_bound = (
                inner_lower if lower_bound is None else pt.maximum(lower_bound, inner_lower)
            )
        if inner_upper is not inner_base:
            upper_bound = (
                inner_upper if upper_bound is None else pt.minimum(upper_bound, inner_upper)
            )
        base_var = inner_base

    # The variable itself is used as the unbounded sentinel because, unlike +-inf
    # constants, it does not upcast discrete variables
    if lower_bound is None:
        lower_bound = base_var
    if upper_bound is None:
        upper_bound = base_var

    clipped_rv = pt.clip(base_var, lower_bound, upper_bound)
    # Dropping an infinite bound also drops the upcast it forced on the original clip,
    # which must be restored for the replacement to be type-compatible.
    # The cast is left plain so that `find_measurable_casts` is the one to claim it:
    # a discrete base hidden behind a float cast is only measurable when directly valued
    out_dtype = node.outputs[0].type.dtype
    if clipped_rv.type.dtype != out_dtype:
        clipped_rv = pt.cast(clipped_rv, out_dtype)
    if (
        base_var is node.inputs[0]
        and lower_bound is node.inputs[1]
        and upper_bound is node.inputs[2]
    ):
        return None
    return [clipped_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_clips",
    find_measurable_clips,
    "basic",
    "censoring",
)


@node_rewriter([elemwise_of(Maximum), elemwise_of(Minimum)])
def measurable_max_min_to_clip(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    """Convert one-sided censoring maximum(x, c) and minimum(x, c) to clip form.

    The unbounded side uses the measurable variable itself, which `find_measurable_clips`
    understands as one-sided clipping and which preserves the dtype of discrete
    variables. Two-sided censoring maximum(minimum(x, ub), lb) is fused into a single
    two-sided clip by `find_measurable_clips`.
    """
    measurable_inputs = filter_measurable_variables(node.inputs)
    if len(measurable_inputs) != 1:
        return None

    [measurable_input] = measurable_inputs
    [other_input] = [inp for inp in node.inputs if inp is not measurable_input]

    if isinstance(node.op.scalar_op, Maximum):
        return [pt.clip(measurable_input, other_input, measurable_input)]
    else:
        return [pt.clip(measurable_input, measurable_input, other_input)]


measurable_ir_rewrites_db.register(
    "measurable_max_min_to_clip",
    measurable_max_min_to_clip,
    "basic",
    "censoring",
)


@rewrite_logprob_query.register(Clip)
def rewrite_clip_logprob(op, fgraph, query, **kwargs):
    r"""Logprob of a clipped censored distribution.

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
    rv, value = query_parts(query)
    base_rv, lower_bound, upper_bound = rv.owner.inputs
    if not contains_random(base_rv) or any(
        bound is not base_rv and contains_random(bound) for bound in (lower_bound, upper_bound)
    ):
        return None

    if other_query_uses(fgraph, density_sources(base_rv), {query}):
        return None

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logprob = logprob_query(bind_value(fgraph, base_rv, value))
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    if upper_bound is not base_rv:
        logccdf = _logccdf_helper(base_rv, value)

        # For right clipped discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv.dtype.startswith("int"):
            logccdf = pt.logaddexp(logccdf, logprob)

        logprob = pt.switch(
            pt.eq(value, upper_bound),
            logccdf,
            pt.switch(pt.gt(value, upper_bound), -np.inf, logprob),
        )
    if lower_bound is not base_rv:
        logprob = pt.switch(
            pt.eq(value, lower_bound),
            logcdf,
            pt.switch(pt.lt(value, lower_bound), -np.inf, logprob),
        )

    if lower_bound is not base_rv and upper_bound is not base_rv:
        logprob = CheckParameterValue("lower_bound <= upper_bound")(
            logprob, pt.all(pt.le(lower_bound, upper_bound))
        )

    return [logprob]


@_logcdf.register(Clip)
def clip_logcdf(op, value, base_rv, lower_bound, upper_bound):
    r"""Log-CDF of a clipped censored distribution.

    .. math::
        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(x, dist) & \text{for } lower <= x < upper, \\
            1 & \text{for } x >= upper,
        \end{cases}
    """
    logcdf = _logcdf_helper(base_rv, value)

    if upper_bound is not base_rv:
        logcdf = pt.switch(pt.ge(value, upper_bound), 0.0, logcdf)
    if lower_bound is not base_rv:
        logcdf = pt.switch(pt.lt(value, lower_bound), -np.inf, logcdf)

    if lower_bound is not base_rv and upper_bound is not base_rv:
        logcdf = CheckParameterValue("lower_bound <= upper_bound")(
            logcdf, pt.all(pt.le(lower_bound, upper_bound))
        )

    return logcdf


@_icdf.register(Clip)
def clip_icdf(op, value, base_rv, lower_bound, upper_bound):
    # The point masses at the bounds absorb the respective tail quantiles
    icdf = _icdf_helper(base_rv, value)

    if lower_bound is not base_rv:
        icdf = pt.maximum(icdf, lower_bound)
    if upper_bound is not base_rv:
        icdf = pt.minimum(icdf, upper_bound)

    if lower_bound is not base_rv and upper_bound is not base_rv:
        icdf = CheckParameterValue("lower_bound <= upper_bound")(
            icdf, pt.all(pt.le(lower_bound, upper_bound))
        )

    return icdf


@node_rewriter([elemwise_of(op) for op in ROUNDING_OPS])
def find_measurable_roundings(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    if not filter_measurable_variables(node.inputs):
        return None

    [base_var] = node.inputs

    if np.dtype(base_var.type.dtype).kind != "f" or (
        isinstance(base_var.owner_op, Elemwise)
        and isinstance(base_var.owner.op.scalar_op, ROUNDING_OPS)
    ):
        # The base already sits on the integers, either because its dtype guarantees it
        # or because it is itself a rounding, so this one leaves it untouched save for
        # the upcast the Ops apply to a discrete input. Reducing the rounding to that
        # cast, rather than deriving the mass of a degenerate cell, leaves the base's
        # own logprob, logcdf and icdf to apply. The cast folds away when the base is
        # already a float, as it is for a rounded base.
        return [pt.cast(base_var, node.outputs[0].type.dtype)]

    return None


measurable_ir_rewrites_db.register(
    "find_measurable_roundings",
    find_measurable_roundings,
    "basic",
    "censoring",
)


@rewrite_logprob_query.register(RoundHalfToEven)
@rewrite_logprob_query.register(RoundHalfAwayFromZero)
@rewrite_logprob_query.register(Floor)
@rewrite_logprob_query.register(Ceil)
@rewrite_logprob_query.register(Trunc)
def rewrite_round_logprob(op, fgraph, query, **kwargs):
    r"""Logprob of a rounded censored distribution.

    The probability of a distribution rounded to the nearest integer is given by
    .. math::
        \begin{cases}
            \text{CDF}(x+\frac{1}{2}, dist) - \text{CDF}(x-\frac{1}{2}, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    The probability of a distribution rounded up is given by
    .. math::
        \begin{cases}
            \text{CDF}(x, dist) - \text{CDF}(x-1, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    The probability of a distribution rounded down is given by
    .. math::
        \begin{cases}
            \text{CDF}(x+1, dist) - \text{CDF}(x, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    The probability of a distribution rounded towards zero is given by the
    rounded-down probability for positive values and the rounded-up probability
    for negative values, with both intervals pooled at zero.

    """
    rv, value = query_parts(query)
    (base_rv,) = rv.owner.inputs
    if not isinstance(base_rv.owner_op, RandomVariable) or base_rv.dtype[0] != "f":
        return None
    if other_query_uses(fgraph, [base_rv], {query}):
        return None

    if not value.type.dtype.startswith("float"):
        # The Ops below snap the value onto the cell whose mass it asks for, and only
        # accept float inputs. An integral value already sits on a cell, so the snapping
        # is an identity and only the dtype needs adjusting (as when the rounding is
        # followed by a cast to int).
        value = pt.cast(value, pytensor.config.floatX)

    if isinstance(op, RoundHalfToEven | RoundHalfAwayFromZero):
        # The tie-breaking rule only matters on a measure-zero set of the
        # continuous base variable, so both variants share the same intervals
        value = pt.round(value)
        value_upper = value + 0.5
        value_lower = value - 0.5
    elif isinstance(op, Floor):
        value = pt.floor(value)
        value_upper = value + 1.0
        value_lower = value
    elif isinstance(op, Ceil):
        value = pt.ceil(value)
        value_upper = value
        value_lower = value - 1.0
    elif isinstance(op, Trunc):
        # Truncation rounds towards zero: [x, x+1) for x >= 0, (x-1, x] for x < 0,
        # and (-1, 1) for x == 0 (open/closed bounds are equivalent for the
        # continuous base variables this rewrite applies to)
        value = pt.trunc(value)
        value_upper = value + (value >= 0)
        value_lower = value - (value <= 0)
    else:
        raise TypeError(f"Unsupported scalar_op {op}")  # pragma: no cover

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logcdf_upper = _logcdf(base_rv_op, value_upper, *base_rv_inputs)
    logcdf_lower = _logcdf(base_rv_op, value_lower, *base_rv_inputs)

    if base_rv_op.name:
        logcdf_upper.name = f"{base_rv_op}_logcdf_upper"
        logcdf_lower.name = f"{base_rv_op}_logcdf_lower"

    # TODO: Figure out better solution to avoid this circular import
    from pymc.math import logdiffexp

    return [logdiffexp(logcdf_upper, logcdf_lower)]
