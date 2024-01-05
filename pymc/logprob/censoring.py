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

from typing import Optional

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph import Op
from pytensor.graph.basic import Apply, Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.raise_op import Assert
from pytensor.scalar.basic import (
    GE,
    GT,
    LE,
    LT,
    Ceil,
    Clip,
    Floor,
    RoundHalfToEven,
    Switch,
)
from pytensor.scalar.basic import clip as scalar_clip

from pytensor.tensor import TensorVariable
from pytensor.tensor import TensorType
from pytensor.tensor.basic import switch as switch
from pytensor.tensor.math import ceil, clip, floor, round_half_to_even
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorConstant

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    _logcdf,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db
from pymc.logprob.utils import CheckParameterValue, check_potential_measurability


class MeasurableClip(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""

    valid_scalar_types = (Clip,)


measurable_clip = MeasurableClip(scalar_clip)


@node_rewriter(tracks=[clip])
def find_measurable_clips(fgraph: FunctionGraph, node: Node) -> Optional[list[TensorVariable]]:
    # TODO: Canonicalize x[x>ub] = ub -> clip(x, x, ub)

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if not rv_map_feature.request_measurable(node.inputs):
        return None

    base_var, lower_bound, upper_bound = node.inputs

    # Replace bounds by `+-inf` if `y = clip(x, x, ?)` or `y=clip(x, ?, x)`
    # This is used in `clip_logprob` to generate a more succinct logprob graph
    # for one-sided clipped random variables
    lower_bound = lower_bound if (lower_bound is not base_var) else pt.constant(-np.inf)
    upper_bound = upper_bound if (upper_bound is not base_var) else pt.constant(np.inf)

    clipped_rv = measurable_clip.make_node(base_var, lower_bound, upper_bound).outputs[0]
    return [clipped_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_clips",
    find_measurable_clips,
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
    if not (isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))):
        is_upper_bounded = True

        logccdf = pt.log1mexp(logcdf)
        # For right clipped discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv.dtype.startswith("int"):
            logccdf = pt.logaddexp(logccdf, logprob)

        logprob = pt.switch(
            pt.eq(value, upper_bound),
            logccdf,
            pt.switch(pt.gt(value, upper_bound), -np.inf, logprob),
        )
    if not (isinstance(lower_bound, TensorConstant) and np.all(np.isneginf(lower_bound.value))):
        is_lower_bounded = True
        logprob = pt.switch(
            pt.eq(value, lower_bound),
            logcdf,
            pt.switch(pt.lt(value, lower_bound), -np.inf, logprob),
        )

    if is_lower_bounded and is_upper_bounded:
        logprob = CheckParameterValue("lower_bound <= upper_bound")(
            logprob, pt.all(pt.le(lower_bound, upper_bound))
        )

    return logprob


class MeasurableRound(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""

    valid_scalar_types = (RoundHalfToEven, Floor, Ceil)


@node_rewriter(tracks=[ceil, floor, round_half_to_even])
def find_measurable_roundings(fgraph: FunctionGraph, node: Node) -> Optional[list[TensorVariable]]:
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if not rv_map_feature.request_measurable(node.inputs):
        return None

    [base_var] = node.inputs
    rounded_op = MeasurableRound(node.op.scalar_op)
    rounded_rv = rounded_op.make_node(base_var).default_output()
    rounded_rv.name = node.outputs[0].name
    return [rounded_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_roundings",
    find_measurable_roundings,
    "basic",
    "censoring",
)


@_logprob.register(MeasurableRound)
def round_logprob(op, values, base_rv, **kwargs):
    r"""Logprob of a rounded censored distribution

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

    """
    (value,) = values

    if isinstance(op.scalar_op, RoundHalfToEven):
        value = pt.round(value)
        value_upper = value + 0.5
        value_lower = value - 0.5
    elif isinstance(op.scalar_op, Floor):
        value = pt.floor(value)
        value_upper = value + 1.0
        value_lower = value
    elif isinstance(op.scalar_op, Ceil):
        value = pt.ceil(value)
        value_upper = value
        value_lower = value - 1.0
    else:
        raise TypeError(f"Unsupported scalar_op {op.scalar_op}")  # pragma: no cover

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logcdf_upper = _logcdf(base_rv_op, value_upper, *base_rv_inputs, **kwargs)
    logcdf_lower = _logcdf(base_rv_op, value_lower, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logcdf_upper.name = f"{base_rv_op}_logcdf_upper"
        logcdf_lower.name = f"{base_rv_op}_logcdf_lower"

    # TODO: Figure out better solution to avoid this circular import
    from pymc.math import logdiffexp

    return logdiffexp(logcdf_upper, logcdf_lower)


class FlatSwitches(Op):
    __props__ = ("out_dtype", "rv_idx")

    def __init__(self, *args, out_dtype, rv_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dtype = out_dtype
        self.rv_idx = rv_idx

    def make_node(self, *inputs):
        return Apply(
            self, list(inputs), [TensorType(dtype=self.out_dtype, shape=inputs[0].type.shape)()]
        )

    def perform(self, *args, **kwargs):
        raise NotImplementedError("This Op should not be evaluated")


MeasurableVariable.register(FlatSwitches)


def get_intervals(binary_node, valued_rvs):
    """
    Handles both "x > 1" and "1 < x"  expressions.
    """

    measurable_inputs = [
        inp for inp in binary_node.inputs if check_potential_measurability([inp], valued_rvs)
    ]

    if len(measurable_inputs) != 1:
        return None

    measurable_var = measurable_inputs[0]
    measurable_var_idx = binary_node.inputs.index(measurable_var)

    const = binary_node.inputs[(measurable_var_idx + 1) % 2]

    # whether it is a lower or an upper bound depends on measurable_var_idx and the binary Op.
    is_gt_or_ge = isinstance(binary_node.op.scalar_op, (GT, GE))
    is_lt_or_le = isinstance(binary_node.op.scalar_op, (LT, LE))

    if not is_lt_or_le and not is_gt_or_ge:
        # Switch condition was not defined using binary Ops
        return None

    intervals = [(-np.inf, const), (const, np.inf)]

    # interval_true for the interval corresponds to true branch in 'Switch', interval_false corresponds to false branch
    if measurable_var_idx == 0:
        # e.g. "x < 1" and "x > 1"
        interval_true, interval_false = intervals if is_lt_or_le else intervals[::-1]
    else:
        # e.g. "1 > x" and "1 < x"
        interval_true, interval_false = intervals[::-1] if is_gt_or_ge else intervals

    return [interval_true, interval_false]


def adjust_intervals(intervals, outer_interval):
    for i in range(2):
        current = intervals[i]
        lower = pt.maximum(current[0], outer_interval[0])
        upper = pt.minimum(current[1], outer_interval[1])

        intervals[i] = (lower, upper)

    return intervals


def flat_switch_helper(node, valued_rvs, encoding_list, outer_interval, base_rv):
    """
    Carries out the main recursion through the branches to fetch the encodings, their respective
    intervals and adjust any overlaps. It also performs several checks on the switch condition and measurable
    components.
    """
    from pymc.distributions.distribution import SymbolicRandomVariable

    switch_cond, *components = node.inputs

    # deny broadcasting of the switch condition
    if switch_cond.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    measurable_var_switch = [
        var for var in switch_cond.owner.inputs if check_potential_measurability([var], valued_rvs)
    ]

    if len(measurable_var_switch) != 1:
        return None

    current_base_var = measurable_var_switch[0]
    # deny cases where base_var is some function of 'x', e.g. pt.exp(x), and measurable var in the current switch is x
    # also check if all the sources of measurability are the same RV. e.g. x1 and x2
    if current_base_var is not base_rv:
        return None

    measurable_var_idx = []
    switch_comp_idx = []

    # get the indices for the switch and other measurable components for further recursion
    for idx, component in enumerate(components):
        if check_potential_measurability([component], valued_rvs):
            if isinstance(
                component.owner.op, (RandomVariable, SymbolicRandomVariable)
            ) or not isinstance(component.owner.op.scalar_op, Switch):
                measurable_var_idx.append(idx)
            else:
                switch_comp_idx.append(idx)

    # Check if measurability source and the component itself are the same for all measurable components
    if any(components[i] is not base_rv for i in measurable_var_idx):
        return None

    # Get intervals for true and false components from the condition
    intervals = get_intervals(switch_cond.owner, valued_rvs)
    adjusted_intervals = adjust_intervals(intervals, outer_interval)

    # Base condition for recursion - when there is no more switch in either of the components
    if not switch_comp_idx:
        # Insert the two components and their respective intervals into encoding_list
        for i in range(2):
            switch_dict = {
                "lower": adjusted_intervals[i][0],
                "upper": adjusted_intervals[i][1],
                "encoding": components[i],
            }
            encoding_list.append(switch_dict)

        return encoding_list

    else:
        for i in range(2):
            if i in switch_comp_idx:
                # Recurse through the switch component(es)
                encoding_list = flat_switch_helper(
                    components[i].owner, valued_rvs, encoding_list, adjusted_intervals[i], base_rv
                )

            else:
                switch_dict = {
                    "lower": adjusted_intervals[i][0],
                    "upper": adjusted_intervals[i][1],
                    "encoding": components[i],
                }
                encoding_list.append(switch_dict)

    return encoding_list


@node_rewriter(tracks=[switch])
def find_measurable_flat_switch_encoding(fgraph: FunctionGraph, node: Node):
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    valued_rvs = rv_map_feature.rv_values.keys()
    switch_cond = node.inputs[0]

    encoding_list = []
    initial_interval = (-np.inf, np.inf)

    # fetch base_var as the only measurable input to the logical op in switch condition
    measurable_switch_inp = [
        component
        for component in switch_cond.owner.inputs
        if check_potential_measurability([component], valued_rvs)
    ]

    if len(measurable_switch_inp) != 1:
        return None

    base_rv = measurable_switch_inp[0]

    # We do not allow discrete RVs yet
    if base_rv.dtype.startswith("int"):
        return None

    # Since we verify the source of measurability to be the same for switch conditions
    # and all measurable components, denying broadcastability of the base_var is enough
    if base_rv.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    encoding_list = flat_switch_helper(node, valued_rvs, encoding_list, initial_interval, base_rv)
    if encoding_list is None:
        return None

    encodings, intervals = [], []
    rv_idx = ()

    # TODO: Some alternative cleaner way to do this
    for idx, item in enumerate(encoding_list):
        encoding = item["encoding"]
        # indices of intervals having base_rv as their "encoding"
        if encoding == base_rv:
            rv_idx += (idx,)

        encodings.append(encoding)
        intervals.extend((item["lower"], item["upper"]))

    flat_switch_op = FlatSwitches(out_dtype=node.outputs[0].dtype, rv_idx=rv_idx)

    new_outs = flat_switch_op.make_node(base_rv, *intervals, *encodings).default_output()
    return [new_outs]


@_logprob.register(FlatSwitches)
def flat_switches_logprob(op, values, base_rv, *inputs, **kwargs):
    from pymc.math import logdiffexp

    (value,) = values

    encodings_count = len(inputs) // 3
    # 'inputs' is of the form (lower1, upper1, lower2, upper2, encoding1, encoding2)
    # Possible TODO:
    # encodings = op.get_encodings_from_inputs(inputs)
    encodings = inputs[2 * encodings_count : 3 * encodings_count]
    encodings = pt.broadcast_arrays(*encodings)

    encodings = Assert(msg="all encodings should be unique")(
        encodings, pt.eq(pt.unique(encodings, axis=0).shape[0], len(encodings))
    )

    # TODO: We do not support the encoding graphs of discrete RVs yet

    interval_bounds = pt.broadcast_arrays(*inputs[0 : 2 * encodings_count])
    lower_interval_bounds = interval_bounds[::2]
    upper_interval_bounds = interval_bounds[1::2]

    lower_interval_bounds = pt.concatenate([i[None] for i in lower_interval_bounds], axis=0)
    upper_interval_bounds = pt.concatenate([j[None] for j in upper_interval_bounds], axis=0)

    interval_bounds = pt.concatenate(
        [lower_interval_bounds[None], upper_interval_bounds[None]], axis=0
    )

    # define a logcdf map on a scalar, use vectorize to calculate it for 2D intervals
    scalar_interval_bound = pt.scalar("scalar_interval_bound", dtype=base_rv.dtype)
    logcdf_scalar_interval_bound = _logcdf_helper(base_rv, scalar_interval_bound, **kwargs)
    logcdf_interval_bounds = pytensor.graph.replace.vectorize(
        logcdf_scalar_interval_bound, replace={scalar_interval_bound: interval_bounds}
    )
    logcdf_intervals = logdiffexp(
        logcdf_interval_bounds[1, ...], logcdf_interval_bounds[0, ...]
    )  # (encoding, *base_rv.shape)

    # default logprob is -inf if there is no RV in branches
    if op.rv_idx:
        logprob = _logprob_helper(base_rv, value, **kwargs)

        # Add rv branch (and checks whether it is possible)
        for i in op.rv_idx:
            logprob = pt.where(
                pt.and_(value <= upper_interval_bounds[i], value >= lower_interval_bounds[i]),
                logprob,
                -np.inf,
            )
    else:
        logprob = -np.inf

    for i in range(encodings_count):
        # if encoding found in interval (Lower, Upper), then Prob = CDF(Upper) - CDF(Lower)
        logprob = pt.where(
            pt.eq(value, encodings[i]),
            logcdf_intervals[i],
            logprob,
        )

    return logprob


measurable_ir_rewrites_db.register(
    "find_measurable_flat_switch_encoding",
    find_measurable_flat_switch_encoding,
    "basic",
    "censoring",
)
