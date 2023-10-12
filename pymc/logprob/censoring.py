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
import pytensor.tensor as pt

from pytensor.graph import Op
from pytensor.graph.basic import Apply, Node, Variable, walk
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
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
from pytensor.tensor.basic import switch as switch
from pytensor.tensor.math import ceil, clip, floor, round_half_to_even
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorConstant

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    _logcdf,
    _logprob,
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
    __props__ = ("meta_info", "out_dtype")

    def __init__(self, *args, meta_info, out_dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_info = meta_info
        self.out_dtype = out_dtype

    def make_node(self, base_rv):
        assert isinstance(base_rv, Variable) and (base_rv.owner.op, MeasurableVariable)
        return Apply(self, [base_rv], [base_rv.type()])

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

    # check if the switch_cond is measurable and if measurability sources for all switch conditions are the same
    if not compare_measurability_source([switch_cond, base_rv], valued_rvs):
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

    # Check that the measurability source is the same for all measurable components within the current branch
    for i in measurable_var_idx:
        component = components[i]
        if not compare_measurability_source([base_rv, component], valued_rvs):
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
        # There is at least one switch, so measurable components can be at most 1
        for i in measurable_var_idx:
            switch_dict = {
                "lower": adjusted_intervals[i][0],
                "upper": adjusted_intervals[i][1],
                "encoding": components[i],
            }
            encoding_list.append(switch_dict)

        # Recurse through the switch component(es)
        for j in switch_comp_idx:
            encoding_list = flat_switch_helper(
                components[j].owner, valued_rvs, encoding_list, adjusted_intervals[j], base_rv
            )

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
    measurable_comp_list = [
        component
        for component in switch_cond.owner.inputs
        if check_potential_measurability([component], valued_rvs)
    ]

    if len(measurable_comp_list) != 1:
        return None

    base_rv = measurable_comp_list[0]

    # We do not allow discrete RVs yet
    if base_rv.dtype.startswith("int"):
        return None

    # Since we verify the source of measurability to be the same for switch conditions
    # and all measurable components, denying broadcastability of the base_var is enough
    if base_rv.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    encoding_list = flat_switch_helper(node, valued_rvs, encoding_list, initial_interval, base_rv)

    flat_switch_op = FlatSwitches(meta_info=encoding_list, out_dtype=base_rv.dtype)

    # print("\n")
    # for i in range(len(encoding_list)):
    #     print("lower:", encoding_list[i]["lower"].eval())
    #     print("upper:", encoding_list[i]["upper"].eval())
    #     print("encoding:", encoding_list[i]["encoding"], "\n")

    new_outs = flat_switch_op.make_node(base_rv).default_output()
    return [new_outs]


@_logprob.register(FlatSwitches)
def flat_switches_logprob(op, values, *inputs):
    # Defined logp expression based on this
    [value] = values

    logp = pt.zeros_like(value)
    logp.name = "interval_logp"
    return logp


measurable_ir_rewrites_db.register(
    "find_measurable_flat_switch_encoding",
    find_measurable_flat_switch_encoding,
    "basic",
    "censoring",
)


def get_measurability_source(
    inp: TensorVariable, valued_rvs: Container[TensorVariable]
) -> Set[TensorVariable]:
    """
    Returns all the sources of measurability in the input boolean condition.
    """
    from pymc.distributions.distribution import SymbolicRandomVariable

    ancestor_var_set = set()

    for ancestor_var in walk_model(
        [inp],
        walk_past_rvs=False,
        stop_at_vars=set(valued_rvs),
    ):
        if (
            ancestor_var.owner
            and isinstance(ancestor_var.owner.op, (RandomVariable, SymbolicRandomVariable))
            # TODO: Check if MeasurableVariable needs to be added
            and ancestor_var not in valued_rvs
        ):
            ancestor_var_set.add(ancestor_var)

    return ancestor_var_set


def compare_measurability_source(
    inputs: Tuple[TensorVariable], valued_rvs: Container[TensorVariable]
) -> bool:
    """
    Compares the source of measurability for all elements in 'inputs' separately
    """
    ancestor_var_set = set()
    [ancestor_var_set.update(get_measurability_source(inp, valued_rvs)) for inp in inputs]
    return len(ancestor_var_set) == 1


def walk_model(
    graphs: Iterable[TensorVariable],
    walk_past_rvs: bool = False,
    stop_at_vars: Optional[Set[TensorVariable]] = None,
    expand_fn: Callable[[TensorVariable], List[TensorVariable]] = lambda var: [],
) -> Generator[TensorVariable, None, None]:
    if stop_at_vars is None:
        stop_at_vars = set()

    def expand(var: TensorVariable, stop_at_vars=stop_at_vars) -> List[TensorVariable]:
        new_vars = expand_fn(var)

        if (
            var.owner
            and (walk_past_rvs or not isinstance(var.owner.op, MeasurableVariable))
            and (var not in stop_at_vars)
        ):
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    yield from walk(graphs, expand, False)
