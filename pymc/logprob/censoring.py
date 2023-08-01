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

from typing import Callable, Container, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Node, walk
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import Ceil, Clip, Floor, RoundHalfToEven, Switch
from pytensor.scalar.basic import clip as scalar_clip
from pytensor.scalar.basic import switch as scalar_switch
from pytensor.tensor.basic import switch as switch
from pytensor.tensor.math import ceil, clip, floor, round_half_to_even
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.var import TensorConstant, TensorVariable

from pymc.logprob.abstract import MeasurableElemwise, _logcdf, _logprob, _logprob_helper
from pymc.logprob.binary import MeasurableBitwise
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db
from pymc.logprob.utils import CheckParameterValue, check_potential_measurability


class MeasurableClip(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""

    valid_scalar_types = (Clip,)


measurable_clip = MeasurableClip(scalar_clip)


@node_rewriter(tracks=[clip])
def find_measurable_clips(fgraph: FunctionGraph, node: Node) -> Optional[List[MeasurableClip]]:
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
def find_measurable_roundings(fgraph: FunctionGraph, node: Node) -> Optional[List[MeasurableRound]]:
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


class MeasurableSwitchEncoding(MeasurableElemwise):
    """A placeholder used to specify the log-likelihood for a encoded RV sub-graph."""

    valid_scalar_types = (Switch,)


measurable_switch_encoding = MeasurableSwitchEncoding(scalar_switch)


@node_rewriter(tracks=[switch])
def find_measurable_switch_encoding(
    fgraph: FunctionGraph, node: Node
) -> Optional[List[MeasurableSwitchEncoding]]:
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    valued_rvs = rv_map_feature.rv_values.keys()

    switch_condn, *components = node.inputs

    # broadcasting of switch condition is not supported
    if switch_condn.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    measurable_comp_list = [
        idx
        for idx, component in enumerate(components)
        if check_potential_measurability([component], valued_rvs)
    ]

    # this automatically checks the measurability of the switch condition and converts switch to MeasurableSwitch
    if rv_map_feature.request_measurable([switch_condn]) != [switch_condn]:
        return None

    # Maximum one branch allowed to be measurable
    if len(measurable_comp_list) > 1:
        return None

    # If at least one of the branches is measurable
    if len(measurable_comp_list) == 1:
        measurable_comp_idx = measurable_comp_list[0]
        measurable_component = components[measurable_comp_idx]

        # broadcasting of the measurable component is not supported
        if (
            (measurable_component.type.broadcastable != node.outputs[0].broadcastable)
            or (not compare_measurability_source([switch_condn, measurable_component], valued_rvs))
            or (not rv_map_feature.request_measurable([measurable_component]))
        ):
            return None

        if measurable_comp_idx == 0:
            # changing the first branch of switch to always be the encoding
            inverted_switch = pt.invert(switch_condn)

            bitwise_op = MeasurableBitwise(inverted_switch.owner.op.scalar_op)
            measurable_inverted_switch = bitwise_op.make_node(switch_condn).default_output()
            encoded_rv = measurable_switch_encoding.make_node(
                measurable_inverted_switch, *components[::-1]
            ).default_output()

            return [encoded_rv]

    encoded_rv = measurable_switch_encoding.make_node(switch_condn, *components).default_output()

    return [encoded_rv]


@_logprob.register(MeasurableSwitchEncoding)
def switch_encoding_logprob(op, values, *inputs, **kwargs):
    (value,) = values

    switch_condn, *components = inputs

    # Right now, this only works for switch with both encoding branches.
    logprob = pt.switch(
        pt.eq(value, components[0]),
        _logprob_helper(switch_condn, pt.as_tensor(np.array(True)), **kwargs),
        pt.switch(
            pt.eq(value, components[1]),
            _logprob_helper(switch_condn, pt.as_tensor(np.array(False))),
            -np.inf,
        ),
    )

    # TODO: Calculate logprob for switch with one measurable component If RV is discrete,
    #  give preference over encoding.

    return logprob


measurable_ir_rewrites_db.register(
    "find_measurable_switch_encoding", find_measurable_switch_encoding, "basic", "censoring"
)


def compare_measurability_source(
    inputs: Tuple[TensorVariable], valued_rvs: Container[TensorVariable]
) -> bool:
    ancestor_var_set = set()

    # retrieve the source of measurability for all elements in 'inputs' separately.
    for inp in inputs:
        for ancestor_var in walk_model(
            [inp],
            walk_past_rvs=False,
            stop_at_vars=set(valued_rvs),
        ):
            if (
                ancestor_var.owner
                and isinstance(ancestor_var.owner.op, RandomVariable)
                and ancestor_var not in valued_rvs
            ):
                ancestor_var_set.add(ancestor_var)

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
            and (walk_past_rvs or not isinstance(var.owner.op, RandomVariable))
            and (var not in stop_at_vars)
        ):
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    yield from walk(graphs, expand, False)
