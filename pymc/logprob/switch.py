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

"""Measurable switch-based transforms."""

from typing import cast

import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar import Switch
from pytensor.scalar import switch as scalar_switch
from pytensor.scalar.basic import GE, GT, LE, LT, Mul
from pytensor.tensor.basic import switch as tensor_switch
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableElemwise, MeasurableOp, _logprob, _logprob_helper
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.transforms import MeasurableTransform
from pymc.logprob.utils import (
    CheckParameterValue,
    check_potential_measurability,
    filter_measurable_variables,
)


class MeasurableSwitchNonOverlapping(MeasurableElemwise):
    """Placeholder for switch transforms whose branch images do not overlap.

    Currently matches leaky-ReLU graphs of the form `switch(x > 0, x, a * x)`.
    """

    valid_scalar_types = (Switch,)


measurable_switch_non_overlapping = MeasurableSwitchNonOverlapping(scalar_switch)


def _is_x_threshold_condition(cond: TensorVariable, x: TensorVariable) -> bool:
    """Check whether `cond` is equivalent to `x > 0` / `x >= 0` (and swapped forms)."""
    if cond.owner is None:
        return False
    if not isinstance(cond.owner.op, Elemwise):
        return False
    scalar_op = cond.owner.op.scalar_op
    if not isinstance(scalar_op, GT | GE | LT | LE):
        return False

    left, right = cond.owner.inputs

    def _is_zero(v: TensorVariable) -> bool:
        try:
            return pt.get_underlying_scalar_constant_value(v) == 0
        except NotScalarConstantError:
            return False

    # x > 0 or x >= 0
    if left is x and _is_zero(cast(TensorVariable, right)) and isinstance(scalar_op, GT | GE):
        return True
    # 0 < x or 0 <= x
    if right is x and _is_zero(cast(TensorVariable, left)) and isinstance(scalar_op, LT | LE):
        return True

    return False


def _extract_scale_from_measurable_mul(
    neg_branch: TensorVariable, x: TensorVariable
) -> TensorVariable | None:
    """Extract scale `a` from a measurable multiplication that represents `a * x`."""
    if neg_branch is x:
        return pt.constant(1.0)

    if neg_branch.owner is None:
        return None

    if not isinstance(neg_branch.owner.op, MeasurableTransform):
        return None

    op = neg_branch.owner.op
    if not isinstance(op.scalar_op, Mul):
        return None

    # MeasurableTransform takes (measurable_input, scale)
    if len(neg_branch.owner.inputs) != 2:
        return None

    if neg_branch.owner.inputs[op.measurable_input_idx] is not x:
        return None

    scale = neg_branch.owner.inputs[1 - op.measurable_input_idx]
    return cast(TensorVariable, scale)


@node_rewriter([tensor_switch])
def find_measurable_switch_non_overlapping(fgraph, node):
    """Detect `switch(x > 0, x, a * x)` and replace it by a measurable op."""
    if isinstance(node.op, MeasurableOp):
        return None

    cond, pos_branch, neg_branch = node.inputs

    # Only mark the switch measurable once both branches are already measurable.
    # Then the logprob can simply gate between branch logps evaluated at `value`.
    if set(filter_measurable_variables([pos_branch, neg_branch])) != {pos_branch, neg_branch}:
        return None

    x = cast(TensorVariable, pos_branch)

    if x.type.dtype.startswith("int"):
        return None

    if x.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    if not _is_x_threshold_condition(cast(TensorVariable, cond), x):
        return None

    a = _extract_scale_from_measurable_mul(cast(TensorVariable, neg_branch), x)
    if a is None:
        return None

    # Disallow slope `a` that could be (directly or indirectly) measurable.
    # This rewrite targets deterministic, non-overlapping transforms parametrized by non-RVs.
    if check_potential_measurability([a]):
        return None

    return [
        measurable_switch_non_overlapping(
            cast(TensorVariable, cond),
            x,
            cast(TensorVariable, neg_branch),
        )
    ]


@_logprob.register(MeasurableSwitchNonOverlapping)
def logprob_switch_non_overlapping(op, values, cond, x, neg_branch, **kwargs):
    (value,) = values

    a = _extract_scale_from_measurable_mul(
        cast(TensorVariable, neg_branch), cast(TensorVariable, x)
    )
    if a is None:
        raise NotImplementedError("Could not extract non-overlapping scale")

    a_is_positive = pt.all(pt.gt(a, 0))

    # For `a > 0`, `switch(x > 0, x, a * x)` maps to disjoint regions in `value`:
    # true branch -> value > 0, false branch -> value <= 0.
    value_implies_true_branch = pt.gt(value, 0)

    logp_expr = pt.switch(
        value_implies_true_branch,
        _logprob_helper(x, value, **kwargs),
        _logprob_helper(neg_branch, value, **kwargs),
    )

    return CheckParameterValue("switch non-overlapping scale > 0")(logp_expr, a_is_positive)


measurable_ir_rewrites_db.register(
    "find_measurable_switch_non_overlapping",
    find_measurable_switch_non_overlapping,
    "basic",
    "transform",
)
