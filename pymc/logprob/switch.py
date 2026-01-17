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
from pytensor.tensor.random.op import RandomVariable
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

    Currently matches graphs with branches `(x, a * x)` in either order and a zero-threshold
    condition on `x` (e.g. `x > 0`, `x <= 0`, `0 < x`).
    """

    valid_scalar_types = (Switch,)


measurable_switch_non_overlapping = MeasurableSwitchNonOverlapping(scalar_switch)


def _zero_x_threshold_true_branch_semantics(
    cond: TensorVariable, x: TensorVariable
) -> tuple[bool, bool] | None:
    """Parse a zero-threshold comparison on `x`.

    Matches `x > 0`, `x >= 0`, `x < 0`, `x <= 0` and swapped forms like `0 < x`, `0 >= x`.

    Returns
    -------
    (includes_zero_in_true, true_is_positive_side)
        - `includes_zero_in_true` is `True` for non-strict comparisons (`>=`/`<=`)
        - `true_is_positive_side` is `True` when the true branch corresponds to `x > 0`/`x >= 0`
    `None`
        If `cond` doesn't match a zero-threshold comparison on `x`.
    """
    if cond.owner is None:
        return None
    if not isinstance(cond.owner.op, Elemwise):
        return None
    scalar_op = cond.owner.op.scalar_op
    if not isinstance(scalar_op, GT | GE | LT | LE):
        return None

    left, right = cond.owner.inputs

    def _is_zero(v: TensorVariable) -> bool:
        try:
            return pt.get_underlying_scalar_constant_value(v) == 0
        except NotScalarConstantError:
            return False

    def _direct_form() -> tuple[bool, bool] | None:
        # x ? 0
        if left is x and _is_zero(cast(TensorVariable, right)):
            if isinstance(scalar_op, GT):
                return (False, True)
            if isinstance(scalar_op, GE):
                return (True, True)
            if isinstance(scalar_op, LT):
                return (False, False)
            if isinstance(scalar_op, LE):
                return (True, False)
        return None

    def _swapped_form() -> tuple[bool, bool] | None:
        # 0 ? x
        if right is x and _is_zero(cast(TensorVariable, left)):
            if isinstance(scalar_op, LT):
                # 0 < x  <=>  x > 0
                return (False, True)
            if isinstance(scalar_op, LE):
                # 0 <= x  <=>  x >= 0
                return (True, True)
            if isinstance(scalar_op, GT):
                # 0 > x  <=>  x < 0
                return (False, False)
            if isinstance(scalar_op, GE):
                # 0 >= x  <=>  x <= 0
                return (True, False)
        return None

    return _direct_form() or _swapped_form()


def _match_scaled_switch_branches(
    true_branch: TensorVariable, false_branch: TensorVariable
) -> tuple[TensorVariable, TensorVariable, TensorVariable] | None:
    """Match a scaled-branch switch pattern.

    Accepts either `(x, a * x)` or `(a * x, x)`.

    Returns
    -------
    (x, scaled_branch, a)
        Where `x` is the base measurable random variable and `scaled_branch` is the branch
        that represents `a * x`.
    """
    for x, scaled_branch in ((true_branch, false_branch), (false_branch, true_branch)):
        if x.owner is None or not isinstance(x.owner.op, RandomVariable):
            continue
        a = _extract_scale_from_measurable_mul(scaled_branch, x)
        if a is None:
            continue
        return x, scaled_branch, a

    return None


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
    """Detect a non-overlapping scaled-branch switch and replace it by a measurable op.

    Currently matches the pair of branches `(x, a * x)` in either order, and a zero-threshold
    condition on `x` expressed in equivalent spellings (e.g. `x > 0`, `0 <= x`, `x <= 0`).
    """
    if isinstance(node.op, MeasurableOp):
        return None

    cond, true_branch, false_branch = node.inputs

    # Only mark the switch measurable once both branches are already measurable.
    # Then the logprob can simply gate between branch logps evaluated at `value`.
    if set(filter_measurable_variables([true_branch, false_branch])) != {true_branch, false_branch}:
        return None

    match = _match_scaled_switch_branches(
        cast(TensorVariable, true_branch), cast(TensorVariable, false_branch)
    )
    if match is None:
        return None

    x, _scaled_branch, a = match

    if x.type.numpy_dtype.kind != "f":
        return None

    # Avoid rewriting cases where `x` is broadcasted/replicated by `cond` or `a`.
    # We require the positive branch to be a base `RandomVariable` output.
    if x.type.broadcastable != node.outputs[0].type.broadcastable:
        return None

    cond_semantics = _zero_x_threshold_true_branch_semantics(cast(TensorVariable, cond), x)
    if cond_semantics is None:
        return None

    # Disallow slope `a` that could be (directly or indirectly) measurable.
    # This rewrite targets deterministic, non-overlapping transforms parametrized by non-RVs.
    if check_potential_measurability([a]):
        return None

    return [
        measurable_switch_non_overlapping(
            cast(TensorVariable, cond),
            cast(TensorVariable, true_branch),
            cast(TensorVariable, false_branch),
        )
    ]


@_logprob.register(MeasurableSwitchNonOverlapping)
def logprob_switch_non_overlapping(op, values, cond, true_branch, false_branch, **kwargs):
    (value,) = values

    match = _match_scaled_switch_branches(
        cast(TensorVariable, true_branch), cast(TensorVariable, false_branch)
    )
    if match is None:
        raise NotImplementedError("Could not identify non-overlapping switch branches")

    x, _scaled_branch, a = match

    # Must be strictly positive: a == 0 is not invertible (collapses a region) and
    # invalidates the non-overlapping branch inference.
    a_is_positive = pt.all(pt.gt(a, 0))

    cond_semantics = _zero_x_threshold_true_branch_semantics(cast(TensorVariable, cond), x)
    if cond_semantics is None:
        raise NotImplementedError("Could not identify zero-threshold condition")

    includes_zero_in_true, true_is_positive_side = cond_semantics

    # For `a > 0`, a scaled-branch switch maps to disjoint regions in `value`.
    # Select the branch using the observed `value` and the strictness of the original
    # comparison (`>` vs `>=`).
    if true_is_positive_side:
        value_implies_true_branch = pt.ge(value, 0) if includes_zero_in_true else pt.gt(value, 0)
    else:
        value_implies_true_branch = pt.le(value, 0) if includes_zero_in_true else pt.lt(value, 0)

    logp_expr = pt.switch(
        value_implies_true_branch,
        _logprob_helper(true_branch, value, **kwargs),
        _logprob_helper(false_branch, value, **kwargs),
    )

    return CheckParameterValue("switch non-overlapping scale > 0")(logp_expr, a_is_positive)


measurable_ir_rewrites_db.register(
    "find_measurable_switch_non_overlapping",
    find_measurable_switch_non_overlapping,
    "basic",
    "transform",
)
