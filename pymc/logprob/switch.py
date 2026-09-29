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

from pytensor.scalar.basic import GE, GT, LE, LT, Mul
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import logprob_query
from pymc.logprob.query import bind_value, contains_random, query_parts
from pymc.logprob.utils import (
    CheckParameterValue,
)


def _zero_x_threshold_true_includes_zero(cond: TensorVariable, x: TensorVariable) -> bool | None:
    """Return whether `cond` is a zero threshold on `x` and includes `0` in the true branch.

    Matches `x > 0`, `x >= 0` and swapped forms `0 < x`, `0 <= x`.

    Returns
    -------
        - `False` for strict comparisons (`>`/`<`)
        - `True` for non-strict comparisons (`>=`/`<=`)
        - `None` if `cond` doesn't match a zero-threshold comparison on `x`
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

    # x > 0 or x >= 0
    if left is x and _is_zero(cast(TensorVariable, right)) and isinstance(scalar_op, GT | GE):
        return isinstance(scalar_op, GE)
    # 0 < x or 0 <= x
    if right is x and _is_zero(cast(TensorVariable, left)) and isinstance(scalar_op, LT | LE):
        return isinstance(scalar_op, LE)

    return None


def _extract_scale(neg_branch: TensorVariable, x: TensorVariable) -> TensorVariable | None:
    """Extract the other factors from a multiplication containing `x` once."""
    if neg_branch is x:
        return pt.constant(1.0)

    if neg_branch.owner is None:
        return None

    if not isinstance(neg_branch.owner.op, Elemwise):
        return None

    op = neg_branch.owner.op
    if not isinstance(op.scalar_op, Mul):
        return None

    if neg_branch.owner.inputs.count(x) != 1:
        return None
    others = [inp for inp in neg_branch.owner.inputs if inp is not x]
    return pt.mul(*others)


def rewrite_switch_non_overlapping(fgraph, query, **kwargs):
    rv, value = query_parts(query)
    cond, x, neg_branch = rv.owner.inputs
    if not isinstance(x.owner_op, RandomVariable) or x.type.numpy_dtype.kind != "f":
        return None
    if x.type.broadcastable != rv.type.broadcastable:
        return None

    a = _extract_scale(cast(TensorVariable, neg_branch), cast(TensorVariable, x))
    if a is None:
        return None
    if contains_random(a):
        return None

    # Must be strictly positive: a == 0 is not invertible (collapses a region) and
    # invalidates the non-overlapping branch inference.
    a_is_positive = pt.all(pt.gt(a, 0))

    includes_zero_in_true = _zero_x_threshold_true_includes_zero(
        cast(TensorVariable, cond), cast(TensorVariable, x)
    )
    if includes_zero_in_true is None:
        return None

    # For `a > 0`, `switch(x > 0, x, a * x)` maps to disjoint regions in `value`.
    # Select the branch using the observed `value` and the strictness of the original
    # comparison (`>` vs `>=`).
    value_implies_true_branch = pt.ge(value, 0) if includes_zero_in_true else pt.gt(value, 0)

    inverse = pt.switch(value_implies_true_branch, value, value / a)
    term = logprob_query(bind_value(fgraph, x, inverse))
    logp_expr = term - pt.switch(value_implies_true_branch, 0, pt.log(a))
    return [CheckParameterValue("switch non-overlapping scale > 0")(logp_expr, a_is_positive)]
