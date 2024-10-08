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
from typing import cast

import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import TensorVariable
from pytensor.tensor.shape import SpecifyShape

from pymc.logprob.abstract import MeasurableOp, _logprob, _logprob_helper
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import filter_measurable_variables, replace_rvs_by_values


class MeasurableSpecifyShape(MeasurableOp, SpecifyShape):
    """A placeholder used to specify a log-likelihood for a specify-shape sub-graph."""


@_logprob.register(MeasurableSpecifyShape)
def logprob_specify_shape(op, values, inner_rv, *shapes, **kwargs):
    (value,) = values
    # transfer specify_shape from rv to value
    value = pt.specify_shape(value, shapes)
    return _logprob_helper(inner_rv, value)


@node_rewriter([SpecifyShape])
def find_measurable_specify_shapes(fgraph, node) -> list[TensorVariable] | None:
    r"""Find `SpecifyShapeOp`\s for which a `logprob` can be computed."""
    if isinstance(node.op, MeasurableSpecifyShape):
        return None  # pragma: no cover

    base_rv, *shape = node.inputs

    if not filter_measurable_variables([base_rv]):
        return None

    new_rv = cast(TensorVariable, MeasurableSpecifyShape()(base_rv, *shape))

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_specify_shapes",
    find_measurable_specify_shapes,
    "basic",
    "specify_shape",
)


class MeasurableCheckAndRaise(MeasurableOp, CheckAndRaise):
    """A placeholder used to specify a log-likelihood for an assert sub-graph."""


@_logprob.register(MeasurableCheckAndRaise)
def logprob_check_and_raise(op, values, inner_rv, *assertions, **kwargs):
    (value,) = values
    # transfer assertion from rv to value
    assertions = replace_rvs_by_values(assertions, rvs_to_values={inner_rv: value})
    value = op(value, *assertions)
    return _logprob_helper(inner_rv, value)


@node_rewriter([CheckAndRaise])
def find_measurable_check_and_raise(fgraph, node) -> list[TensorVariable] | None:
    r"""Find `AssertOp`\s for which a `logprob` can be computed."""
    if isinstance(node.op, MeasurableCheckAndRaise):
        return None  # pragma: no cover

    base_rv, *conds = node.inputs

    if not filter_measurable_variables([base_rv]):
        return None

    op = node.op
    new_op = MeasurableCheckAndRaise(exc_type=op.exc_type, msg=op.msg)
    new_rv = new_op.make_node(base_rv, *conds).default_output()

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_check_and_raise",
    find_measurable_check_and_raise,
    "basic",
    "assert",
)
