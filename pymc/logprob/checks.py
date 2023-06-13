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

from typing import List, Optional

import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.shape import SpecifyShape

from pymc.logprob.abstract import MeasurableVariable, _logprob, _logprob_helper
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


class MeasurableSpecifyShape(SpecifyShape):
    """A placeholder used to specify a log-likelihood for a specify-shape sub-graph."""


MeasurableVariable.register(MeasurableSpecifyShape)


@_logprob.register(MeasurableSpecifyShape)
def logprob_specify_shape(op, values, inner_rv, *shapes, **kwargs):
    (value,) = values
    # transfer specify_shape from rv to value
    value = pt.specify_shape(value, shapes)
    return _logprob_helper(inner_rv, value)


@node_rewriter([SpecifyShape])
def find_measurable_specify_shapes(fgraph, node) -> Optional[List[MeasurableSpecifyShape]]:
    r"""Finds `SpecifyShapeOp`\s for which a `logprob` can be computed."""

    if isinstance(node.op, MeasurableSpecifyShape):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    rv = node.outputs[0]

    base_rv, *shape = node.inputs

    if not (
        base_rv.owner
        and isinstance(base_rv.owner.op, MeasurableVariable)
        and base_rv not in rv_map_feature.rv_values
    ):
        return None  # pragma: no cover

    new_op = MeasurableSpecifyShape()
    new_rv = new_op.make_node(base_rv, *shape).default_output()

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_specify_shapes",
    find_measurable_specify_shapes,
    "basic",
    "specify_shape",
)


class MeasurableCheckAndRaise(CheckAndRaise):
    """A placeholder used to specify a log-likelihood for an assert sub-graph."""


MeasurableVariable.register(MeasurableCheckAndRaise)


@_logprob.register(MeasurableCheckAndRaise)
def logprob_check_and_raise(op, values, inner_rv, *assertions, **kwargs):
    from pymc.pytensorf import replace_rvs_by_values

    (value,) = values
    # transfer assertion from rv to value
    assertions = replace_rvs_by_values(assertions, rvs_to_values={inner_rv: value})
    value = op(value, *assertions)
    return _logprob_helper(inner_rv, value)


@node_rewriter([CheckAndRaise])
def find_measurable_check_and_raise(fgraph, node) -> Optional[List[MeasurableCheckAndRaise]]:
    r"""Finds `AssertOp`\s for which a `logprob` can be computed."""

    if isinstance(node.op, MeasurableCheckAndRaise):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    base_rv, *conds = node.inputs
    if not rv_map_feature.request_measurable([base_rv]):
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
