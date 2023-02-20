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

import pytensor.tensor as at

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.shape import SpecifyShape

from pymc.logprob.abstract import (
    MeasurableVariable,
    _logprob,
    assign_custom_measurable_outputs,
    logprob,
)
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


class MeasurableSpecifyShape(SpecifyShape):
    """A placeholder used to specify a log-likelihood for a specify-shape sub-graph."""


MeasurableVariable.register(MeasurableSpecifyShape)


@_logprob.register(MeasurableSpecifyShape)
def logprob_specify_shape(op, values, inner_rv, *shapes, **kwargs):
    (value,) = values
    # transfer specify_shape from rv to value
    value = at.specify_shape(value, shapes)
    return logprob(inner_rv, value)


@node_rewriter([SpecifyShape])
def find_measurable_specify_shapes(fgraph, node):
    r"""Finds `SpecifyShapeOp`\s for which a `logprob` can be computed."""

    if not (isinstance(node.op, SpecifyShape) and node.op.mode == "add"):
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableSpecifyShape):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    rv = node.outputs[0]

    base_rv = node.inputs[0]
    if not (
        base_rv.owner
        and isinstance(base_rv.owner.op, MeasurableVariable)
        and base_rv not in rv_map_feature.rv_values
    ):
        return None  # pragma: no cover

    new_op = MeasurableSpecifyShape(node.input_shapes)
    # Make base_var unmeasurable
    unmeasurable_base_rv = assign_custom_measurable_outputs(base_rv.owner)
    new_rv = new_op.make_node(unmeasurable_base_rv).default_output()
    new_rv.name = rv.name

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_specify_shapes",
    find_measurable_specify_shapes,
    "basic",
    "specify_shape",
)
