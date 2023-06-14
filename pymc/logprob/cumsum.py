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
from pytensor.tensor.extra_ops import CumOp

from pymc.logprob.abstract import MeasurableVariable, _logprob, _logprob_helper
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db


class MeasurableCumsum(CumOp):
    """A placeholder used to specify a log-likelihood for a cumsum sub-graph."""


MeasurableVariable.register(MeasurableCumsum)


@_logprob.register(MeasurableCumsum)
def logprob_cumsum(op, values, base_rv, **kwargs):
    """Compute the log-likelihood graph for a `Cumsum`."""
    (value,) = values

    value_diff = pt.diff(value, axis=op.axis)
    value_diff = pt.concatenate(
        (
            # Take first element of axis and add a broadcastable dimension so
            # that it can be concatenated with the rest of value_diff
            pt.shape_padaxis(
                pt.take(value, 0, axis=op.axis),
                axis=op.axis,
            ),
            value_diff,
        ),
        axis=op.axis,
    )

    cumsum_logp = _logprob_helper(base_rv, value_diff)

    return cumsum_logp


@node_rewriter([CumOp])
def find_measurable_cumsums(fgraph, node) -> Optional[List[MeasurableCumsum]]:
    r"""Finds `Cumsums`\s for which a `logprob` can be computed."""

    if not (isinstance(node.op, CumOp) and node.op.mode == "add"):
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableCumsum):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    base_rv = node.inputs[0]

    # Check that cumsum does not mix dimensions
    if base_rv.ndim > 1 and node.op.axis is None:
        return None

    if not rv_map_feature.request_measurable(node.inputs):
        return None

    new_op = MeasurableCumsum(axis=node.op.axis or 0, mode="add")
    new_rv = new_op.make_node(base_rv).default_output()

    return [new_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_cumsums",
    find_measurable_cumsums,
    "basic",
    "cumsum",
)
