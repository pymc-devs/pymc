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


import pytensor.tensor as pt

from pytensor.tensor.extra_ops import CumOp

from pymc.logprob.abstract import logprob_query
from pymc.logprob.query import (
    bind_value,
    infer_support_axes,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)


@infer_support_axes.register(CumOp)
def measure_cumsum(op, var):
    if op.mode != "add" or (op.axis is None and var.owner.inputs[0].ndim > 1):
        raise NotImplementedError("Only cumulative sums along one axis are supported")
    return support_axes(var.owner.inputs[0])


@rewrite_logprob_query.register(CumOp)
def rewrite_cumsum_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    axis = op.axis or 0
    backward = pt.concatenate(
        [pt.take(value, [0], axis=axis), pt.diff(value, axis=axis)], axis=axis
    )
    return [logprob_query(bind_value(fgraph, rv.owner.inputs[0], backward))]
