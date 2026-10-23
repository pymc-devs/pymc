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

from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.shape import SpecifyShape

from pymc.logprob.abstract import logprob_query
from pymc.logprob.query import (
    bind_value,
    infer_support_axes,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)
from pymc.logprob.utils import replace_rvs_by_values


@infer_support_axes.register(SpecifyShape)
@infer_support_axes.register(CheckAndRaise)
def measure_check(op, var):
    return support_axes(var.owner.inputs[0])


@rewrite_logprob_query.register(SpecifyShape)
def rewrite_specify_shape_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    base, *shape = rv.owner.inputs
    return [logprob_query(bind_value(fgraph, base, pt.specify_shape(value, shape)))]


@rewrite_logprob_query.register(CheckAndRaise)
def rewrite_check_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    base, *checks = rv.owner.inputs
    checks = replace_rvs_by_values(checks, rvs_to_values={base: value})
    return [
        logprob_query(bind_value(fgraph, base, CheckAndRaise(op.exc_type, op.msg)(value, *checks)))
    ]
