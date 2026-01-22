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
"""Measurable rewrites for arithmetic operations."""

from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.extra_ops import broadcast_shape
from pytensor.tensor.math import Sum
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.type_other import NoneConst, NoneTypeT
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.rewriting import measurable_ir_rewrites_db


@node_rewriter([Sum])
def sum_of_normals(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    [base_var] = node.inputs
    if base_var.owner is None:
        return None

    latent_op = base_var.owner.op
    if not isinstance(latent_op, NormalRV):
        return None

    mu, sigma = latent_op.dist_params(base_var.owner)

    size = latent_op.size_param(base_var.owner)
    if size is None or isinstance(size.type, NoneTypeT):
        target_shape = broadcast_shape(mu, sigma)  # type: ignore[arg-type]
    else:
        target_shape = size  # type: ignore[assignment]

    mu_b = pt.broadcast_to(mu, target_shape)  # type: ignore[arg-type]
    sigma_b = pt.broadcast_to(sigma, target_shape)  # type: ignore[arg-type]

    axis = node.op.axis
    mu_sum = pt.sum(mu_b, axis=axis)
    sigma_sum = pt.sqrt(pt.sum(pt.square(sigma_b), axis=axis))

    rng = base_var.owner.inputs[0]
    sum_rv = latent_op.make_node(rng, NoneConst, mu_sum, sigma_sum).outputs[1]
    return [sum_rv]


measurable_ir_rewrites_db.register(
    "sum_of_normals",
    sum_of_normals,
    "basic",
    "arithmetic",
)
