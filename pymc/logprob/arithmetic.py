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
from pytensor.scalar import Add, Sub
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import Sum
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.type_other import NoneTypeT
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

    rng, size, mu, sigma = base_var.owner.inputs

    if isinstance(size.type, NoneTypeT):
        mu_b, sigma_b = pt.broadcast_arrays(mu, sigma)
    else:
        mu_b = pt.broadcast_to(mu, size)
        sigma_b = pt.broadcast_to(sigma, size)

    axis = node.op.axis
    mu_sum = pt.sum(mu_b, axis=axis)
    sigma_sum = pt.sqrt(pt.sum(pt.square(sigma_b), axis=axis))

    sum_rv = latent_op(mu_sum, sigma_sum, rng=rng, size=None)
    return [sum_rv]


@node_rewriter([Elemwise])
def add_of_normals(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    if not isinstance(node.op.scalar_op, Add):
        return None

    base_vars = node.inputs
    if not base_vars:
        return None

    if not all(v.owner and isinstance(v.owner.op, NormalRV) for v in base_vars):
        return None

    out_bcast = node.outputs[0].type.broadcastable
    for v in base_vars:
        if v.type.broadcastable != out_bcast:
            return None

    rng0 = base_vars[0].owner.inputs[0]
    sizes = [v.owner.inputs[1] for v in base_vars]
    size0 = None
    for s in sizes:
        if not isinstance(s.type, NoneTypeT):
            size0 = s
            break

    mus = [v.owner.inputs[2] for v in base_vars]
    sigmas = [v.owner.inputs[3] for v in base_vars]

    mu_sum = pt.add(*mus)
    sigma_sum = pt.sqrt(pt.add(*[pt.square(s) for s in sigmas]))

    latent_op = base_vars[0].owner.op
    new_rv = latent_op(mu_sum, sigma_sum, rng=rng0, size=size0)
    return [new_rv]


@node_rewriter([Elemwise])
def sub_of_normals(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    if not isinstance(node.op.scalar_op, Sub):
        return None

    base_vars = node.inputs

    if not all(v.owner and isinstance(v.owner.op, NormalRV) for v in base_vars):
        return None

    out_bcast = node.outputs[0].type.broadcastable
    for v in base_vars:
        if v.type.broadcastable != out_bcast:
            return None

    rng0 = base_vars[0].owner.inputs[0]
    sizes = [v.owner.inputs[1] for v in base_vars]
    size0 = None
    for s in sizes:
        if not isinstance(s.type, NoneTypeT):
            size0 = s
            break

    mu0 = base_vars[0].owner.inputs[2]
    mu1 = base_vars[1].owner.inputs[2]
    s0 = base_vars[0].owner.inputs[3]
    s1 = base_vars[1].owner.inputs[3]

    mu_diff = mu0 - mu1
    sigma_sum = pt.sqrt(pt.square(s0) + pt.square(s1))

    latent_op = base_vars[0].owner.op
    new_rv = latent_op(mu_diff, sigma_sum, rng=rng0, size=size0)
    return [new_rv]


measurable_ir_rewrites_db.register(
    "add_of_normals",
    add_of_normals,
    "basic",
    "arithmetic",
)
measurable_ir_rewrites_db.register(
    "sub_of_normals",
    sub_of_normals,
    "basic",
    "arithmetic",
)

measurable_ir_rewrites_db.register(
    "sum_of_normals",
    sum_of_normals,
    "basic",
    "arithmetic",
)
