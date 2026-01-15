#   Copyright 2026 - present The PyMC Developers
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
import numpy as np

from pytensor.graph.rewriting.basic import (
    NodeRewriter,
    Rewriter,
    node_rewriter,
)
from pytensor.graph.rewriting.db import RewriteDatabase, RewriteDatabaseQuery
from pytensor.tensor.basic import eye, switch, zeros_like
from pytensor.tensor.math import argmax, ceil, cos, exp, log, sqrt
from pytensor.tensor.random.basic import (
    BernoulliRV,
    CategoricalRV,
    CauchyRV,
    DirichletRV,
    ExponentialRV,
    GammaRV,
    GeometricRV,
    GumbelRV,
    HalfCauchyRV,
    HalfNormalRV,
    InvGammaRV,
    LaplaceRV,
    LogisticRV,
    LogNormalRV,
    MvNormalRV,
    NormalRV,
    ParetoRV,
    StudentTRV,
    TriangularRV,
    UniformRV,
    WaldRV,
    WeibullRV,
)
from pytensor.tensor.slinalg import cholesky

reparametrization_trick_db = RewriteDatabaseQuery(include=["random_reparametrization_trick"])


def register_random_reparametrization(
    node_rewriter: RewriteDatabase | NodeRewriter | str, *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: RewriteDatabase | Rewriter):
            return register_random_reparametrization(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__
        reparametrization_trick_db.register(name, node_rewriter, *tags, **kwargs)
        return node_rewriter


@register_random_reparametrization
@node_rewriter([BernoulliRV])
def bernoulli_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    return switch(
        UniformRV(0.0, 1.0, rng=rng, size=size) <= p,
        1,
        0,
    )


@register_random_reparametrization
@node_rewriter([CategoricalRV])
def categorical_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    return argmax(log(p) + GumbelRV(loc=0.0, scale=1.0, rng=rng, size=size))


@register_random_reparametrization
@node_rewriter(
    [
        CauchyRV,
        GumbelRV,
        HalfCauchyRV,
        HalfNormalRV,
        LaplaceRV,
        LogisticRV,
        NormalRV,
    ]
)
def loc_scale_reparametrization(fgraph, node):
    rng, size, loc, scale = node.inputs
    return loc + scale * node.op(0.0, 1.0, rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([DirichletRV])
def dirichlet_reparametrization(fgraph, node):
    raise NotImplementedError("DirichletRV is not reparametrizable")


@register_random_reparametrization
@node_rewriter([ExponentialRV])
def scale_reparametrization(fgraph, node):
    rng, size, scale = node.inputs
    return scale * node.op(1.0, rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([GammaRV, InvGammaRV])
def gamma_reparametrization(fgraph, node):
    rng, size, shape, scale = node.inputs
    return scale * node.op(shape, 1.0, rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([GeometricRV])
def geometric_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    return ceil(log(UniformRV(0.0, 1.0, rng=rng, size=size)) / log(1 - p))


@register_random_reparametrization
@node_rewriter([LogNormalRV])
def log_normal_reparametrization(fgraph, node):
    rng, size, mean, sigma = node.inputs
    return exp(mean + sigma * NormalRV(0.0, 1.0, rng=rng, size=size))


@register_random_reparametrization
@node_rewriter([MvNormalRV])
def mv_normal_reparametrization(fgraph, node):
    rng, size, mean, cov = node.inputs
    zero_mean = zeros_like(mean)
    return mean + cholesky(cov) @ MvNormalRV(
        zero_mean, eye(zero_mean.shape[-1]), size=size, rng=rng
    )


@register_random_reparametrization
@node_rewriter([ParetoRV])
def pareto_reparametrization(fgraph, node):
    rng, size, b, scale = node.inputs
    return scale / UniformRV(0.0, 1.0, rng=rng, size=size) ** (1 / b)


@register_random_reparametrization
@node_rewriter([StudentTRV])
def student_t_reparametrization(fgraph, node):
    rng, size, df, loc, scale = node.inputs
    u1 = UniformRV(0.0, 1.0, rng=rng, size=size)
    u2 = UniformRV(0.0, 1.0, size=size)
    return loc + scale * (sqrt(df * (u1 ** (2 / df) - 1)) * cos(2 * np.pi * u2))


@register_random_reparametrization
@node_rewriter([TriangularRV])
def triangular_reparametrization(fgraph, node):
    rng, size, left, mode, right = node.inputs
    c = (mode - left) / (right - left)
    u = UniformRV(0.0, 1.0, rng=rng, size=size)
    return left + (right - left) * switch(
        u < c**2,
        sqrt(u * c),
        1 - sqrt((1 - u) * (1 - c)),
    )


@register_random_reparametrization
@node_rewriter([UniformRV])
def uniform_reparametrization(fgraph, node):
    rng, size, low, high = node.inputs
    return low + (high - low) * UniformRV(0.0, 1.0, rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([WaldRV])
def wald_reparametrization(fgraph, node):
    rng, size, mean, scale = node.inputs
    nu = NormalRV(0.0, 1.0, rng=rng, size=size)
    u = UniformRV(0.0, 1.0, size=size)
    y = nu**2
    x = (
        mean
        + mean**2 * y / 2 / scale
        - mean / 2 / scale * sqrt(4 * mean * scale * y + mean**2 * y**2)
    )
    return switch(
        u <= mean / (mean + x),
        x,
        mean**2 / x,
    )


@register_random_reparametrization
@node_rewriter([WeibullRV])
def weibull_reparametrization(fgraph, node):
    rng, size, shape = node.inputs
    return -(log(UniformRV(0.0, 1.0, rng=rng, size=size)) ** (1 / shape))
