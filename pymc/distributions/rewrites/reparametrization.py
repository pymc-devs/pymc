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

from pytensor import tensor as pt
from pytensor.graph.rewriting.basic import (
    NodeRewriter,
    node_rewriter,
)
from pytensor.graph.rewriting.db import RewriteDatabase, RewriteDatabaseQuery
from pytensor.scan.basic import scan
from pytensor.scan.utils import until
from pytensor.tensor.basic import eye, ones_like, switch, zeros_like
from pytensor.tensor.extra_ops import broadcast_arrays
from pytensor.tensor.math import and_, argmax, ceil, cos, exp, log, sqrt
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

        def register(inner_rewriter: RewriteDatabase | NodeRewriter):
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
    return argmax(log(p) + GumbelRV(loc=zeros_like(p), scale=1.0, rng=rng, size=size))


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
    return loc + scale * node.op(zeros_like(loc), ones_like(scale), rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([DirichletRV])
def dirichlet_reparametrization(fgraph, node):
    raise NotImplementedError("DirichletRV is not reparametrizable")


@register_random_reparametrization
@node_rewriter([ExponentialRV])
def scale_reparametrization(fgraph, node):
    rng, size, scale = node.inputs
    return scale * node.op(ones_like(scale), rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([GammaRV])
def gamma_reparametrization(fgraph, node):
    rng, size, shape, scale = node.inputs
    return gamma_reparametrization_impl(rng, size, shape, scale)


@register_random_reparametrization
@node_rewriter([InvGammaRV])
def inv_gamma_reparametrization(fgraph, node):
    rng, size, shape, scale = node.inputs
    return 1 / gamma_reparametrization_impl(rng, size, shape, scale)


def gamma_reparametrization_impl(rng, size, shape, scale):
    # We'll implement the Marsaglia-Tsang boosted algorithm to sample
    # https://dl.acm.org/doi/epdf/10.1145/358407.358414
    # We follow the algorithm from section 3 without squeezing.
    # The squeeze algorithm from section 4 is more efficient if we can avoid
    # computing all branches of the if-else clauses, which is not possible
    # when using pytensor switches.
    # For context, shape is equal to alpha in all of the following math.
    # Sampling for alpha >= 1 is done with a rejection algorithm that finishes in constant time.
    # For alpha < 1, we need to boost the samples using:
    # Gamma(alpha, 1) = Gamma(alpha + 1, 1) * Uniform(0, 1) ** (1/alpha)
    # But this can cause numerical issues, so we'll use log samples from the boosting process:
    # log(Gamma(alpha, 1)) = log(Gamma(alpha + 1, 1)) + log(Uniform(0, 1)) / alpha
    # We can note that log(Uniform(0, 1)) = -Exponential(1)
    # and we have to guard agains the case where the exponential sample is equal to 0.
    assert size == (), (
        "Gamma reparametrization requires that you first apply the local_rv_size_lift "
        "rewrite in order to have size equal to an empty tuple."
    )

    shape, scale = broadcast_arrays(shape, scale)

    must_boost = pt.lt(shape, 1)
    alpha_orig = shape
    alpha = pt.switch(must_boost, shape + 1, shape)

    d = alpha - 1 / 3
    c = 1 / (3 * sqrt(d))

    def rejection_step(output, chosen, rng, c, d, alpha):
        next_rng, U = UniformRV()(
            zeros_like(alpha),
            ones_like(alpha),
            rng=rng,
        ).owner.outputs
        next_rng, x = NormalRV()(
            zeros_like(c),
            ones_like(c),
            rng=next_rng,
        ).owner.outputs
        V = (1 + c * x) ** 3

        X = x * x

        indicators = and_(
            ~chosen,
            and_(
                V > 0,
                pt.lt(log(U), X / 2 + d * (1 - V + log(V))),
            ),
        )
        chosen = switch(indicators, ones_like(chosen, dtype="bool"), chosen)
        output = switch(indicators, V, output)
        return (output, chosen, next_rng), until(pt.all(chosen))

    Vs, _, next_rng = scan(
        rejection_step,
        outputs_info=[zeros_like(alpha), zeros_like(alpha, dtype="bool"), rng],
        non_sequences=[c, d, alpha],
        n_steps=10_000,
        return_updates=False,
    )
    V = Vs[-1]

    log_boosting_dist = -ExponentialRV()(
        ones_like(alpha),
        rng=next_rng,
    )
    log_boost = pt.switch(
        must_boost & pt.neq(log_boosting_dist, 0),
        log_boosting_dist / alpha_orig,
        0,
    )
    return exp(log(d) + log(V) + log_boost) * scale


@register_random_reparametrization
@node_rewriter([GeometricRV])
def geometric_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    return ceil(log(UniformRV(zeros_like(p), ones_like(p), rng=rng, size=size)) / log(1 - p))


@register_random_reparametrization
@node_rewriter([LogNormalRV])
def log_normal_reparametrization(fgraph, node):
    rng, size, mean, sigma = node.inputs
    return exp(mean + sigma * NormalRV(zeros_like(mean), ones_like(sigma), rng=rng, size=size))


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
    return scale / UniformRV(zeros_like(b), ones_like(scale), rng=rng, size=size) ** (1 / b)


@register_random_reparametrization
@node_rewriter([StudentTRV])
def student_t_reparametrization(fgraph, node):
    rng, size, df, loc, scale = node.inputs
    u1 = UniformRV(zeros_like(loc), ones_like(scale), rng=rng, size=size)
    u2 = UniformRV(zeros_like(loc), ones_like(scale), rng=rng, size=size)
    return loc + scale * (sqrt(df * (u1 ** (2 / df) - 1)) * cos(2 * np.pi * u2))


@register_random_reparametrization
@node_rewriter([TriangularRV])
def triangular_reparametrization(fgraph, node):
    rng, size, left, mode, right = node.inputs
    c = (mode - left) / (right - left)
    u = UniformRV(zeros_like(c), ones_like(c), rng=rng, size=size)
    return left + (right - left) * switch(
        u < c**2,
        sqrt(u * c),
        1 - sqrt((1 - u) * (1 - c)),
    )


@register_random_reparametrization
@node_rewriter([UniformRV])
def uniform_reparametrization(fgraph, node):
    rng, size, low, high = node.inputs
    return low + (high - low) * UniformRV(zeros_like(low), ones_like(high), rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([WaldRV])
def wald_reparametrization(fgraph, node):
    rng, size, mean, scale = node.inputs
    nu = NormalRV(zeros_like(mean), ones_like(scale), rng=rng, size=size)
    u = UniformRV(zeros_like(mean), ones_like(scale), rng=rng, size=size)
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
    return -(log(UniformRV(zeros_like(shape), ones_like(shape), rng=rng, size=size)) ** (1 / shape))
