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
from pytensor.tensor.math import and_, argmax, ceil, cos, exp, floor, log, sqrt
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
    # from the Gamma(alpha, 1) distribution. The process is divided into 3 parts:
    # 1. Gamma(floor(alpha), 1) ~ sum_0^{floor(alpha)}(Exponential(1)) when floor(alpha) > 0
    # 2. delta = alpha - floor(alpha) -> Gamma(delta + 1, 1) is sampled via rejection method that uses a uniform and a Normal
    # 3. Gamma(delta, 1) = Gamma(delta + 1, 1) * Uniform(0, 1) ** (1/delta) which is called boosting
    # The rejection method from step 2 cannot be used directly on delta because for delta < 1, it
    # does not converge in constant time. That's why we need to boost it to be bigger than 1.
    # The Gamma(alpha, beta) can be computed as Gamma(alpha, 1) * beta
    assert size == (), (
        "Gamma reparametrization requires that you first apply the local_rv_size_lift "
        "rewrite in order to have size equal to an empty tuple."
    )

    shape, scale = broadcast_arrays(shape, scale)

    # TODO: Spawn 3 RNGs from the supplied rng and use those in the scans and boosting uniform

    # Part 1: Integer part of alpha by summing IID exponentials
    def branch_floor_alpha(step, output, rng, int_alpha):
        next_rng, summand = ExponentialRV()(
            ones_like(int_alpha).astype("floatX"),
            rng=rng,
        ).owner.outputs

        return step + 1, switch(step < int_alpha, output + summand, output), next_rng

    int_alpha = floor(shape).astype("int")
    _, gamma_int_samples, _ = scan(
        branch_floor_alpha,
        outputs_info=[pt.zeros((), dtype="int"), zeros_like(shape), rng],
        non_sequences=[int_alpha],
        n_steps=pt.max(int_alpha),
        return_updates=False,
    )

    delta = shape - int_alpha

    # Part 2: Marsaglia-Tsang rejection algorithm
    d = (delta + 1) - 1 / 3
    c = 1 / sqrt(9 * d)

    def rejection_step(output, chosen, rng, c, d):
        next_rng, U = UniformRV()(
            zeros_like(delta),
            ones_like(delta),
            rng=rng,
        ).owner.outputs
        next_rng, X = NormalRV()(
            zeros_like(delta),
            ones_like(delta),
            rng=next_rng,
        ).owner.outputs
        v = (1 + c * X) ** 3
        indicators = and_(
            ~chosen, and_(pt.gt(v, 0), pt.lt(log(U), (0.5 * X**2 + d - d * v + d * log(v))))
        )
        chosen = switch(indicators, ones_like(chosen, dtype="bool"), chosen)
        output = switch(indicators, d * v, output)
        return (output, chosen, next_rng), until(pt.all(chosen))

    gamma_delta_samples, _, _ = scan(
        rejection_step,
        outputs_info=[zeros_like(delta), zeros_like(delta, dtype="bool"), rng],
        non_sequences=[c, d],
        n_steps=10_000,
        return_updates=False,
    )

    # Part 3: Invert the boosting to get delta instead of delta + 1 and sum
    boosting_uniform = UniformRV()(
        zeros_like(delta),
        ones_like(delta),
        rng=rng,
    )

    # Part 2 and 3 only have to be included if delta is different than zero

    return (
        gamma_int_samples[-1]
        + pt.switch(
            pt.gt(delta, 0),
            (gamma_delta_samples[-1] * boosting_uniform) ** (1 / delta),
            zeros_like(delta),
        )
    ) * scale


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
