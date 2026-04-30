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
import pytensor

from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import DisconnectedType
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
from pymc.distributions.continuous import KumaraswamyRV
from pytensor.tensor.random.basic import (
    BernoulliRV,
    BetaRV,
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
from pytensor.tensor.special import softmax

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
        UniformRV()(zeros_like(p), ones_like(p), rng=rng, size=size) <= p,
        1,
        0,
    )


@register_random_reparametrization
@node_rewriter([BetaRV])
def beta_reparametrization(fgraph, node):
    rng, size, alpha, beta = node.inputs
    alpha, beta = broadcast_arrays(alpha, beta)
    gamma_a, next_rng = gamma_reparametrization_impl(rng, size, alpha, 1)
    gamma_b, _ = gamma_reparametrization_impl(next_rng, size, beta, 1)
    log_gamma_a = log(gamma_a)
    log_gamma_b = log(gamma_b)
    # Compute gamma_a / (gamma_a + gamma_b) without losing precision.
    log_max = pt.max(pt.stack([log_gamma_a, log_gamma_b], axis=0), axis=0)
    gamma_a_scaled = exp(log_gamma_a - log_max)
    gamma_b_scaled = exp(log_gamma_b - log_max)
    return gamma_a_scaled / (gamma_a_scaled + gamma_b_scaled)


@register_random_reparametrization
@node_rewriter([CategoricalRV])
def categorical_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    if getattr(size, "data", size) is None:
        gumbel_shape = p.shape
    else:
        gumbel_shape = pt.concatenate([size, p.shape[-1:]])
    gumbel_zeros = pt.zeros(gumbel_shape)
    g = GumbelRV()(gumbel_zeros, ones_like(gumbel_zeros), rng=rng)
    return argmax(log(p) + g, axis=-1)


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
    rng, size, alpha = node.inputs
    gamma_samples, _ = gamma_reparametrization_impl(rng, size, alpha, 1)
    return softmax(log(gamma_samples), -1)


@register_random_reparametrization
@node_rewriter([ExponentialRV])
def scale_reparametrization(fgraph, node):
    rng, size, scale = node.inputs
    return scale * node.op(ones_like(scale), rng=rng, size=size)


@register_random_reparametrization
@node_rewriter([GammaRV])
def gamma_reparametrization(fgraph, node):
    rng, size, shape, scale = node.inputs
    value, _ = gamma_reparametrization_impl(rng, size, shape, scale)
    return value


@register_random_reparametrization
@node_rewriter([InvGammaRV])
def inv_gamma_reparametrization(fgraph, node):
    rng, size, shape, scale = node.inputs
    value, _ = gamma_reparametrization_impl(rng, size, shape, 1 / scale)
    return 1 / value


def _gamma_grad_implicit(alpha, sample, n_terms=128):
    """Symbolic ∂x/∂α at fixed sample for x ~ Gamma(α, scale=1).

    Implicit-function theorem on F(α, x) = u (the regularized lower
    incomplete gamma): ∂x/∂α = -∂P(α, x)/∂α / p(x; α).
    Uses the power-series expansion of γ(α, x). The series always
    converges but slows when x ≫ α; n_terms = 128 covers typical
    Gamma samples for α, x in roughly [0.01, 100].
    """
    log_x = log(sample)
    psi = pt.digamma(alpha)
    return _gamma_grad_series(alpha, sample, log_x, psi, n_terms)


def _gamma_grad_series(alpha, x, log_x, psi, n_terms):
    # γ(α, x) = x^α e^(-x) · S(α, x),   S = Σ T_k,    T_k = x^k / (α)_(k+1)
    # ∂S/∂α  = -Σ h_k T_k,              h_k = Σ_{j=0..k} 1/(α + j)
    # g = x · [S · (ψ(α) - log x) + Σ h_k T_k]
    T0 = pt.cast(1.0, pytensor.config.floatX) / alpha

    def step(k, T_prev, h_prev, S_prev, H_prev, alpha, x):
        T_next = T_prev * x / (alpha + k)
        h_next = h_prev + 1.0 / (alpha + k)
        S_next = S_prev + T_next
        H_next = H_prev + h_next * T_next
        return T_next, h_next, S_next, H_next

    _, _, S_seq, H_seq = scan(
        step,
        sequences=[pt.arange(1, n_terms, dtype=pytensor.config.floatX)],
        outputs_info=[T0, T0, T0, T0 * T0],
        non_sequences=[alpha, x],
        return_updates=False,
    )
    # TODO: For x ≫ α the series converges slowly; XLA's `random_gamma_grad`
    #  switches to the upper-incomplete-gamma continued fraction at x > α:
    #    Γ(α, x) = e^(-x) x^α · CF,
    #    CF = 1 / (x+1-α − 1·(1-α)/(x+3-α − 2·(2-α)/(x+5-α − …)))
    #  Differentiate via a parallel modified-Lentz recurrence carrying
    #  (A_n, B_n, ∂A_n/∂α, ∂B_n/∂α). Then
    #    g = -∂P/∂α / p
    #      = (Q/p) · (log x − ψ(α)) + (1/CF) · (Q/p) · ∂CF/∂α
    #  converges geometrically at rate ≈ α/x once x > α + 1.
    return x * (S_seq[-1] * (psi - log_x) + H_seq[-1])


def _gamma_marsaglia_tsang(rng, shape, scale):
    """Forward-only Marsaglia-Tsang sampler. Returns (sample, next_rng)."""
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

    Vs, _, post_rng = scan(
        rejection_step,
        outputs_info=[zeros_like(alpha), zeros_like(alpha, dtype="bool"), rng],
        non_sequences=[c, d, alpha],
        n_steps=10_000,
        return_updates=False,
    )
    V = Vs[-1]

    final_rng, log_boosting_dist_pos = ExponentialRV()(
        ones_like(alpha),
        rng=post_rng,
    ).owner.outputs
    log_boosting_dist = -log_boosting_dist_pos
    log_boost = pt.switch(
        must_boost & pt.neq(log_boosting_dist, 0),
        log_boosting_dist / alpha_orig,
        0,
    )
    sample = exp(log(d) + log(V) + log_boost + log(scale))
    return sample, final_rng


def _gamma_lop(inputs, outputs, output_grads):
    """Custom L_op for the Gamma sampler — implicit reparam, no autodiff through scan."""
    _, shape, scale = inputs
    sample = outputs[0]
    g_sample = output_grads[0]
    x_unit = sample / scale
    g_shape = g_sample * scale * _gamma_grad_implicit(shape, x_unit)
    g_scale = g_sample * sample / scale
    return [DisconnectedType()(), g_shape, g_scale]


def gamma_reparametrization_impl(rng, size, shape, scale):
    """Marsaglia-Tsang Gamma sampler with implicit-reparameterization backward.

    The forward pass keeps the rejection-sampling scan (with `until`); the
    backward pass uses ∂x/∂α = -∂P(α,x)/∂α / p(x;α) so we never differentiate
    through the scan. Wrapped in an `OpFromGraph` whose `lop_overrides` hooks
    in the analytical gradient.

    Returns
    -------
    sample : TensorVariable
    next_rng : TensorVariable
    """
    # https://dl.acm.org/doi/epdf/10.1145/358407.358414
    # We follow the algorithm from section 3 without squeezing.
    # The squeeze algorithm from section 4 is more efficient if we can avoid
    # computing all branches of the if-else clauses, which is not possible
    # when using pytensor switches.
    # For context, shape is equal to alpha in all of the following math.
    # Sampling for alpha >= 1 is done with a rejection algorithm that finishes in constant time.
    # For alpha < 1, we boost via Gamma(alpha, 1) = Gamma(alpha + 1, 1) * Uniform()^(1/alpha)
    # in log-space to avoid underflow.
    assert getattr(size, "data", size) is None, (
        "Gamma reparametrization requires that you first apply the local_rv_size_lift "
        "rewrite so that size is None (broadcast pushed into the parameters)."
    )

    shape = pt.cast(shape, pytensor.config.floatX)
    scale = pt.cast(scale, pytensor.config.floatX)
    shape, scale = broadcast_arrays(shape, scale)

    inner_rng = rng.type()
    inner_shape = shape.type()
    inner_scale = scale.type()
    inner_sample, inner_next_rng = _gamma_marsaglia_tsang(inner_rng, inner_shape, inner_scale)

    gamma_op = OpFromGraph(
        [inner_rng, inner_shape, inner_scale],
        [inner_sample, inner_next_rng],
        lop_overrides=_gamma_lop,
        connection_pattern=[
            [False, True],
            [True, False],
            [True, False],
        ],
        on_unused_input="ignore",
    )
    sample, final_rng = gamma_op(rng, shape, scale)
    return sample, final_rng


@register_random_reparametrization
@node_rewriter([GeometricRV])
def geometric_reparametrization(fgraph, node):
    rng, size, p = node.inputs
    return ceil(log(UniformRV()(zeros_like(p), ones_like(p), rng=rng, size=size)) / log(1 - p))


@register_random_reparametrization
@node_rewriter([KumaraswamyRV])
def kumaraswamy_reparametrization(fgraph, node):
    rng, size, a, b = node.inputs
    u = UniformRV()(zeros_like(a), ones_like(b), rng=rng, size=size)
    return (1 - (1 - u) ** (1 / b)) ** (1 / a)


@register_random_reparametrization
@node_rewriter([LogNormalRV])
def log_normal_reparametrization(fgraph, node):
    rng, size, mean, sigma = node.inputs
    return exp(mean + sigma * NormalRV()(zeros_like(mean), ones_like(sigma), rng=rng, size=size))


@register_random_reparametrization
@node_rewriter([MvNormalRV])
def mv_normal_reparametrization(fgraph, node):
    rng, size, mean, cov = node.inputs
    k = mean.shape[-1]
    if getattr(size, "data", size) is None:
        z_shape = mean.shape
    else:
        z_shape = pt.concatenate([size, [k]])
    z = NormalRV()(pt.zeros(z_shape), pt.ones(z_shape), rng=rng)
    L = cholesky(cov)
    return mean + pt.einsum("...ij,...j->...i", L, z)


@register_random_reparametrization
@node_rewriter([ParetoRV])
def pareto_reparametrization(fgraph, node):
    rng, size, b, scale = node.inputs
    return scale / UniformRV()(zeros_like(b), ones_like(scale), rng=rng, size=size) ** (1 / b)


@register_random_reparametrization
@node_rewriter([StudentTRV])
def student_t_reparametrization(fgraph, node):
    rng, size, df, loc, scale = node.inputs
    next_rng, u1 = UniformRV()(zeros_like(loc), ones_like(scale), rng=rng, size=size).owner.outputs
    u2 = UniformRV()(zeros_like(loc), ones_like(scale), rng=next_rng, size=size)
    return loc + scale * (sqrt(df * (u1 ** (-2 / df) - 1)) * cos(2 * np.pi * u2))


@register_random_reparametrization
@node_rewriter([TriangularRV])
def triangular_reparametrization(fgraph, node):
    rng, size, left, mode, right = node.inputs
    c = (mode - left) / (right - left)
    u = UniformRV()(zeros_like(c), ones_like(c), rng=rng, size=size)
    return left + (right - left) * switch(
        u < c,
        sqrt(u * c),
        1 - sqrt((1 - u) * (1 - c)),
    )


@register_random_reparametrization
@node_rewriter([UniformRV])
def uniform_reparametrization(fgraph, node):
    rng, size, low, high = node.inputs
    return low + (high - low) * UniformRV()(zeros_like(low), ones_like(high), rng=rng, size=size)


def _wald_michael_schucany_haas(rng, size, mean, scale):
    """Forward-only Michael-Schucany-Haas sampler. Returns (sample, next_rng)."""
    next_rng_n, nu = NormalRV()(
        zeros_like(mean), ones_like(scale), rng=rng, size=size
    ).owner.outputs
    final_rng, u = UniformRV()(
        zeros_like(mean), ones_like(scale), rng=next_rng_n, size=size
    ).owner.outputs
    y = nu**2
    x = (
        mean
        + mean**2 * y / 2 / scale
        - mean / 2 / scale * sqrt(4 * mean * scale * y + mean**2 * y**2)
    )
    sample = switch(u <= mean / (mean + x), x, mean**2 / x)
    return sample, final_rng


def _wald_grad_implicit(mean, scale, sample):
    """Symbolic (∂x/∂μ, ∂x/∂λ) at fixed sample for Wald(μ=mean, λ=scale).

    Closed-form via implicit reparameterization on the CDF
        F(x; μ, λ) = Φ(A) + e^(2λ/μ) Φ(B),
        A = √(λ/x)·(x/μ - 1),  B = -√(λ/x)·(x/μ + 1),
    so ∂x/∂θ = -∂F/∂θ / p(x; μ, λ).
    """
    mu = mean
    lam = scale
    x = sample

    sqrt_lambda_x = sqrt(lam * x)
    sqrt_lambda_over_x = sqrt(lam / x)

    A = sqrt_lambda_x / mu - sqrt_lambda_over_x
    B = -sqrt_lambda_x / mu - sqrt_lambda_over_x

    inv_sqrt_2pi = pt.cast(1.0 / np.sqrt(2.0 * np.pi), pytensor.config.floatX)
    sqrt_2 = pt.cast(np.sqrt(2.0), pytensor.config.floatX)
    phi_A = inv_sqrt_2pi * exp(-A * A / 2)
    phi_B = inv_sqrt_2pi * exp(-B * B / 2)
    Phi_B = 0.5 * (1 + pt.erf(B / sqrt_2))

    e_2_lam_over_mu = exp(2 * lam / mu)

    dF_dmu = (
        -phi_A * sqrt_lambda_x / mu**2
        - (2 * lam / mu**2) * e_2_lam_over_mu * Phi_B
        + e_2_lam_over_mu * phi_B * sqrt_lambda_x / mu**2
    )

    dF_dlam = (
        phi_A * (x - mu) / (2 * mu * sqrt_lambda_x)
        + (2 / mu) * e_2_lam_over_mu * Phi_B
        - e_2_lam_over_mu * phi_B * (x + mu) / (2 * mu * sqrt_lambda_x)
    )

    p = sqrt(lam / (2 * np.pi * x**3)) * exp(-lam * (x - mu) ** 2 / (2 * mu**2 * x))

    return -dF_dmu / p, -dF_dlam / p


def _wald_lop(inputs, outputs, output_grads):
    """Custom L_op for Wald — implicit reparam, no autodiff through the conditional."""
    _, _, mean, scale = inputs
    sample = outputs[0]
    g_sample = output_grads[0]
    g_mu, g_lam = _wald_grad_implicit(mean, scale, sample)
    return [
        DisconnectedType()(),
        DisconnectedType()(),
        g_sample * g_mu,
        g_sample * g_lam,
    ]


@register_random_reparametrization
@node_rewriter([WaldRV])
def wald_reparametrization(fgraph, node):
    """Michael-Schucany-Haas Wald sampler with implicit-reparam backward.

    The forward pass keeps the original conditional `switch(u ≤ μ/(μ+x), x, μ²/x)`,
    which is discontinuous in μ. The backward pass swaps autodiff for the analytical
    implicit-reparameterization gradient (closed-form via the Wald CDF), avoiding
    the bias that pathwise-through-`switch` would produce.
    """
    rng, size, mean, scale = node.inputs

    inner_rng = rng.type()
    inner_size = size.type()
    inner_mean = mean.type()
    inner_scale = scale.type()
    inner_sample, inner_next_rng = _wald_michael_schucany_haas(
        inner_rng, inner_size, inner_mean, inner_scale
    )

    wald_op = OpFromGraph(
        [inner_rng, inner_size, inner_mean, inner_scale],
        [inner_sample, inner_next_rng],
        lop_overrides=_wald_lop,
        connection_pattern=[
            [False, True],
            [False, False],
            [True, False],
            [True, False],
        ],
        on_unused_input="ignore",
    )
    sample, _ = wald_op(rng, size, mean, scale)
    return sample


@register_random_reparametrization
@node_rewriter([WeibullRV])
def weibull_reparametrization(fgraph, node):
    rng, size, shape = node.inputs
    u = UniformRV()(zeros_like(shape), ones_like(shape), rng=rng, size=size)
    return (-log(u)) ** (1 / shape)
