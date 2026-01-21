#   Copyright 2025 - present The PyMC Developers
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
import pytensor.xtensor as ptx
import pytensor.xtensor.random as ptxr

from pytensor.xtensor import as_xtensor

import pymc.distributions as regular_dists

from pymc.dims.distributions.core import (
    DimDistribution,
    PositiveDimDistribution,
    UnitDimDistribution,
    copy_docstring,
)
from pymc.dims.distributions.transforms import IntervalTransform
from pymc.distributions.continuous import Beta as RegularBeta
from pymc.distributions.continuous import Gamma as RegularGamma
from pymc.distributions.continuous import (
    HalfCauchyRV,
    HalfStudentTRV,
    flat,
    halfflat,
    truncated_normal,
)
from pymc.util import UNSET


def _get_sigma_from_either_sigma_or_tau(*, sigma, tau):
    if sigma is not None and tau is not None:
        raise ValueError("Can't pass both tau and sigma")

    if sigma is None and tau is None:
        return 1.0

    if sigma is not None:
        return sigma

    return ptx.math.reciprocal(ptx.math.sqrt(tau))


@copy_docstring(regular_dists.Flat)
class Flat(DimDistribution):
    xrv_op = ptxr.as_xrv(flat)

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([], **kwargs)


@copy_docstring(regular_dists.HalfFlat)
class HalfFlat(PositiveDimDistribution):
    xrv_op = ptxr.as_xrv(halfflat, [], ())

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([], **kwargs)


@copy_docstring(regular_dists.Uniform)
class Uniform(DimDistribution):
    xrv_op = ptxr.uniform

    def __new__(cls, name, lower=0, upper=1, default_transform=UNSET, observed=None, **kwargs):
        if observed is None and default_transform is UNSET:
            default_transform = IntervalTransform(lower, upper)
        return super().__new__(
            cls,
            name,
            lower=lower,
            upper=upper,
            default_transform=default_transform,
            observed=observed,
            **kwargs,
        )

    @classmethod
    def dist(cls, lower=0, upper=1, **kwargs):
        return super().dist([lower, upper], **kwargs)


@copy_docstring(regular_dists.Normal)
class Normal(DimDistribution):
    xrv_op = ptxr.normal

    @classmethod
    def dist(cls, mu=0, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([mu, sigma], **kwargs)


@copy_docstring(regular_dists.HalfNormal)
class HalfNormal(PositiveDimDistribution):
    xrv_op = ptxr.halfnormal

    @classmethod
    def dist(cls, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([0.0, sigma], **kwargs)


@copy_docstring(regular_dists.TruncatedNormal)
class TruncatedNormal(DimDistribution):
    xrv_op = ptxr.as_xrv(truncated_normal)

    def __new__(
        cls,
        name,
        *args,
        lower=-np.inf,
        upper=np.inf,
        default_transform=UNSET,
        observed=None,
        **kwargs,
    ):
        if observed is None and default_transform is UNSET:
            default_transform = IntervalTransform(lower, upper)
        return super().__new__(
            cls,
            name,
            *args,
            lower=lower,
            upper=upper,
            default_transform=default_transform,
            observed=observed,
            **kwargs,
        )

    @classmethod
    def dist(cls, mu=0, sigma=None, *, tau=None, lower=-np.inf, upper=np.inf, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([mu, sigma, lower, upper], **kwargs)


@copy_docstring(regular_dists.LogNormal)
class LogNormal(PositiveDimDistribution):
    xrv_op = ptxr.lognormal

    @classmethod
    def dist(cls, mu=0, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([mu, sigma], **kwargs)


@copy_docstring(regular_dists.StudentT)
class StudentT(DimDistribution):
    xrv_op = ptxr.t

    @classmethod
    def dist(cls, nu, mu=0, sigma=None, *, lam=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=lam)
        return super().dist([nu, mu, sigma], **kwargs)


@copy_docstring(regular_dists.HalfStudentT)
class HalfStudentT(PositiveDimDistribution):
    @classmethod
    def dist(cls, nu, sigma=None, *, lam=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=lam)
        return super().dist([nu, sigma], **kwargs)

    @classmethod
    def xrv_op(self, nu, sigma, core_dims=None, extra_dims=None, rng=None):
        nu = as_xtensor(nu)
        sigma = as_xtensor(sigma)
        core_rv = HalfStudentTRV.rv_op(nu=nu.values, sigma=sigma.values).owner.op
        xop = ptxr.as_xrv(core_rv)
        return xop(nu, sigma, core_dims=core_dims, extra_dims=extra_dims, rng=rng)


@copy_docstring(regular_dists.Cauchy)
class Cauchy(DimDistribution):
    xrv_op = ptxr.cauchy

    @classmethod
    def dist(cls, alpha, beta, **kwargs):
        return super().dist([alpha, beta], **kwargs)


@copy_docstring(regular_dists.HalfCauchy)
class HalfCauchy(PositiveDimDistribution):
    @classmethod
    def dist(cls, beta, **kwargs):
        return super().dist([beta], **kwargs)

    @classmethod
    def xrv_op(self, beta, core_dims, extra_dims=None, rng=None):
        beta = as_xtensor(beta)
        core_rv = HalfCauchyRV.rv_op(beta=beta.values).owner.op
        xop = ptxr.as_xrv(core_rv)
        return xop(beta, core_dims=core_dims, extra_dims=extra_dims, rng=rng)


@copy_docstring(regular_dists.Beta)
class Beta(UnitDimDistribution):
    xrv_op = ptxr.beta

    @classmethod
    def dist(cls, alpha=None, beta=None, *, mu=None, sigma=None, nu=None, **kwargs):
        alpha, beta = RegularBeta.get_alpha_beta(alpha=alpha, beta=beta, mu=mu, sigma=sigma, nu=nu)
        return super().dist([alpha, beta], **kwargs)


@copy_docstring(regular_dists.Laplace)
class Laplace(DimDistribution):
    xrv_op = ptxr.laplace

    @classmethod
    def dist(cls, mu=0, b=1, **kwargs):
        return super().dist([mu, b], **kwargs)


@copy_docstring(regular_dists.Exponential)
class Exponential(PositiveDimDistribution):
    xrv_op = ptxr.exponential

    @classmethod
    def dist(cls, lam=None, *, scale=None, **kwargs):
        if lam is None and scale is None:
            scale = 1.0
        elif lam is not None and scale is not None:
            raise ValueError("Cannot pass both 'lam' and 'scale'. Use one of them.")
        elif lam is not None:
            scale = 1 / lam
        return super().dist([scale], **kwargs)


@copy_docstring(regular_dists.Gamma)
class Gamma(PositiveDimDistribution):
    xrv_op = ptxr.gamma

    @classmethod
    def dist(cls, alpha=None, beta=None, *, mu=None, sigma=None, **kwargs):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            # Use sign of sigma to not let negative sigma fly by
            alpha = (mu**2 / sigma**2) * ptx.math.sign(sigma)
            beta = mu / sigma**2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use alpha and beta, or mu and sigma."
            )
        alpha, beta = RegularGamma.get_alpha_beta(alpha=alpha, beta=beta, mu=mu, sigma=sigma)
        return super().dist([alpha, ptx.math.reciprocal(beta)], **kwargs)


@copy_docstring(regular_dists.InverseGamma)
class InverseGamma(PositiveDimDistribution):
    xrv_op = ptxr.invgamma

    @classmethod
    def dist(cls, alpha=None, beta=None, *, mu=None, sigma=None, **kwargs):
        if alpha is not None:
            if beta is None:
                beta = 1.0
        elif (mu is not None) and (sigma is not None):
            # Use sign of sigma to not let negative sigma fly by
            alpha = ((2 * sigma**2 + mu**2) / sigma**2) * ptx.math.sign(sigma)
            beta = mu * (mu**2 + sigma**2) / sigma**2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use alpha and (optionally) beta, or mu and sigma"
            )
        return super().dist([alpha, beta], **kwargs)
