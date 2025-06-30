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
import pytensor.xtensor as ptx
import pytensor.xtensor.random as pxr

from pytensor.xtensor import as_xtensor

from pymc.dims.distributions.core import (
    DimDistribution,
    PositiveDimDistribution,
    UnitDimDistribution,
)
from pymc.distributions.continuous import Beta as RegularBeta
from pymc.distributions.continuous import Gamma as RegularGamma
from pymc.distributions.continuous import HalfStudentTRV, flat, halfflat


def _get_sigma_from_either_sigma_or_tau(*, sigma, tau):
    if sigma is not None and tau is not None:
        raise ValueError("Can't pass both tau and sigma")

    if sigma is None and tau is None:
        return 1.0

    if sigma is not None:
        return sigma

    return ptx.math.reciprocal(ptx.math.sqrt(tau))


class Flat(DimDistribution):
    xrv_op = pxr.as_xrv(flat)

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([], **kwargs)


class HalfFlat(PositiveDimDistribution):
    xrv_op = pxr.as_xrv(halfflat, [], ())

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([], **kwargs)


class Normal(DimDistribution):
    xrv_op = pxr.normal

    @classmethod
    def dist(cls, mu=0, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([mu, sigma], **kwargs)


class HalfNormal(PositiveDimDistribution):
    xrv_op = pxr.halfnormal

    @classmethod
    def dist(cls, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([0.0, sigma], **kwargs)


class LogNormal(PositiveDimDistribution):
    xrv_op = pxr.lognormal

    @classmethod
    def dist(cls, mu=0, sigma=None, *, tau=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=tau)
        return super().dist([mu, sigma], **kwargs)


class StudentT(DimDistribution):
    xrv_op = pxr.t

    @classmethod
    def dist(cls, nu, mu=0, sigma=None, *, lam=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=lam)
        return super().dist([nu, mu, sigma], **kwargs)


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
        xop = pxr.as_xrv(core_rv)
        return xop(nu, sigma, core_dims=core_dims, extra_dims=extra_dims, rng=rng)


class Cauchy(DimDistribution):
    xrv_op = pxr.cauchy

    @classmethod
    def dist(cls, alpha, beta, **kwargs):
        return super().dist([alpha, beta], **kwargs)


class HalfCauchy(PositiveDimDistribution):
    xrv_op = pxr.halfcauchy

    @classmethod
    def dist(cls, beta, **kwargs):
        return super().dist([0.0, beta], **kwargs)


class Beta(UnitDimDistribution):
    xrv_op = pxr.beta

    @classmethod
    def dist(cls, alpha=None, beta=None, *, mu=None, sigma=None, nu=None, **kwargs):
        alpha, beta = RegularBeta.get_alpha_beta(alpha=alpha, beta=beta, mu=mu, sigma=sigma, nu=nu)
        return super().dist([alpha, beta], **kwargs)


class Laplace(DimDistribution):
    xrv_op = pxr.laplace

    @classmethod
    def dist(cls, mu=0, b=1, **kwargs):
        return super().dist([mu, b], **kwargs)


class Exponential(PositiveDimDistribution):
    xrv_op = pxr.exponential

    @classmethod
    def dist(cls, lam=None, *, scale=None, **kwargs):
        if lam is None and scale is None:
            scale = 1.0
        elif lam is not None and scale is not None:
            raise ValueError("Cannot pass both 'lam' and 'scale'. Use one of them.")
        elif lam is not None:
            scale = 1 / lam
        return super().dist([scale], **kwargs)


class Gamma(PositiveDimDistribution):
    xrv_op = pxr.gamma

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


class InverseGamma(PositiveDimDistribution):
    xrv_op = pxr.invgamma

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
