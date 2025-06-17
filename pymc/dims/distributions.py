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

from pymc.dims.distribution_core import (
    DimDistribution,
    MultivariateDimDistribution,
    PositiveDimDistribution,
    UnitDimDistribution,
)
from pymc.distributions.continuous import Beta as RegularBeta
from pymc.distributions.continuous import Gamma as RegularGamma
from pymc.distributions.continuous import HalfStudentTRV, flat, halfflat
from pymc.distributions.continuous import InverseGamma as RegularInverseGamma


def _get_sigma_from_either_sigma_or_tau(*, sigma, tau):
    if sigma is not None and tau is not None:
        raise ValueError("Can't pass both tau and sigma")

    if sigma is None and tau is None:
        return 1.0

    if sigma is not None:
        return sigma

    return ptx.math.reciprocal(ptx.math.square(sigma))


class Flat(DimDistribution):
    xrv_op = pxr._as_xrv(flat)

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([], **kwargs)


class HalfFlat(PositiveDimDistribution):
    xrv_op = pxr._as_xrv(halfflat, [], ())

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
    xrv_op = pxr._as_xrv(HalfStudentTRV.rv_op, [(), ()], ())

    @classmethod
    def dist(cls, nu, sigma=None, *, lam=None, **kwargs):
        sigma = _get_sigma_from_either_sigma_or_tau(sigma=sigma, tau=lam)
        return super().dist([nu, sigma], **kwargs)


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
        alpha, beta = RegularGamma.get_alpha_beta(alpha=alpha, beta=beta, mu=None, sigma=None)
        return super().dist([alpha, 1 / beta], **kwargs)


class InverseGamma(PositiveDimDistribution):
    xrv_op = pxr.invgamma

    @classmethod
    def dist(cls, alpha=None, beta=None, *, mu=None, sigma=None, **kwargs):
        alpha, beta = RegularInverseGamma.get_alpha_beta(alpha=alpha, beta=beta, mu=mu, sigma=sigma)
        return super().dist([alpha, beta], **kwargs)


class MvNormal(MultivariateDimDistribution):
    """Multivariate Normal distribution.

    Parameters
    ----------
    mu : xtensor_like
        Mean vector of the distribution.
    cov : xtensor_like, optional
        Covariance matrix of the distribution. Only one of `cov` or `chol` must be provided.
    chol : xtensor_like, optional
        Cholesky decomposition of the covariance matrix. only one of `cov` or `chol` must be provided.
    lower : bool, default True
        If True, the Cholesky decomposition is assumed to be lower triangular.
        If False, it is assumed to be upper triangular.
    core_dims: Sequence of string, optional
        Sequence of two strings representing the core dimensions of the distribution.
        The two dimensions must be present in `cov` or `chol`, and exactly one must also be present in `mu`.
    **kwargs
        Additional keyword arguments used to define the distribution.

    Returns
    -------
    XTensorVariable
        An xtensor variable representing the multivariate normal distribution.
        The output contains the core dimension that is shared between `mu` and `cov` or `chol`.

    """

    xrv_op = pxr.multivariate_normal

    @classmethod
    def dist(cls, mu, cov=None, *, chol=None, lower=True, core_dims=None, **kwargs):
        if "tau" in kwargs:
            raise NotImplementedError("MvNormal does not support 'tau' parameter.")

        if not (isinstance(core_dims, tuple | list) and len(core_dims) == 2):
            raise ValueError("MvNormal requires 2 core_dims")

        if cov is None and chol is None:
            raise ValueError("Either 'cov' or 'chol' must be provided.")

        if chol is not None:
            d0, d1 = core_dims
            if not lower:
                # By logical symmetry this must be the only correct way to implement lower
                # We refuse to test it because it is not useful
                d1, d0 = d0, d1

            chol = cls._as_xtensor(chol)
            # chol @ chol.T in xarray semantics requires a rename
            safe_name = "_"
            if "_" in chol.type.dims:
                safe_name *= max(map(len, chol.type.dims)) + 1
            cov = chol.dot(chol.rename({d0: safe_name}), dim=d1).rename({safe_name: d1})

        return super().dist([mu, cov], core_dims=core_dims, **kwargs)
