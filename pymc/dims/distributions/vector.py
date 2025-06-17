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
import pytensor.xtensor.random as ptxr

from pytensor.tensor.random.utils import normalize_size_param
from pytensor.xtensor import random as pxr

from pymc.dims.distributions.core import VectorDimDistribution
from pymc.distributions.multivariate import ZeroSumNormalRV


class Categorical(VectorDimDistribution):
    xrv_op = ptxr.categorical

    @classmethod
    def dist(cls, p=None, *, logit_p=None, core_dims=None, **kwargs):
        if p is not None and logit_p is not None:
            raise ValueError("Incompatible parametrization. Can't specify both p and logit_p.")
        elif p is None and logit_p is None:
            raise ValueError("Incompatible parametrization. Must specify either p or logit_p.")

        if logit_p is not None:
            p = ptx.math.softmax(logit_p, dim=core_dims)
        return super().dist([p], core_dims=core_dims, **kwargs)


class MvNormal(VectorDimDistribution):
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


class DimZeroSumNormalRV(ZeroSumNormalRV):
    def make_node(self, rng, size, sigma, support_shape):
        if not self.input_types[1].in_same_class(normalize_size_param(size).type):
            # We need to rebuild the graph with new size type
            return self.rv_op(sigma, support_shape, size=size, rng=rng).owner
        return super().make_node(rng, size, sigma, support_shape)
