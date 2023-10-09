#   Copyright 2023 The PyMC Developers
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

from functools import partial, reduce
from typing import Optional

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.op import Op
from pytensor.raise_op import Assert
from pytensor.sparse.basic import sp_sum
from pytensor.tensor import TensorConstant, gammaln, sigmoid
from pytensor.tensor.linalg import cholesky, det, eigh
from pytensor.tensor.linalg import inv as matrix_inverse
from pytensor.tensor.linalg import solve_triangular, trace
from pytensor.tensor.random.basic import dirichlet, multinomial, multivariate_normal
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import (
    broadcast_params,
    supp_shape_from_ref_param_shape,
)
from pytensor.tensor.type import TensorType
from scipy import stats

import pymc as pm

from pymc.distributions import transforms
from pymc.distributions.continuous import BoundedContinuous, ChiSquared, Normal
from pymc.distributions.dist_math import (
    betaln,
    check_parameters,
    factln,
    logpow,
    multigammaln,
)
from pymc.distributions.distribution import (
    Continuous,
    Discrete,
    Distribution,
    SymbolicRandomVariable,
    _moment,
    moment,
)
from pymc.distributions.shape_utils import (
    _change_dist_size,
    broadcast_dist_samples_shape,
    change_dist_size,
    get_support_shape,
    rv_size_is_none,
    to_tuple,
)
from pymc.distributions.transforms import Interval, ZeroSumTransform, _default_transform
from pymc.logprob.abstract import _logprob
from pymc.math import kron_diag, kron_dot
from pymc.pytensorf import floatX, intX
from pymc.util import check_dist_not_registered

__all__ = [
    "MvNormal",
    "ZeroSumNormal",
    "MvStudentT",
    "Dirichlet",
    "Multinomial",
    "DirichletMultinomial",
    "OrderedMultinomial",
    "Wishart",
    "WishartBartlett",
    "LKJCorr",
    "LKJCholeskyCov",
    "MatrixNormal",
    "KroneckerNormal",
    "CAR",
    "ICAR",
    "StickBreakingWeights",
]

solve_lower = partial(solve_triangular, lower=True)
solve_upper = partial(solve_triangular, lower=False)


class SimplexContinuous(Continuous):
    """Base class for simplex continuous distributions"""


@_default_transform.register(SimplexContinuous)
def simplex_cont_transform(op, rv):
    return transforms.simplex


# Step methods and advi do not catch LinAlgErrors at the
# moment. We work around that by using a cholesky op
# that returns a nan as first entry instead of raising
# an error.
nan_lower_cholesky = partial(cholesky, lower=True, on_error="nan")


def quaddist_matrix(cov=None, chol=None, tau=None, lower=True, *args, **kwargs):
    if len([i for i in [tau, cov, chol] if i is not None]) != 1:
        raise ValueError("Incompatible parameterization. Specify exactly one of tau, cov, or chol.")

    if cov is not None:
        cov = pt.as_tensor_variable(cov)
        if cov.ndim < 2:
            raise ValueError("cov must be at least two dimensional.")
    elif tau is not None:
        tau = pt.as_tensor_variable(tau)
        if tau.ndim < 2:
            raise ValueError("tau must be at least two dimensional.")
        cov = matrix_inverse(tau)
    else:
        chol = pt.as_tensor_variable(chol)
        if chol.ndim < 2:
            raise ValueError("chol must be at least two dimensional.")

        if not lower:
            chol = pt.swapaxes(chol, -1, -2)

        # tag as lower triangular to enable pytensor rewrites of chol(l.l') -> l
        chol.tag.lower_triangular = True
        cov = pt.matmul(chol, pt.swapaxes(chol, -1, -2))

    return cov


def quaddist_chol(value, mu, cov):
    """Compute (x - mu).T @ Sigma^-1 @ (x - mu) and the logdet of Sigma."""
    if value.ndim == 0:
        raise ValueError("Value can't be a scalar")
    if value.ndim == 1:
        onedim = True
        value = value[None, :]
    else:
        onedim = False

    delta = value - mu
    chol_cov = nan_lower_cholesky(cov)

    diag = pt.diagonal(chol_cov, axis1=-2, axis2=-1)
    # Check if the covariance matrix is positive definite.
    ok = pt.all(diag > 0, axis=-1)
    # If not, replace the diagonal. We return -inf later, but
    # need to prevent solve_lower from throwing an exception.
    chol_cov = pt.switch(ok[..., None, None], chol_cov, 1)
    delta_trans = solve_lower(chol_cov, delta, b_ndim=1)
    quaddist = (delta_trans**2).sum(axis=-1)
    logdet = pt.log(diag).sum(axis=-1)

    if onedim:
        return quaddist[0], logdet, ok
    else:
        return quaddist, logdet, ok


class MvNormal(Continuous):
    r"""
    Multivariate normal log-likelihood.

    .. math::

       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{k/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Vector of means.
    cov : tensor_like of float, optional
        Covariance matrix. Exactly one of cov, tau, or chol is needed.
    tau : tensor_like of float, optional
        Precision matrix. Exactly one of cov, tau, or chol is needed.
    chol : tensor_like of float, optional
        Cholesky decomposition of covariance matrix. Exactly one of cov,
        tau, or chol is needed.
    lower: bool, default=True
        Whether chol is the lower tridiagonal cholesky factor.

    Examples
    --------
    Define a multivariate normal variable for a given covariance
    matrix::

        cov = np.array([[1., 0.5], [0.5, 2]])
        mu = np.zeros(2)
        vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=(5, 2))

    Most of the time it is preferable to specify the cholesky
    factor of the covariance instead. For example, we could
    fit a multivariate outcome like this (see the docstring
    of `LKJCholeskyCov` for more information about this)::

        mu = np.zeros(3)
        true_cov = np.array([[1.0, 0.5, 0.1],
                             [0.5, 2.0, 0.2],
                             [0.1, 0.2, 1.0]])
        data = np.random.multivariate_normal(mu, true_cov, 10)

        sd_dist = pm.Exponential.dist(1.0, shape=3)
        chol, corr, stds = pm.LKJCholeskyCov('chol_cov', n=3, eta=2,
            sd_dist=sd_dist, compute_corr=True)
        vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)

    For unobserved values it can be better to use a non-centered
    parametrization::

        sd_dist = pm.Exponential.dist(1.0, shape=3)
        chol, _, _ = pm.LKJCholeskyCov('chol_cov', n=3, eta=2,
            sd_dist=sd_dist, compute_corr=True)
        vals_raw = pm.Normal('vals_raw', mu=0, sigma=1, shape=(5, 3))
        vals = pm.Deterministic('vals', pt.dot(chol, vals_raw.T).T)
    """
    rv_op = multivariate_normal

    @classmethod
    def dist(cls, mu=0, cov=None, *, tau=None, chol=None, lower=True, **kwargs):
        mu = pt.as_tensor_variable(mu)
        cov = quaddist_matrix(cov, chol, tau, lower)
        # PyTensor is stricter about the shape of mu, than PyMC used to be
        mu, _ = pt.broadcast_arrays(mu, cov[..., -1])
        return super().dist([mu, cov], **kwargs)

    def moment(rv, size, mu, cov):
        # mu is broadcasted to the potential length of cov in `dist`
        moment = mu
        if not rv_size_is_none(size):
            moment_size = pt.concatenate([size, [mu.shape[-1]]])
            moment = pt.full(moment_size, mu)
        return moment

    def logp(value, mu, cov):
        """
        Calculate log-probability of Multivariate Normal distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        quaddist, logdet, ok = quaddist_chol(value, mu, cov)
        k = floatX(value.shape[-1])
        norm = -0.5 * k * pm.floatX(np.log(2 * np.pi))
        return check_parameters(
            norm - 0.5 * quaddist - logdet,
            ok,
            msg="posdef",
        )


class MvStudentTRV(RandomVariable):
    name = "multivariate_studentt"
    ndim_supp = 1
    ndims_params = [0, 1, 2]
    dtype = "floatX"
    _print_name = ("MvStudentT", "\\operatorname{MvStudentT}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=1,
        )

    @classmethod
    def rng_fn(cls, rng, nu, mu, cov, size):
        if size is None:
            # When size is implicit, we need to broadcast parameters correctly,
            # so that the MvNormal draws and the chisquare draws have the same number of batch dimensions.
            # nu broadcasts mu and cov
            if np.ndim(nu) > max(mu.ndim - 1, cov.ndim - 2):
                _, mu, cov = broadcast_params((nu, mu, cov), ndims_params=cls.ndims_params)
            # nu is broadcasted by either mu or cov
            elif np.ndim(nu) < max(mu.ndim - 1, cov.ndim - 2):
                nu, _, _ = broadcast_params((nu, mu, cov), ndims_params=cls.ndims_params)

        mv_samples = multivariate_normal.rng_fn(rng=rng, mean=np.zeros_like(mu), cov=cov, size=size)

        # Take chi2 draws and add an axis of length 1 to the right for correct broadcasting below
        chi2_samples = np.sqrt(rng.chisquare(nu, size=size) / nu)[..., None]

        return (mv_samples / chi2_samples) + mu


mv_studentt = MvStudentTRV()


class MvStudentT(Continuous):
    r"""
    Multivariate Student-T log-likelihood.

    .. math::
        f(\mathbf{x}| \nu,\mu,\Sigma) =
        \frac
            {\Gamma\left[(\nu+p)/2\right]}
            {\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}
             \left|{\Sigma}\right|^{1/2}
             \left[
               1+\frac{1}{\nu}
               ({\mathbf x}-{\mu})^T
               {\Sigma}^{-1}({\mathbf x}-{\mu})
             \right]^{-(\nu+p)/2}}

    ========  =============================================
    Support   :math:`x \in \mathbb{R}^p`
    Mean      :math:`\mu` if :math:`\nu > 1` else undefined
    Variance  :math:`\frac{\nu}{\mu-2}\Sigma`
                  if :math:`\nu>2` else undefined
    ========  =============================================

    Parameters
    ----------
    nu : tensor_like of float
        Degrees of freedom, should be a positive scalar.
    Sigma : tensor_like of float, optional
        Scale matrix. Use `scale` in new code.
    mu : tensor_like of float, optional
        Vector of means.
    scale : tensor_like of float, optional
        The scale matrix.
    tau : tensor_like of float, optional
        The precision matrix.
    chol : tensor_like of float, optional
        The cholesky factor of the scale matrix.
    lower : bool, default=True
        Whether the cholesky fatcor is given as a lower triangular matrix.
    """
    rv_op = mv_studentt

    @classmethod
    def dist(cls, nu, *, Sigma=None, mu=0, scale=None, tau=None, chol=None, lower=True, **kwargs):
        cov = kwargs.pop("cov", None)
        if cov is not None:
            warnings.warn(
                "Use the scale argument to specify the scale matrix. "
                "cov will be removed in future versions.",
                FutureWarning,
            )
            scale = cov
        if Sigma is not None:
            if scale is not None:
                raise ValueError("Specify only one of scale and Sigma")
            scale = Sigma
        nu = pt.as_tensor_variable(floatX(nu))
        mu = pt.as_tensor_variable(floatX(mu))
        scale = quaddist_matrix(scale, chol, tau, lower)
        # PyTensor is stricter about the shape of mu, than PyMC used to be
        mu, _ = pt.broadcast_arrays(mu, scale[..., -1])

        return super().dist([nu, mu, scale], **kwargs)

    def moment(rv, size, nu, mu, scale):
        # mu is broadcasted to the potential length of scale in `dist`
        mu, _ = pt.random.utils.broadcast_params([mu, nu], ndims_params=[1, 0])
        moment = mu
        if not rv_size_is_none(size):
            moment_size = pt.concatenate([size, [mu.shape[-1]]])
            moment = pt.full(moment_size, moment)
        return moment

    def logp(value, nu, mu, scale):
        """
        Calculate log-probability of Multivariate Student's T distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        quaddist, logdet, ok = quaddist_chol(value, mu, scale)
        k = floatX(value.shape[-1])

        norm = gammaln((nu + k) / 2.0) - gammaln(nu / 2.0) - 0.5 * k * pt.log(nu * np.pi)
        inner = -(nu + k) / 2.0 * pt.log1p(quaddist / nu)
        res = norm + inner - logdet

        return check_parameters(res, ok, nu > 0, msg="posdef, nu > 0")


class Dirichlet(SimplexContinuous):
    r"""
    Dirichlet log-likelihood.

    .. math::

       f(\mathbf{x}|\mathbf{a}) =
           \frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}
           \prod_{i=1}^k x_i^{a_i - 1}

    ========  ===============================================
    Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum x_i = 1`
    Mean      :math:`\dfrac{a_i}{\sum a_i}`
    Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}`
              where :math:`a_0 = \sum a_i`
    ========  ===============================================

    Parameters
    ----------
    a : tensor_like of float
        Concentration parameters (a > 0). The number of categories is given by the
        length of the last axis.
    """
    rv_op = dirichlet

    @classmethod
    def dist(cls, a, **kwargs):
        a = pt.as_tensor_variable(a)
        # mean = a / pt.sum(a)
        # mode = pt.switch(pt.all(a > 1), (a - 1) / pt.sum(a - 1), np.nan)

        return super().dist([a], **kwargs)

    def moment(rv, size, a):
        norm_constant = pt.sum(a, axis=-1)[..., None]
        moment = a / norm_constant
        if not rv_size_is_none(size):
            moment = pt.full(pt.concatenate([size, [a.shape[-1]]]), moment)
        return moment

    def logp(value, a):
        """
        Calculate log-probability of Dirichlet distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        # only defined for sum(value) == 1
        res = pt.sum(logpow(value, a - 1) - gammaln(a), axis=-1) + gammaln(pt.sum(a, axis=-1))
        res = pt.switch(
            pt.or_(
                pt.any(pt.lt(value, 0), axis=-1),
                pt.any(pt.gt(value, 1), axis=-1),
            ),
            -np.inf,
            res,
        )
        return check_parameters(
            res,
            a > 0,
            msg="a > 0",
        )


class Multinomial(Discrete):
    r"""
    Multinomial log-likelihood.

    Generalizes binomial distribution, but instead of each trial resulting
    in "success" or "failure", each one results in exactly one of some
    fixed finite number k of possible outcomes over n independent trials.
    'x[i]' indicates the number of times outcome number i was observed
    over the n trials.

    .. math::

       f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    ==========  ===========================================
    Support     :math:`x \in \{0, 1, \ldots, n\}` such that
                :math:`\sum x_i = n`
    Mean        :math:`n p_i`
    Variance    :math:`n p_i (1 - p_i)`
    Covariance  :math:`-n p_i p_j` for :math:`i \ne j`
    ==========  ===========================================

    Parameters
    ----------
    n : tensor_like of int
        Total counts in each replicate (n > 0).
    p : tensor_like of float
        Probability of each one of the different outcomes (0 <= p <= 1). The number of
        categories is given by the length of the last axis. Elements are expected to sum
        to 1 along the last axis.
    """
    rv_op = multinomial

    @classmethod
    def dist(cls, n, p, *args, **kwargs):
        p = pt.as_tensor_variable(p)
        if isinstance(p, TensorConstant):
            p_ = np.asarray(p.data)
            if np.any(p_ < 0):
                raise ValueError(f"Negative `p` parameters are not valid, got: {p_}")
            p_sum_ = np.sum([p_], axis=-1)
            if not np.all(np.isclose(p_sum_, 1.0)):
                warnings.warn(
                    f"`p` parameters sum to {p_sum_}, instead of 1.0. "
                    "They will be automatically rescaled. "
                    "You can rescale them directly to get rid of this warning.",
                    UserWarning,
                )
                p_ = p_ / pt.sum(p_, axis=-1, keepdims=True)
                p = pt.as_tensor_variable(p_)
        n = pt.as_tensor_variable(n)
        p = pt.as_tensor_variable(p)
        return super().dist([n, p], *args, **kwargs)

    def moment(rv, size, n, p):
        n = pt.shape_padright(n)
        mode = pt.round(n * p)
        diff = n - pt.sum(mode, axis=-1, keepdims=True)
        inc_bool_arr = pt.abs(diff) > 0
        mode = pt.inc_subtensor(mode[inc_bool_arr.nonzero()], diff[inc_bool_arr.nonzero()])
        if not rv_size_is_none(size):
            output_size = pt.concatenate([size, [p.shape[-1]]])
            mode = pt.full(output_size, mode)
        return mode

    def logp(value, n, p):
        """
        Calculate log-probability of Multinomial distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        res = factln(n) + pt.sum(-factln(value) + logpow(p, value), axis=-1)
        res = pt.switch(
            pt.or_(pt.any(pt.lt(value, 0), axis=-1), pt.neq(pt.sum(value, axis=-1), n)),
            -np.inf,
            res,
        )
        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            pt.isclose(pt.sum(p, axis=-1), 1),
            pt.ge(n, 0),
            msg="0 <= p <= 1, sum(p) = 1, n >= 0",
        )


class DirichletMultinomialRV(RandomVariable):
    name = "dirichlet_multinomial"
    ndim_supp = 1
    ndims_params = [0, 1]
    dtype = "int64"
    _print_name = ("DirichletMN", "\\operatorname{DirichletMN}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=1,
        )

    @classmethod
    def rng_fn(cls, rng, n, a, size):
        if n.ndim > 0 or a.ndim > 1:
            n, a = broadcast_params([n, a], cls.ndims_params)
            size = tuple(size or ())

            if size:
                n = np.broadcast_to(n, size)
                a = np.broadcast_to(a, size + (a.shape[-1],))

            res = np.empty(a.shape)
            for idx in np.ndindex(a.shape[:-1]):
                p = rng.dirichlet(a[idx])
                res[idx] = rng.multinomial(n[idx], p)
            return res
        else:
            # n is a scalar, a is a 1d array
            p = rng.dirichlet(a, size=size)  # (size, a.shape)

            res = np.empty(p.shape)
            for idx in np.ndindex(p.shape[:-1]):
                res[idx] = rng.multinomial(n, p[idx])

            return res


dirichlet_multinomial = DirichletMultinomialRV()


class DirichletMultinomial(Discrete):
    r"""Dirichlet Multinomial log-likelihood.

    Dirichlet mixture of Multinomials distribution, with a marginalized PMF.

    .. math::

        f(x \mid n, a) = \frac{\Gamma(n + 1)\Gamma(\sum a_k)}
                              {\Gamma(n + \sum a_k)}
                         \prod_{k=1}^K
                         \frac{\Gamma(x_k +  a_k)}
                              {\Gamma(x_k + 1)\Gamma(a_k)}

    ==========  ===========================================
    Support     :math:`x \in \{0, 1, \ldots, n\}` such that
                :math:`\sum x_i = n`
    Mean        :math:`n \frac{a_i}{\sum{a_k}}`
    ==========  ===========================================

    Parameters
    ----------
    n : tensor_like of int
        Total counts in each replicate (n > 0).

    a : tensor_like of float
        Dirichlet concentration parameters (a > 0). The number of categories is given by
        the length of the last axis.
    """
    rv_op = dirichlet_multinomial

    @classmethod
    def dist(cls, n, a, *args, **kwargs):
        n = intX(n)
        a = floatX(a)

        return super().dist([n, a], **kwargs)

    def moment(rv, size, n, a):
        p = a / pt.sum(a, axis=-1, keepdims=True)
        return moment(Multinomial.dist(n=n, p=p, size=size))

    def logp(value, n, a):
        """
        Calculate log-probability of DirichletMultinomial distribution
        at specified value.

        Parameters
        ----------
        value: integer array
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        sum_a = a.sum(axis=-1)
        const = (gammaln(n + 1) + gammaln(sum_a)) - gammaln(n + sum_a)
        series = gammaln(value + a) - (gammaln(value + 1) + gammaln(a))
        res = const + series.sum(axis=-1)

        res = pt.switch(
            pt.or_(
                pt.any(pt.lt(value, 0), axis=-1),
                pt.neq(pt.sum(value, axis=-1), n),
            ),
            -np.inf,
            res,
        )

        return check_parameters(
            res,
            a > 0,
            n >= 0,
            msg="a > 0, n >= 0",
        )


class _OrderedMultinomial(Multinomial):
    r"""
    Underlying class for ordered multinomial distributions.
    See docs for the OrderedMultinomial wrapper class for more details on how to use it in models.
    """
    rv_op = multinomial

    @classmethod
    def dist(cls, eta, cutpoints, n, *args, **kwargs):
        eta = pt.as_tensor_variable(floatX(eta))
        cutpoints = pt.as_tensor_variable(cutpoints)
        n = pt.as_tensor_variable(intX(n))

        pa = sigmoid(cutpoints - pt.shape_padright(eta))
        p_cum = pt.concatenate(
            [
                pt.zeros_like(pt.shape_padright(pa[..., 0])),
                pa,
                pt.ones_like(pt.shape_padright(pa[..., 0])),
            ],
            axis=-1,
        )
        p = p_cum[..., 1:] - p_cum[..., :-1]

        return super().dist(n, p, *args, **kwargs)


class OrderedMultinomial:
    r"""
    Wrapper class for Ordered Multinomial distributions.

    Useful for regression on ordinal data whose values range
    from 1 to K as a function of some predictor, :math:`\eta`, but
    which are _aggregated_ by trial, like multinomial observations (in
    contrast to `pm.OrderedLogistic`, which only accepts ordinal data
    in a _disaggregated_ format, like categorical observations).
    The cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables. The number
    of cutpoints is K - 1. It is recommended that the cutpoints are
    constrained to be ordered.

    .. math::

       f(k \mid \eta, c) = \left\{
         \begin{array}{l}
           1 - \text{logit}^{-1}(\eta - c_1)
             \,, \text{if } k = 0 \\
           \text{logit}^{-1}(\eta - c_{k - 1}) -
           \text{logit}^{-1}(\eta - c_{k})
             \,, \text{if } 0 < k < K \\
           \text{logit}^{-1}(\eta - c_{K - 1})
             \,, \text{if } k = K \\
         \end{array}
       \right.

    Parameters
    ----------
    eta : tensor_like of float
        The predictor.
    cutpoints : tensor_like of float
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges. Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.
    n : tensor_like of int
        The total number of multinomial trials.
    compute_p : boolean, default=True
        Whether to compute and store in the trace the inferred probabilities of each
        categories,
        based on the cutpoints' values. Defaults to True.
        Might be useful to disable it if memory usage is of interest.

    Examples
    --------
    .. code-block:: python

        # Generate data for a simple 1 dimensional example problem
        true_cum_p = np.array([0.1, 0.15, 0.25, 0.50, 0.65, 0.90, 1.0])
        true_p = np.hstack([true_cum_p[0], true_cum_p[1:] - true_cum_p[:-1]])
        fake_elections = np.random.multinomial(n=1_000, pvals=true_p, size=60)

        # Ordered multinomial regression
        with pm.Model() as model:
            cutpoints = pm.Normal(
                "cutpoints",
                mu=np.arange(6) - 2.5,
                sigma=1.5,
                initval=np.arange(6) - 2.5,
                transform=pm.distributions.transforms.ordered,
            )

            pm.OrderedMultinomial(
                "results",
                eta=0.0,
                cutpoints=cutpoints,
                n=fake_elections.sum(1),
                observed=fake_elections,
            )

            trace = pm.sample()

        # Plot the results
        arviz.plot_posterior(trace_12_4, var_names=["complete_p"], ref_val=list(true_p));
    """

    def __new__(cls, name, *args, compute_p=True, **kwargs):
        out_rv = _OrderedMultinomial(name, *args, **kwargs)
        if compute_p:
            pm.Deterministic(f"{name}_probs", out_rv.owner.inputs[4], dims=kwargs.get("dims"))
        return out_rv

    @classmethod
    def dist(cls, *args, **kwargs):
        return _OrderedMultinomial.dist(*args, **kwargs)


def posdef(AA):
    try:
        scipy.linalg.cholesky(AA)
        return True
    except scipy.linalg.LinAlgError:
        return False


class PosDefMatrix(Op):
    """
    Check if input is positive definite. Input should be a square matrix.

    """

    # Properties attribute
    __props__ = ()

    # Compulsory if itypes and otypes are not defined

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        assert x.ndim == 2
        o = TensorType(dtype="bool", shape=[])()
        return Apply(self, [x], [o])

    # Python implementation:
    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.array(posdef(x), dtype="bool")
        except Exception:
            pm._log.exception("Failed to check if %s positive definite", x)
            raise

    def infer_shape(self, fgraph, node, shapes):
        return [[]]

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(pytensor.config.floatX)]

    def __str__(self):
        return "MatrixIsPositiveDefinite"


matrix_pos_def = PosDefMatrix()


class WishartRV(RandomVariable):
    name = "wishart"
    ndim_supp = 2
    ndims_params = [0, 2]
    dtype = "floatX"
    _print_name = ("Wishart", "\\operatorname{Wishart}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        # The shape of second parameter `V` defines the shape of the output.
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=1,
        )

    @classmethod
    def rng_fn(cls, rng, nu, V, size):
        scipy_size = size if size else 1  # Default size for Scipy's wishart.rvs is 1
        result = stats.wishart.rvs(int(nu), V, size=scipy_size, random_state=rng)
        if size == (1,):
            return result[np.newaxis, ...]
        else:
            return result


wishart = WishartRV()


class Wishart(Continuous):
    r"""
    Wishart log-likelihood.

    The Wishart distribution is the probability distribution of the
    maximum-likelihood estimator (MLE) of the precision matrix of a
    multivariate normal distribution.  If V=1, the distribution is
    identical to the chi-square distribution with nu degrees of
    freedom.

    .. math::

       f(X \mid nu, T) =
           \frac{{\mid T \mid}^{nu/2}{\mid X \mid}^{(nu-k-1)/2}}{2^{nu k/2}
           \Gamma_p(nu/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of :math:`X`.

    ========  =========================================
    Support   :math:`X(p x p)` positive definite matrix
    Mean      :math:`nu V`
    Variance  :math:`nu (v_{ij}^2 + v_{ii} v_{jj})`
    ========  =========================================

    Parameters
    ----------
    nu : tensor_like of int
        Degrees of freedom, > 0.
    V : tensor_like of float
        p x p positive definite matrix.

    Notes
    -----
    This distribution is unusable in a PyMC model. You should instead
    use LKJCholeskyCov or LKJCorr.
    """
    rv_op = wishart

    @classmethod
    def dist(cls, nu, V, *args, **kwargs):
        nu = pt.as_tensor_variable(intX(nu))
        V = pt.as_tensor_variable(floatX(V))

        warnings.warn(
            "The Wishart distribution can currently not be used "
            "for MCMC sampling. The probability of sampling a "
            "symmetric matrix is basically zero. Instead, please "
            "use LKJCholeskyCov or LKJCorr. For more information "
            "on the issues surrounding the Wishart see here: "
            "https://github.com/pymc-devs/pymc/issues/538.",
            UserWarning,
        )

        # mean = nu * V
        # p = V.shape[0]
        # mode = pt.switch(pt.ge(nu, p + 1), (nu - p - 1) * V, np.nan)
        return super().dist([nu, V], *args, **kwargs)

    def logp(X, nu, V):
        """
        Calculate log-probability of Wishart distribution
        at specified value.

        Parameters
        ----------
        X: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        p = V.shape[0]

        IVI = det(V)
        IXI = det(X)

        return check_parameters(
            (
                (nu - p - 1) * pt.log(IXI)
                - trace(matrix_inverse(V).dot(X))
                - nu * p * pt.log(2)
                - nu * pt.log(IVI)
                - 2 * multigammaln(nu / 2.0, p)
            )
            / 2,
            matrix_pos_def(X),
            pt.eq(X, X.T),
            nu > (p - 1),
        )


def WishartBartlett(name, S, nu, is_cholesky=False, return_cholesky=False, initval=None):
    r"""
    Bartlett decomposition of the Wishart distribution. As the Wishart
    distribution requires the matrix to be symmetric positive semi-definite
    it is impossible for MCMC to ever propose acceptable matrices.

    Instead, we can use the Barlett decomposition which samples a lower
    diagonal matrix. Specifically:

    .. math::
        \text{If} L \sim \begin{pmatrix}
        \sqrt{c_1} & 0 & 0 \\
        z_{21} & \sqrt{c_2} & 0 \\
        z_{31} & z_{32} & \sqrt{c_3}
        \end{pmatrix}

        \text{with} c_i \sim \chi^2(n-i+1) \text{ and } n_{ij} \sim \mathcal{N}(0, 1), \text{then} \\
        L \times A \times A.T \times L.T \sim \text{Wishart}(L \times L.T, \nu)

    See http://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
    for more information.

    Parameters
    ----------
    S : ndarray
        p x p positive definite matrix
        Or:
        p x p lower-triangular matrix that is the Cholesky factor
        of the covariance matrix.
    nu : tensor_like of int
        Degrees of freedom, > dim(S).
    is_cholesky : bool, default=False
        Input matrix S is already Cholesky decomposed as S.T * S
    return_cholesky : bool, default=False
        Only return the Cholesky decomposed matrix.
    initval : ndarray
        p x p positive definite matrix used to initialize

    Notes
    -----
    This is not a standard Distribution class but follows a similar
    interface. Besides the Wishart distribution, it will add RVs
    name_c and name_z to your model which make up the matrix.

    This distribution is usually a bad idea to use as a prior for multivariate
    normal. You should instead use LKJCholeskyCov or LKJCorr.
    """

    L = S if is_cholesky else scipy.linalg.cholesky(S)
    diag_idx = np.diag_indices_from(S)
    tril_idx = np.tril_indices_from(S, k=-1)
    n_diag = len(diag_idx[0])
    n_tril = len(tril_idx[0])

    if initval is not None:
        # Inverse transform
        initval = np.dot(np.dot(np.linalg.inv(L), initval), np.linalg.inv(L.T))
        initval = scipy.linalg.cholesky(initval, lower=True)
        diag_testval = initval[diag_idx] ** 2
        tril_testval = initval[tril_idx]
    else:
        diag_testval = None
        tril_testval = None

    c = pt.sqrt(
        ChiSquared("%s_c" % name, nu - np.arange(2, 2 + n_diag), shape=n_diag, initval=diag_testval)
    )
    pm._log.info("Added new variable %s_c to model diagonal of Wishart." % name)
    z = Normal("%s_z" % name, 0.0, 1.0, shape=n_tril, initval=tril_testval)
    pm._log.info("Added new variable %s_z to model off-diagonals of Wishart." % name)
    # Construct A matrix
    A = pt.zeros(S.shape, dtype=np.float32)
    A = pt.set_subtensor(A[diag_idx], c)
    A = pt.set_subtensor(A[tril_idx], z)

    # L * A * A.T * L.T ~ Wishart(L*L.T, nu)
    if return_cholesky:
        return pm.Deterministic(name, pt.dot(L, A))
    else:
        return pm.Deterministic(name, pt.dot(pt.dot(pt.dot(L, A), A.T), L.T))


def _lkj_normalizing_constant(eta, n):
    # TODO: This is mixing python branching with the potentially symbolic n and eta variables
    if not isinstance(eta, (int, float)):
        raise NotImplementedError("eta must be an int or float")
    if not isinstance(n, int):
        raise NotImplementedError("n must be an integer")
    if eta == 1:
        result = gammaln(2.0 * pt.arange(1, int((n - 1) / 2) + 1)).sum()
        if n % 2 == 1:
            result += (
                0.25 * (n**2 - 1) * pt.log(np.pi)
                - 0.25 * (n - 1) ** 2 * pt.log(2.0)
                - (n - 1) * gammaln(int((n + 1) / 2))
            )
        else:
            result += (
                0.25 * n * (n - 2) * pt.log(np.pi)
                + 0.25 * (3 * n**2 - 4 * n) * pt.log(2.0)
                + n * gammaln(n / 2)
                - (n - 1) * gammaln(n)
            )
    else:
        result = -(n - 1) * gammaln(eta + 0.5 * (n - 1))
        k = pt.arange(1, n)
        result += (0.5 * k * pt.log(np.pi) + gammaln(eta + 0.5 * (n - 1 - k))).sum()
    return result


class _LKJCholeskyCovBaseRV(RandomVariable):
    name = "_lkjcholeskycovbase"
    ndim_supp = 1
    ndims_params = [0, 0, 1]
    dtype = "floatX"
    _print_name = ("_lkjcholeskycovbase", "\\operatorname{_lkjcholeskycovbase}")

    def make_node(self, rng, size, dtype, n, eta, D):
        n = pt.as_tensor_variable(n)
        if not n.ndim == 0:
            raise ValueError("n must be a scalar (ndim=0).")

        eta = pt.as_tensor_variable(eta)
        if not eta.ndim == 0:
            raise ValueError("eta must be a scalar (ndim=0).")

        D = pt.as_tensor_variable(D)

        return super().make_node(rng, size, dtype, n, eta, D)

    def _supp_shape_from_params(self, dist_params, param_shapes):
        n = dist_params[0]
        return ((n * (n + 1)) // 2,)

    def rng_fn(self, rng, n, eta, D, size):
        # We flatten the size to make operations easier, and then rebuild it
        if size is None:
            size = D.shape[:-1]
        flat_size = np.prod(size).astype(int)

        C = LKJCorrRV._random_corr_matrix(rng=rng, n=n, eta=eta, flat_size=flat_size)
        D = D.reshape(flat_size, n)
        C *= D[..., :, np.newaxis] * D[..., np.newaxis, :]

        tril_idx = np.tril_indices(n, k=0)
        samples = np.linalg.cholesky(C)[..., tril_idx[0], tril_idx[1]]

        if size is None:
            samples = samples[0]
        else:
            dist_shape = (n * (n + 1)) // 2
            samples = np.reshape(samples, (*size, dist_shape))

        return samples


_ljk_cholesky_cov_base = _LKJCholeskyCovBaseRV()


# _LKJCholeskyCovBaseRV requires a properly shaped `D`, which means the variable can't
# be safely resized. Because of this, we add the thin SymbolicRandomVariable wrapper
class _LKJCholeskyCovRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("_lkjcholeskycov", "\\operatorname{_lkjcholeskycov}")

    def update(self, node):
        return {node.inputs[0]: node.outputs[0]}


class _LKJCholeskyCov(Distribution):
    r"""Underlying class for covariance matrix with LKJ distributed correlations.
    See docs for LKJCholeskyCov function for more details on how to use it in models.
    """

    rv_type = _LKJCholeskyCovRV

    @classmethod
    def dist(cls, n, eta, sd_dist, **kwargs):
        n = pt.as_tensor_variable(intX(n))
        eta = pt.as_tensor_variable(floatX(eta))

        if not (
            isinstance(sd_dist, Variable)
            and sd_dist.owner is not None
            and isinstance(sd_dist.owner.op, (RandomVariable, SymbolicRandomVariable))
            and sd_dist.owner.op.ndim_supp < 2
        ):
            raise TypeError("sd_dist must be a scalar or vector distribution variable")

        check_dist_not_registered(sd_dist)
        return super().dist([n, eta, sd_dist], **kwargs)

    @classmethod
    def rv_op(cls, n, eta, sd_dist, size=None):
        # We resize the sd_dist automatically so that it has (size x n) independent
        # draws which is what the `_LKJCholeskyCovBaseRV.rng_fn` expects. This makes the
        # random and logp methods equivalent, as the latter also assumes a unique value
        # for each diagonal element.
        # Since `eta` and `n` are forced to be scalars we don't need to worry about
        # implied batched dimensions from those for the time being.
        if size is None:
            size = sd_dist.shape[:-1]
        shape = tuple(size) + (n,)
        if sd_dist.owner.op.ndim_supp == 0:
            sd_dist = change_dist_size(sd_dist, shape)
        else:
            # The support shape must be `n` but we have no way of controlling it
            sd_dist = change_dist_size(sd_dist, shape[:-1])

        # Create new rng for the _lkjcholeskycov internal RV
        rng = pytensor.shared(np.random.default_rng())

        rng_, n_, eta_, sd_dist_ = rng.type(), n.type(), eta.type(), sd_dist.type()
        next_rng_, lkjcov_ = _ljk_cholesky_cov_base(n_, eta_, sd_dist_, rng=rng_).owner.outputs

        return _LKJCholeskyCovRV(
            inputs=[rng_, n_, eta_, sd_dist_],
            outputs=[next_rng_, lkjcov_],
            ndim_supp=1,
        )(rng, n, eta, sd_dist)


@_change_dist_size.register(_LKJCholeskyCovRV)
def change_LKJCholeksyCovRV_size(op, dist, new_size, expand=False):
    n, eta, sd_dist = dist.owner.inputs[1:]

    if expand:
        old_size = sd_dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return _LKJCholeskyCov.rv_op(n, eta, sd_dist, size=new_size)


@_moment.register(_LKJCholeskyCovRV)
def _LKJCholeksyCovRV_moment(op, rv, rng, n, eta, sd_dist):
    diag_idxs = (pt.cumsum(pt.arange(1, n + 1)) - 1).astype("int32")
    moment = pt.zeros_like(rv)
    moment = pt.set_subtensor(moment[..., diag_idxs], 1)
    return moment


@_default_transform.register(_LKJCholeskyCovRV)
def _LKJCholeksyCovRV_default_transform(op, rv):
    _, n, _, _ = rv.owner.inputs
    return transforms.CholeskyCovPacked(n)


@_logprob.register(_LKJCholeskyCovRV)
def _LKJCholeksyCovRV_logp(op, values, rng, n, eta, sd_dist, **kwargs):
    (value,) = values

    if value.ndim > 1:
        raise ValueError("_LKJCholeskyCov logp is only implemented for vector values (ndim=1)")

    diag_idxs = pt.cumsum(pt.arange(1, n + 1)) - 1
    cumsum = pt.cumsum(value**2)
    variance = pt.zeros(pt.atleast_1d(n))
    variance = pt.inc_subtensor(variance[0], value[0] ** 2)
    variance = pt.inc_subtensor(variance[1:], cumsum[diag_idxs[1:]] - cumsum[diag_idxs[:-1]])
    sd_vals = pt.sqrt(variance)

    logp_sd = pm.logp(sd_dist, sd_vals).sum()
    corr_diag = value[diag_idxs] / sd_vals

    logp_lkj = (2 * eta - 3 + n - pt.arange(n)) * pt.log(corr_diag)
    logp_lkj = pt.sum(logp_lkj)

    # Compute the log det jacobian of the second transformation
    # described in the docstring.
    idx = pt.arange(n)
    det_invjac = pt.log(corr_diag) - idx * pt.log(sd_vals)
    det_invjac = det_invjac.sum()

    # TODO: _lkj_normalizing_constant currently requires `eta` and `n` to be constants
    if not isinstance(n, Constant):
        raise NotImplementedError("logp only implemented for constant `n`")
    n = int(n.data)

    if not isinstance(eta, Constant):
        raise NotImplementedError("logp only implemented for constant `eta`")
    eta = float(eta.data)

    norm = _lkj_normalizing_constant(eta, n)

    return norm + logp_lkj + logp_sd + det_invjac


class LKJCholeskyCov:
    r"""Wrapper class for covariance matrix with LKJ distributed correlations.

    This defines a distribution over Cholesky decomposed covariance
    matrices, such that the underlying correlation matrices follow an
    LKJ distribution [1] and the standard deviations follow an arbitrary
    distribution specified by the user.

    Parameters
    ----------
    name : str
        The name given to the variable in the model.
    eta : tensor_like of float
        The shape parameter (eta > 0) of the LKJ distribution. eta = 1
        implies a uniform distribution of the correlation matrices;
        larger values put more weight on matrices with few correlations.
    n : tensor_like of int
        Dimension of the covariance matrix (n > 1).
    sd_dist : Distribution
        A positive scalar or vector distribution for the standard deviations, created
        with the `.dist()` API. Should have `shape[-1]=n`. Scalar distributions will be
        automatically resized to ensure this.

        .. warning:: sd_dist will be cloned, rendering it independent of the one passed as input.

    compute_corr : bool, default=True
        If `True`, returns three values: the Cholesky decomposition, the correlations
        and the standard deviations of the covariance matrix. Otherwise, only returns
        the packed Cholesky decomposition. Defaults to `True`.
        compatibility.
    store_in_trace : bool, default=True
        Whether to store the correlations and standard deviations of the covariance
        matrix in the posterior trace. If `True`, they will automatically be named as
        `{name}_corr` and `{name}_stds` respectively. Effective only when
        `compute_corr=True`.

    Returns
    -------
    chol :  TensorVariable
        If `compute_corr=True`. The unpacked Cholesky covariance decomposition.
    corr : TensorVariable
        If `compute_corr=True`. The correlations of the covariance matrix.
    stds : TensorVariable
        If `compute_corr=True`. The standard deviations of the covariance matrix.
    packed_chol : TensorVariable
        If `compute_corr=False` The packed Cholesky covariance decomposition.

    Notes
    -----
    Since the Cholesky factor is a lower triangular matrix, we use packed storage for
    the matrix: We store the values of the lower triangular matrix in a one-dimensional
    array, numbered by row::

        [[0 - - -]
         [1 2 - -]
         [3 4 5 -]
         [6 7 8 9]]

    The unpacked Cholesky covariance matrix is automatically computed and returned when
    you specify `compute_corr=True` in `pm.LKJCholeskyCov` (see example below).
    Otherwise, you can use `pm.expand_packed_triangular(packed_cov, lower=True)`
    to convert the packed Cholesky matrix to a regular two-dimensional array.

    Examples
    --------
    .. code:: python

        with pm.Model() as model:
            # Note that we access the distribution for the standard
            # deviations, and do not create a new random variable.
            sd_dist = pm.Exponential.dist(1.0, size=10)
            chol, corr, sigmas = pm.LKJCholeskyCov(
                'chol_cov', eta=4, n=10, sd_dist=sd_dist
            )

            # if you only want the packed Cholesky:
            # packed_chol = pm.LKJCholeskyCov(
                'chol_cov', eta=4, n=10, sd_dist=sd_dist, compute_corr=False
            )
            # chol = pm.expand_packed_triangular(10, packed_chol, lower=True)

            # Define a new MvNormal with the given covariance
            vals = pm.MvNormal('vals', mu=np.zeros(10), chol=chol, shape=10)

            # Or transform an uncorrelated normal:
            vals_raw = pm.Normal('vals_raw', mu=0, sigma=1, shape=10)
            vals = pt.dot(chol, vals_raw)

            # Or compute the covariance matrix
            cov = pt.dot(chol, chol.T)

    **Implementation** In the unconstrained space all values of the cholesky factor
    are stored untransformed, except for the diagonal entries, where
    we use a log-transform to restrict them to positive values.

    To correctly compute log-likelihoods for the standard deviations
    and the correlation matrix separately, we need to consider a
    second transformation: Given a cholesky factorization
    :math:`LL^T = \Sigma` of a covariance matrix we can recover the
    standard deviations :math:`\sigma` as the euclidean lengths of
    the rows of :math:`L`, and the cholesky factor of the
    correlation matrix as :math:`U = \text{diag}(\sigma)^{-1}L`.
    Since each row of :math:`U` has length 1, we do not need to
    store the diagonal. We define a transformation :math:`\phi`
    such that :math:`\phi(L)` is the lower triangular matrix containing
    the standard deviations :math:`\sigma` on the diagonal and the
    correlation matrix :math:`U` below. In this form we can easily
    compute the different likelihoods separately, as the likelihood
    of the correlation matrix only depends on the values below the
    diagonal, and the likelihood of the standard deviation depends
    only on the diagonal values.

    We still need the determinant of the jacobian of :math:`\phi^{-1}`.
    If we think of :math:`\phi` as an automorphism on
    :math:`\mathbb{R}^{\tfrac{n(n+1)}{2}}`, where we order
    the dimensions as described in the notes above, the jacobian
    is a block-diagonal matrix, where each block corresponds to
    one row of :math:`U`. Each block has arrowhead shape, and we
    can compute the determinant of that as described in [2]. Since
    the determinant of a block-diagonal matrix is the product
    of the determinants of the blocks, we get

    .. math::

       \text{det}(J_{\phi^{-1}}(U)) =
       \left[
         \prod_{i=2}^N u_{ii}^{i - 1} L_{ii}
       \right]^{-1}

    References
    ----------
    .. [1] Lewandowski, D., Kurowicka, D. and Joe, H. (2009).
       "Generating random correlation matrices based on vines and
       extended onion method." Journal of multivariate analysis,
       100(9), pp.1989-2001.

    .. [2] J. M. isn't a mathematician (http://math.stackexchange.com/users/498/
       j-m-isnt-a-mathematician), Different approaches to evaluate this
       determinant, URL (version: 2012-04-14):
       http://math.stackexchange.com/q/130026
    """

    def __new__(cls, name, eta, n, sd_dist, *, compute_corr=True, store_in_trace=True, **kwargs):
        packed_chol = _LKJCholeskyCov(name, eta=eta, n=n, sd_dist=sd_dist, **kwargs)
        if not compute_corr:
            return packed_chol
        else:
            chol, corr, stds = cls.helper_deterministics(n, packed_chol)
            if store_in_trace:
                corr = pm.Deterministic(f"{name}_corr", corr)
                stds = pm.Deterministic(f"{name}_stds", stds)
            return chol, corr, stds

    @classmethod
    def dist(cls, eta, n, sd_dist, *, compute_corr=True, **kwargs):
        # compute Cholesky decomposition
        packed_chol = _LKJCholeskyCov.dist(eta=eta, n=n, sd_dist=sd_dist, **kwargs)
        if not compute_corr:
            return packed_chol
        else:
            return cls.helper_deterministics(n, packed_chol)

    @classmethod
    def helper_deterministics(cls, n, packed_chol):
        chol = pm.expand_packed_triangular(n, packed_chol, lower=True)
        # compute covariance matrix
        cov = pt.dot(chol, chol.T)
        # extract standard deviations and rho
        stds = pt.sqrt(pt.diag(cov))
        inv_stds = 1 / stds
        corr = inv_stds[None, :] * cov * inv_stds[:, None]
        return chol, corr, stds


class LKJCorrRV(RandomVariable):
    name = "lkjcorr"
    ndim_supp = 1
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("LKJCorrRV", "\\operatorname{LKJCorrRV}")

    def make_node(self, rng, size, dtype, n, eta):
        n = pt.as_tensor_variable(n)
        if not n.ndim == 0:
            raise ValueError("n must be a scalar (ndim=0).")

        eta = pt.as_tensor_variable(eta)
        if not eta.ndim == 0:
            raise ValueError("eta must be a scalar (ndim=0).")

        return super().make_node(rng, size, dtype, n, eta)

    def _supp_shape_from_params(self, dist_params, **kwargs):
        n = dist_params[0]
        dist_shape = ((n * (n - 1)) // 2,)
        return dist_shape

    @classmethod
    def rng_fn(cls, rng, n, eta, size):
        # We flatten the size to make operations easier, and then rebuild it
        if size is None:
            flat_size = 1
        else:
            flat_size = np.prod(size)

        C = cls._random_corr_matrix(rng=rng, n=n, eta=eta, flat_size=flat_size)

        triu_idx = np.triu_indices(n, k=1)
        samples = C[..., triu_idx[0], triu_idx[1]]

        if size is None:
            samples = samples[0]
        else:
            dist_shape = (n * (n - 1)) // 2
            samples = np.reshape(samples, (*size, dist_shape))
        return samples

    @classmethod
    def _random_corr_matrix(cls, rng, n, eta, flat_size):
        # original implementation in R see:
        # https://github.com/rmcelreath/rethinking/blob/master/R/distributions.r
        beta = eta - 1.0 + n / 2.0
        r12 = 2.0 * stats.beta.rvs(a=beta, b=beta, size=flat_size, random_state=rng) - 1.0
        P = np.full((flat_size, n, n), np.eye(n))
        P[..., 0, 1] = r12
        P[..., 1, 1] = np.sqrt(1.0 - r12**2)
        for mp1 in range(2, n):
            beta -= 0.5
            y = stats.beta.rvs(a=mp1 / 2.0, b=beta, size=flat_size, random_state=rng)
            z = stats.norm.rvs(loc=0, scale=1, size=(flat_size, mp1), random_state=rng)
            z = z / np.sqrt(np.einsum("ij,ij->i", z, z))[..., np.newaxis]
            P[..., 0:mp1, mp1] = np.sqrt(y[..., np.newaxis]) * z
            P[..., mp1, mp1] = np.sqrt(1.0 - y)
        C = np.einsum("...ji,...jk->...ik", P, P)
        return C


lkjcorr = LKJCorrRV()


class LKJCorr(BoundedContinuous):
    r"""
    The LKJ (Lewandowski, Kurowicka and Joe) log-likelihood.

    The LKJ distribution is a prior distribution for correlation matrices.
    If eta = 1 this corresponds to the uniform distribution over correlation
    matrices. For eta -> oo the LKJ prior approaches the identity matrix.

    ========  ==============================================
    Support   Upper triangular matrix with values in [-1, 1]
    ========  ==============================================

    Parameters
    ----------
    n : tensor_like of int
        Dimension of the covariance matrix (n > 1).
    eta : tensor_like of float
        The shape parameter (eta > 0) of the LKJ distribution. eta = 1
        implies a uniform distribution of the correlation matrices;
        larger values put more weight on matrices with few correlations.

    Notes
    -----
    This implementation only returns the values of the upper triangular
    matrix excluding the diagonal. Here is a schematic for n = 5, showing
    the indexes of the elements::

        [[- 0 1 2 3]
         [- - 4 5 6]
         [- - - 7 8]
         [- - - - 9]
         [- - - - -]]


    References
    ----------
    .. [LKJ2009] Lewandowski, D., Kurowicka, D. and Joe, H. (2009).
        "Generating random correlation matrices based on vines and
        extended onion method." Journal of multivariate analysis,
        100(9), pp.1989-2001.
    """

    rv_op = lkjcorr

    @classmethod
    def dist(cls, n, eta, **kwargs):
        n = pt.as_tensor_variable(intX(n))
        eta = pt.as_tensor_variable(floatX(eta))
        return super().dist([n, eta], **kwargs)

    def moment(rv, *args):
        return pt.zeros_like(rv)

    def logp(value, n, eta):
        """
        Calculate log-probability of LKJ distribution at specified
        value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        # TODO: PyTensor does not have a `triu_indices`, so we can only work with constant
        #  n (or else find a different expression)
        if not isinstance(n, Constant):
            raise NotImplementedError("logp only implemented for constant `n`")

        n = int(n.data)
        shape = n * (n - 1) // 2
        tri_index = np.zeros((n, n), dtype="int32")
        tri_index[np.triu_indices(n, k=1)] = np.arange(shape)
        tri_index[np.triu_indices(n, k=1)[::-1]] = np.arange(shape)

        value = pt.take(value, tri_index)
        value = pt.fill_diagonal(value, 1)

        # TODO: _lkj_normalizing_constant currently requires `eta` and `n` to be constants
        if not isinstance(eta, Constant):
            raise NotImplementedError("logp only implemented for constant `eta`")
        eta = float(eta.data)
        result = _lkj_normalizing_constant(eta, n)
        result += (eta - 1.0) * pt.log(det(value))
        return check_parameters(
            result,
            value >= -1,
            value <= 1,
            matrix_pos_def(value),
            eta > 0,
        )


@_default_transform.register(LKJCorr)
def lkjcorr_default_transform(op, rv):
    return Interval(floatX(-1.0), floatX(1.0))


class MatrixNormalRV(RandomVariable):
    name = "matrixnormal"
    ndim_supp = 2
    ndims_params = [2, 2, 2]
    dtype = "floatX"
    _print_name = ("MatrixNormal", "\\operatorname{MatrixNormal}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=0,
        )

    @classmethod
    def rng_fn(cls, rng, mu, rowchol, colchol, size=None):
        size = to_tuple(size)
        dist_shape = to_tuple([rowchol.shape[0], colchol.shape[0]])
        output_shape = size + dist_shape

        # Broadcasting all parameters
        shapes = [mu.shape, output_shape]
        broadcastable_shape = broadcast_dist_samples_shape(shapes, size=size)
        mu = np.broadcast_to(mu, shape=broadcastable_shape)
        rowchol = np.broadcast_to(rowchol, shape=size + rowchol.shape[-2:])

        colchol = np.broadcast_to(colchol, shape=size + colchol.shape[-2:])
        colchol = np.swapaxes(colchol, -1, -2)  # Take transpose

        standard_normal = rng.standard_normal(output_shape)
        samples = mu + np.matmul(rowchol, np.matmul(standard_normal, colchol))

        return samples


matrixnormal = MatrixNormalRV()


class MatrixNormal(Continuous):
    r"""
    Matrix-valued normal log-likelihood.

    .. math::
       f(x \mid \mu, U, V) =
           \frac{1}{(2\pi^{m n} |U|^n |V|^m)^{1/2}}
           \exp\left\{
                -\frac{1}{2} \mathrm{Tr}[ V^{-1} (x-\mu)^{\prime} U^{-1} (x-\mu)]
            \right\}

    ===============  =====================================
    Support          :math:`x \in \mathbb{R}^{m \times n}`
    Mean             :math:`\mu`
    Row Variance     :math:`U`
    Column Variance  :math:`V`
    ===============  =====================================

    Parameters
    ----------
    mu : tensor_like of float
        Array of means. Must be broadcastable with the random variable X such
        that the shape of mu + X is (M, N).
    rowcov : (M, M) tensor_like of float, optional
        Among-row covariance matrix. Defines variance within
        columns. Exactly one of rowcov or rowchol is needed.
    rowchol : (M, M) tensor_like of float, optional
        Cholesky decomposition of among-row covariance matrix. Exactly one of
        rowcov or rowchol is needed.
    colcov : (N, N) tensor_like of float, optional
        Among-column covariance matrix. If rowcov is the identity matrix,
        this functions as `cov` in MvNormal.
        Exactly one of colcov or colchol is needed.
    colchol : (N, N) tensor_like of float, optional
        Cholesky decomposition of among-column covariance matrix. Exactly one
        of colcov or colchol is needed.

    Examples
    --------
    Define a matrixvariate normal variable for given row and column covariance
    matrices::

        colcov = np.array([[1., 0.5], [0.5, 2]])
        rowcov = np.array([[1, 0, 0], [0, 4, 0], [0, 0, 16]])
        m = rowcov.shape[0]
        n = colcov.shape[0]
        mu = np.zeros((m, n))
        vals = pm.MatrixNormal('vals', mu=mu, colcov=colcov,
                               rowcov=rowcov)

    Above, the ith row in vals has a variance that is scaled by 4^i.
    Alternatively, row or column cholesky matrices could be substituted for
    either covariance matrix. The MatrixNormal is quicker way compute
    MvNormal(mu, np.kron(rowcov, colcov)) that takes advantage of kronecker product
    properties for inversion. For example, if draws from MvNormal had the same
    covariance structure, but were scaled by different powers of an unknown
    constant, both the covariance and scaling could be learned as follows
    (see the docstring of `LKJCholeskyCov` for more information about this)

    .. code:: python

        # Setup data
        true_colcov = np.array([[1.0, 0.5, 0.1],
                                [0.5, 1.0, 0.2],
                                [0.1, 0.2, 1.0]])
        m = 3
        n = true_colcov.shape[0]
        true_scale = 3
        true_rowcov = np.diag([true_scale**(2*i) for i in range(m)])
        mu = np.zeros((m, n))
        true_kron = np.kron(true_rowcov, true_colcov)
        data = np.random.multivariate_normal(mu.flatten(), true_kron)
        data = data.reshape(m, n)

        with pm.Model() as model:
            # Setup right cholesky matrix
            sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
            colchol_packed = pm.LKJCholeskyCov('colcholpacked', n=3, eta=2,
                                               sd_dist=sd_dist)
            colchol = pm.expand_packed_triangular(3, colchol_packed)

            # Setup left covariance matrix
            scale = pm.LogNormal('scale', mu=np.log(true_scale), sigma=0.5)
            rowcov = pt.diag([scale**(2*i) for i in range(m)])

            vals = pm.MatrixNormal('vals', mu=mu, colchol=colchol, rowcov=rowcov,
                                   observed=data)
    """
    rv_op = matrixnormal

    @classmethod
    def dist(
        cls,
        mu,
        rowcov=None,
        rowchol=None,
        colcov=None,
        colchol=None,
        *args,
        **kwargs,
    ):
        lower_cholesky = partial(cholesky, lower=True, on_error="raise")

        # Among-row matrices
        if len([i for i in [rowcov, rowchol] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. Specify exactly one of rowcov, or rowchol."
            )
        if rowcov is not None:
            if rowcov.ndim != 2:
                raise ValueError("rowcov must be two dimensional.")
            rowchol_cov = lower_cholesky(rowcov)
        else:
            if rowchol.ndim != 2:
                raise ValueError("rowchol must be two dimensional.")
            rowchol_cov = pt.as_tensor_variable(rowchol)

        # Among-column matrices
        if len([i for i in [colcov, colchol] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. Specify exactly one of colcov, or colchol."
            )
        if colcov is not None:
            colcov = pt.as_tensor_variable(colcov)
            if colcov.ndim != 2:
                raise ValueError("colcov must be two dimensional.")
            colchol_cov = lower_cholesky(colcov)
        else:
            if colchol.ndim != 2:
                raise ValueError("colchol must be two dimensional.")
            colchol_cov = pt.as_tensor_variable(colchol)

        dist_shape = (rowchol_cov.shape[-1], colchol_cov.shape[-1])

        # Broadcasting mu
        mu = pt.extra_ops.broadcast_to(mu, shape=dist_shape)
        mu = pt.as_tensor_variable(floatX(mu))

        return super().dist([mu, rowchol_cov, colchol_cov], **kwargs)

    def moment(rv, size, mu, rowchol, colchol):
        return pt.full_like(rv, mu)

    def logp(value, mu, rowchol, colchol):
        """
        Calculate log-probability of Matrix-valued Normal distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        if value.ndim != 2:
            raise ValueError("Value must be two dimensional.")

        # Compute Tr[colcov^-1 @ (x - mu).T @ rowcov^-1 @ (x - mu)] and
        # the logdet of colcov and rowcov.
        delta = value - mu

        # Find exponent piece by piece
        right_quaddist = solve_lower(rowchol, delta)
        quaddist = pt.linalg.matrix_dot(right_quaddist.T, right_quaddist)
        quaddist = solve_lower(colchol, quaddist)
        quaddist = solve_upper(colchol.T, quaddist)
        trquaddist = pt.linalg.trace(quaddist)

        coldiag = pt.diag(colchol)
        rowdiag = pt.diag(rowchol)
        half_collogdet = pt.sum(pt.log(coldiag))  # logdet(M) = 2*Tr(log(L))
        half_rowlogdet = pt.sum(pt.log(rowdiag))  # Using Cholesky: M = L L^T

        m = rowchol.shape[0]
        n = colchol.shape[0]

        norm = -0.5 * m * n * pm.floatX(np.log(2 * np.pi))
        return norm - 0.5 * trquaddist - m * half_collogdet - n * half_rowlogdet


class KroneckerNormalRV(RandomVariable):
    name = "kroneckernormal"
    ndim_supp = 1
    ndims_params = [1, 0, 2]
    dtype = "floatX"
    _print_name = ("KroneckerNormal", "\\operatorname{KroneckerNormal}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=0,
        )

    def rng_fn(self, rng, mu, sigma, *covs, size=None):
        size = size if size else covs[-1]
        covs = covs[:-1] if covs[-1] == size else covs

        cov = reduce(scipy.linalg.kron, covs)

        if sigma:
            cov = cov + sigma**2 * np.eye(cov.shape[0])

        x = multivariate_normal.rng_fn(rng=rng, mean=mu, cov=cov, size=size)
        return x


kroneckernormal = KroneckerNormalRV()


class KroneckerNormal(Continuous):
    r"""
    Multivariate normal log-likelihood with Kronecker-structured covariance.

    .. math::

       f(x \mid \mu, K) =
           \frac{1}{(2\pi |K|)^{1/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} K^{-1} (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^N`
    Mean      :math:`\mu`
    Variance  :math:`K = \bigotimes K_i + \sigma^2 I_N`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Vector of means, just as in `MvNormal`.
    covs : list of arrays
        The set of covariance matrices :math:`[K_1, K_2, ...]` to be
        Kroneckered in the order provided :math:`\bigotimes K_i`.
    chols : list of arrays
        The set of lower cholesky matrices :math:`[L_1, L_2, ...]` such that
        :math:`K_i = L_i L_i'`.
    evds : list of tuples
        The set of eigenvalue-vector, eigenvector-matrix pairs
        :math:`[(v_1, Q_1), (v_2, Q_2), ...]` such that
        :math:`K_i = Q_i \text{diag}(v_i) Q_i'`. For example::

            v_i, Q_i = pt.linalg.eigh(K_i)
    sigma : scalar, optional
        Standard deviation of the Gaussian white noise.

    Examples
    --------
    Define a multivariate normal variable with a covariance
    :math:`K = K_1 \otimes K_2`

    .. code:: python

        K1 = np.array([[1., 0.5], [0.5, 2]])
        K2 = np.array([[1., 0.4, 0.2], [0.4, 2, 0.3], [0.2, 0.3, 1]])
        covs = [K1, K2]
        N = 6
        mu = np.zeros(N)
        with pm.Model() as model:
            vals = pm.KroneckerNormal('vals', mu=mu, covs=covs, shape=N)

    Efficiency gains are made by cholesky decomposing :math:`K_1` and
    :math:`K_2` individually rather than the larger :math:`K` matrix. Although
    only two matrices :math:`K_1` and :math:`K_2` are shown here, an arbitrary
    number of submatrices can be combined in this way. Choleskys and
    eigendecompositions can be provided instead

    .. code:: python

        chols = [np.linalg.cholesky(Ki) for Ki in covs]
        evds = [np.linalg.eigh(Ki) for Ki in covs]
        with pm.Model() as model:
            vals2 = pm.KroneckerNormal('vals2', mu=mu, chols=chols, shape=N)
            # or
            vals3 = pm.KroneckerNormal('vals3', mu=mu, evds=evds, shape=N)

    neither of which will be converted. Diagonal noise can also be added to
    the covariance matrix, :math:`K = K_1 \otimes K_2 + \sigma^2 I_N`.
    Despite the noise removing the overall Kronecker structure of the matrix,
    `KroneckerNormal` can continue to make efficient calculations by
    utilizing eigendecompositons of the submatrices behind the scenes [1].
    Thus,

    .. code:: python

        sigma = 0.1
        with pm.Model() as noise_model:
            vals = pm.KroneckerNormal('vals', mu=mu, covs=covs, sigma=sigma, shape=N)
            vals2 = pm.KroneckerNormal('vals2', mu=mu, chols=chols, sigma=sigma, shape=N)
            vals3 = pm.KroneckerNormal('vals3', mu=mu, evds=evds, sigma=sigma, shape=N)

    are identical, with `covs` and `chols` each converted to
    eigendecompositions.

    References
    ----------
    .. [1] Saatchi, Y. (2011). "Scalable inference for structured Gaussian process models"
    """
    rv_op = kroneckernormal

    @classmethod
    def dist(cls, mu, covs=None, chols=None, evds=None, sigma=None, *args, **kwargs):
        if len([i for i in [covs, chols, evds] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. Specify exactly one of covs, chols, or evds."
            )

        sigma = sigma if sigma else 0

        if chols is not None:
            covs = [chol.dot(chol.T) for chol in chols]
        elif evds is not None:
            eigh_iterable = evds
            covs = []
            eigs_sep, Qs = zip(*eigh_iterable)  # Unzip
            for eig, Q in zip(eigs_sep, Qs):
                cov_i = pt.dot(Q, pt.dot(pt.diag(eig), Q.T))
                covs.append(cov_i)

        mu = pt.as_tensor_variable(mu)

        return super().dist([mu, sigma, *covs], **kwargs)

    def moment(rv, size, mu, covs, chols, evds):
        mean = mu
        if not rv_size_is_none(size):
            moment_size = pt.concatenate([size, mu.shape])
            mean = pt.full(moment_size, mu)
        return mean

    def logp(value, mu, sigma, *covs):
        """
        Calculate log-probability of Multivariate Normal distribution
        with Kronecker-structured covariance at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        # Computes the quadratic (x-mu)^T @ K^-1 @ (x-mu) and log(det(K))
        if value.ndim > 2 or value.ndim == 0:
            raise ValueError(f"Invalid dimension for value: {value.ndim}")
        if value.ndim == 1:
            onedim = True
            value = value[None, :]
        else:
            onedim = False

        delta = value - mu

        eigh_iterable = map(eigh, covs)
        eigs_sep, Qs = zip(*eigh_iterable)  # Unzip
        Qs = list(map(pt.as_tensor_variable, Qs))
        QTs = list(map(pt.transpose, Qs))

        eigs_sep = list(map(pt.as_tensor_variable, eigs_sep))
        eigs = kron_diag(*eigs_sep)  # Combine separate eigs
        eigs += sigma**2
        N = eigs.shape[0]

        sqrt_quad = kron_dot(QTs, delta.T)
        sqrt_quad = sqrt_quad / pt.sqrt(eigs[:, None])
        logdet = pt.sum(pt.log(eigs))

        # Square each sample
        quad = pt.batched_dot(sqrt_quad.T, sqrt_quad.T)
        if onedim:
            quad = quad[0]

        a = -(quad + logdet + N * pt.log(2 * np.pi)) / 2.0
        return a


class CARRV(RandomVariable):
    name = "car"
    ndim_supp = 1
    ndims_params = [1, 2, 0, 0]
    dtype = "floatX"
    _print_name = ("CAR", "\\operatorname{CAR}")

    def make_node(self, rng, size, dtype, mu, W, alpha, tau):
        mu = pt.as_tensor_variable(floatX(mu))

        W = pytensor.sparse.as_sparse_or_tensor_variable(floatX(W))
        if not W.ndim == 2:
            raise ValueError("W must be a matrix (ndim=2).")

        sparse = isinstance(W, pytensor.sparse.SparseVariable)
        msg = "W must be a symmetric adjacency matrix."
        if sparse:
            abs_diff = pytensor.sparse.basic.mul(pytensor.sparse.sign(W - W.T), W - W.T)
            W = Assert(msg)(W, pt.isclose(pytensor.sparse.sp_sum(abs_diff), 0))
        else:
            W = Assert(msg)(W, pt.allclose(W, W.T))

        tau = pt.as_tensor_variable(floatX(tau))

        alpha = pt.as_tensor_variable(floatX(alpha))

        return super().make_node(rng, size, dtype, mu, W, alpha, tau)

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=0,
        )

    @classmethod
    def rng_fn(cls, rng: np.random.RandomState, mu, W, alpha, tau, size):
        """
        Implementation of algorithm from paper
        Havard Rue, 2001. "Fast sampling of Gaussian Markov random fields,"
        Journal of the Royal Statistical Society Series B, Royal Statistical Society,
        vol. 63(2), pages 325-338. DOI: 10.1111/1467-9868.00288
        """
        if np.any(alpha >= 1) or np.any(alpha <= -1):
            raise ValueError("the domain of alpha is: -1 < alpha < 1")

        if not scipy.sparse.issparse(W):
            W = scipy.sparse.csr_matrix(W)
        s = np.asarray(W.sum(axis=0))[0]
        D = scipy.sparse.diags(s)
        tau = scipy.sparse.csr_matrix(tau)
        alpha = scipy.sparse.csr_matrix(alpha)

        Q = tau.multiply(D - alpha.multiply(W))

        perm_array = scipy.sparse.csgraph.reverse_cuthill_mckee(Q, symmetric_mode=True)
        inv_perm = np.argsort(perm_array)

        Q = Q[perm_array, :][:, perm_array]

        Qb = Q.diagonal()
        u = 1
        while np.count_nonzero(Q.diagonal(u)) > 0:
            Qb = np.vstack((np.pad(Q.diagonal(u), (u, 0), constant_values=(0, 0)), Qb))
            u += 1

        L = scipy.linalg.cholesky_banded(Qb, lower=False)

        size = tuple(size or ())
        if size:
            mu = np.broadcast_to(mu, size + (mu.shape[-1],))
        z = rng.normal(size=mu.shape)
        samples = np.empty(z.shape)
        for idx in np.ndindex(mu.shape[:-1]):
            samples[idx] = scipy.linalg.cho_solve_banded((L, False), z[idx]) + mu[idx][perm_array]
        samples = samples[..., inv_perm]
        return samples


car = CARRV()


class CAR(Continuous):
    r"""
    Likelihood for a conditional autoregression. This is a special case of the
    multivariate normal with an adjacency-structured covariance matrix.

    .. math::

       f(x \mid W, \alpha, \tau) =
           \frac{|T|^{1/2}}{(2\pi)^{k/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T^{-1} (x-\mu) \right\}

    where :math:`T = (\tau D(I-\alpha W))^{-1}` and :math:`D = diag(\sum_i W_{ij})`.

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu \in \mathbb{R}^k`
    Variance  :math:`(\tau D(I-\alpha W))^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Real-valued mean vector
    W : (M, M) tensor_like of int
        Symmetric adjacency matrix of 1s and 0s indicating
        adjacency between elements. If possible, *W* is converted
        to a sparse matrix, falling back to a dense variable.
        :func:`~pytensor.sparse.basic.as_sparse_or_tensor_variable` is
        used for this sparse or tensorvariable conversion.
    alpha : tensor_like of float
        Autoregression parameter taking values greater than -1 and less than 1.
        Values closer to 0 indicate weaker correlation and values closer to
        1 indicate higher autocorrelation. For most use cases, the
        support of alpha should be restricted to (0, 1).
    tau : tensor_like of float
        Positive precision variable controlling the scale of the underlying normal variates.

    References
    ----------
    ..  Jin, X., Carlin, B., Banerjee, S.
        "Generalized Hierarchical Multivariate CAR Models for Areal Data"
        Biometrics, Vol. 61, No. 4 (Dec., 2005), pp. 950-961
    """
    rv_op = car

    @classmethod
    def dist(cls, mu, W, alpha, tau, *args, **kwargs):
        return super().dist([mu, W, alpha, tau], **kwargs)

    def moment(rv, size, mu, W, alpha, tau):
        return pt.full_like(rv, mu)

    def logp(value, mu, W, alpha, tau):
        """
        Calculate log-probability of a CAR-distributed vector
        at specified value. This log probability function differs from
        the true CAR log density (AKA a multivariate normal with CAR-structured
        covariance matrix) by an additive constant.

        Parameters
        ----------
        value: array
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        sparse = isinstance(W, (pytensor.sparse.SparseConstant, pytensor.sparse.SparseVariable))

        if sparse:
            D = sp_sum(W, axis=0)
            Dinv_sqrt = pt.diag(1 / pt.sqrt(D))
            DWD = pt.dot(pytensor.sparse.dot(Dinv_sqrt, W), Dinv_sqrt)
        else:
            D = W.sum(axis=0)
            Dinv_sqrt = pt.diag(1 / pt.sqrt(D))
            DWD = pt.dot(pt.dot(Dinv_sqrt, W), Dinv_sqrt)
        lam = pt.linalg.eigvalsh(DWD, pt.eye(DWD.shape[0]))

        d, _ = W.shape

        if value.ndim == 1:
            value = value[None, :]

        logtau = d * pt.log(tau).sum()
        logdet = pt.log(1 - alpha.T * lam[:, None]).sum()
        delta = value - mu

        if sparse:
            Wdelta = pytensor.sparse.dot(delta, W)
        else:
            Wdelta = pt.dot(delta, W)

        tau_dot_delta = D[None, :] * delta - alpha * Wdelta
        logquad = (tau * delta * tau_dot_delta).sum(axis=-1)
        return check_parameters(
            0.5 * (logtau + logdet - logquad),
            -1 < alpha,
            alpha < 1,
            tau > 0,
            msg="-1 < alpha < 1, tau > 0",
        )


class ICARRV(RandomVariable):
    name = "icar"
    ndim_supp = 1
    ndims_params = [2, 1, 1, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ICAR", "\\operatorname{ICAR}")

    def __call__(self, W, node1, node2, N, sigma, zero_sum_stdev, size=None, **kwargs):
        return super().__call__(W, node1, node2, N, sigma, zero_sum_stdev, size=size, **kwargs)

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return supp_shape_from_ref_param_shape(
            ndim_supp=self.ndim_supp,
            dist_params=dist_params,
            param_shapes=param_shapes,
            ref_param_idx=0,
        )

    @classmethod
    def rng_fn(cls, rng, size, W, node1, node2, N, sigma, zero_sum_stdev):
        raise NotImplementedError("Cannot sample from ICAR prior")


icar = ICARRV()


class ICAR(Continuous):
    r"""
    The intrinsic conditional autoregressive prior. It is primarily used to model
    covariance between neighboring areas. It is a special case
    of the :class:`~pymc.CAR` distribution where alpha is set to 1.

    The log probability density function is

    .. math::
        f(\phi| W,\sigma) =
          -\frac{1}{2\sigma^{2}} \sum_{i\sim j} (\phi_{i} - \phi_{j})^2 -
          \frac{1}{2}*\frac{\sum_{i}{\phi_{i}}}{0.001N}^{2} - \ln{\sqrt{2\\pi}} -
          \ln{0.001N}

    The first term represents the spatial covariance component. Each $\\phi_{i}$ is penalized
    based on the square distance from each of its neighbors. The notation $i\\sim j$
    indicates a sum over all the neighbors of $\\phi_{i}$. The last three terms are the
    Normal log density function where the mean is zero and the standard deviation is
    $N * 0.001$ (where N is the length of the vector $\\phi$). This component imposes
    a zero-sum constraint by finding the sum of the vector $\\phi$ and penalizing based
    on its distance from zero.

    Parameters
    ----------
    W : ndarray of int
        Symmetric adjacency matrix of 1s and 0s indicating adjacency between elements.

    sigma : scalar, default 1
        Standard deviation of the vector of phi's. Putting a prior on sigma
        will result in a centered parameterization. In most cases, it is
        preferable to use a non-centered parameterization by using the default
        value and multiplying the resulting phi's by sigma. See the example below.

    zero_sum_stdev : scalar, default 0.001
        Controls how strongly to enforce the zero-sum constraint. The sum of
        phi is normally distributed with a mean of zero and small standard deviation.
        This parameter sets the standard deviation of a normal density function with
        mean zero.


    Examples
    --------
    This example illustrates how to switch between centered and non-centered
    parameterizations.

    .. code-block:: python

        import numpy as np
        import pymc as pm

        # 4x4 adjacency matrix
        # arranged in a square lattice

        W = np.array([
            [0,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0]
        ])

        # centered parameterization
        with pm.Model():
            sigma = pm.Exponential('sigma', 1)
            phi = pm.ICAR('phi', W=W, sigma=sigma)
            mu = phi

        # non-centered parameterization
        with pm.Model():
            sigma = pm.Exponential('sigma', 1)
            phi = pm.ICAR('phi', W=W)
            mu = sigma * phi

    References
    ----------
    ..  Mitzi, M., Wheeler-Martin, K., Simpson, D., Mooney, J. S.,
        Gelman, A., Dimaggio, C.
        "Bayesian hierarchical spatial models: Implementing the Besag York
        Mollié model in stan"
        Spatial and Spatio-temporal Epidemiology, Vol. 31, (Aug., 2019),
        pp 1-18
    ..  Banerjee, S., Carlin, B., Gelfand, A. Hierarchical Modeling
        and Analysis for Spatial Data. Second edition. CRC press. (2015)

    """

    rv_op = icar

    @classmethod
    def dist(cls, W, sigma=1, zero_sum_stdev=0.001, **kwargs):
        if not W.ndim == 2:
            raise ValueError("W must be matrix with ndim=2")

        if not W.shape[0] == W.shape[1]:
            raise ValueError("W must be a square matrix")

        if not np.allclose(W.T, W):
            raise ValueError("W must be a symmetric matrix")

        if np.any((W != 0) & (W != 1)):
            raise ValueError("W must be composed of only 1s and 0s")

        # convert adjacency matrix to edgelist representation
        # An edgelist is a pair of lists.
        # If node i and node j are connected then one list
        # will contain i and the other will contain j at the same
        # index value.
        # We only use the lower triangle here because adjacency
        # is a undirected connection.

        node1, node2 = np.where(np.tril(W) == 1)

        node1 = pt.as_tensor_variable(node1, dtype=int)
        node2 = pt.as_tensor_variable(node2, dtype=int)

        W = pt.as_tensor_variable(W, dtype=int)

        N = pt.shape(W)[0]
        N = pt.as_tensor_variable(N)

        sigma = pt.as_tensor_variable(floatX(sigma))
        zero_sum_stdev = pt.as_tensor_variable(floatX(zero_sum_stdev))

        return super().dist([W, node1, node2, N, sigma, zero_sum_stdev], **kwargs)

    def moment(rv, size, W, node1, node2, N, sigma, zero_sum_stdev):
        return pt.zeros(N)

    def logp(value, W, node1, node2, N, sigma, zero_sum_stdev):
        pairwise_difference = (-1 / (2 * sigma**2)) * pt.sum(
            pt.square(value[node1] - value[node2])
        )
        zero_sum = (
            -0.5 * pt.pow(pt.sum(value) / (zero_sum_stdev * N), 2)
            - pt.log(pt.sqrt(2.0 * np.pi))
            - pt.log(zero_sum_stdev * N)
        )

        return check_parameters(pairwise_difference + zero_sum, sigma > 0, msg="sigma > 0")


class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("StickBreakingWeights", "\\operatorname{StickBreakingWeights}")

    def make_node(self, rng, size, dtype, alpha, K):
        alpha = pt.as_tensor_variable(alpha)
        K = pt.as_tensor_variable(intX(K))

        if K.ndim > 0:
            raise ValueError("K must be a scalar.")

        return super().make_node(rng, size, dtype, alpha, K)

    def _supp_shape_from_params(self, dist_params, param_shapes):
        K = dist_params[1]
        return (K + 1,)

    @classmethod
    def rng_fn(cls, rng, alpha, K, size):
        if K < 0:
            raise ValueError("K needs to be positive.")

        size = to_tuple(size) if size is not None else alpha.shape
        size = size + (K,)
        alpha = alpha[..., np.newaxis]

        betas = rng.beta(1, alpha, size=size)

        sticks = np.concatenate(
            (
                np.ones(shape=(size[:-1] + (1,))),
                np.cumprod(1 - betas[..., :-1], axis=-1),
            ),
            axis=-1,
        )

        weights = sticks * betas
        weights = np.concatenate(
            (weights, 1 - weights.sum(axis=-1)[..., np.newaxis]),
            axis=-1,
        )

        return weights


stickbreakingweights = StickBreakingWeightsRV()


class StickBreakingWeights(SimplexContinuous):
    r"""
    Likelihood of truncated stick-breaking weights. The weights are generated from a
    stick-breaking proceduce where :math:`x_k = v_k \prod_{\ell < k} (1 - v_\ell)` for
    :math:`k \in \{1, \ldots, K\}` and :math:`x_K = \prod_{\ell = 1}^{K} (1 - v_\ell) = 1 - \sum_{\ell=1}^K x_\ell`
    with :math:`v_k \stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \alpha)`.

    .. math:

        f(\mathbf{x}|\alpha, K) =
            B(1, \alpha)^{-K}x_{K+1}^\alpha \prod_{k=1}^{K+1}\left\{\sum_{j=k}^{K+1} x_j\right\}^{-1}

    ========  ===============================================
    Support   :math:`x_k \in (0, 1)` for :math:`k \in \{1, \ldots, K+1\}`
              such that :math:`\sum x_k = 1`
    Mean      :math:`\mathbb{E}[x_k] = \dfrac{1}{1 + \alpha}\left(\dfrac{\alpha}{1 + \alpha}\right)^{k - 1}`
              for :math:`k \in \{1, \ldots, K\}` and :math:`\mathbb{E}[x_{K+1}] = \left(\dfrac{\alpha}{1 + \alpha}\right)^{K}`
    ========  ===============================================

    Parameters
    ----------
    alpha : tensor_like of float
        Concentration parameter (alpha > 0).
    K : tensor_like of int
        The number of "sticks" to break off from an initial one-unit stick. The length of the weight
        vector is K + 1, where the last weight is one minus the sum of all the first sticks.

    References
    ----------
    .. [1] Ishwaran, H., & James, L. F. (2001). Gibbs sampling methods for stick-breaking priors.
           Journal of the American Statistical Association, 96(453), 161-173.

    .. [2] Müller, P., Quintana, F. A., Jara, A., & Hanson, T. (2015). Bayesian nonparametric data
           analysis. New York: Springer.
    """
    rv_op = stickbreakingweights

    @classmethod
    def dist(cls, alpha, K, *args, **kwargs):
        alpha = pt.as_tensor_variable(floatX(alpha))
        K = pt.as_tensor_variable(intX(K))

        return super().dist([alpha, K], **kwargs)

    def moment(rv, size, alpha, K):
        alpha = alpha[..., np.newaxis]
        moment = (alpha / (1 + alpha)) ** pt.arange(K)
        moment *= 1 / (1 + alpha)
        moment = pt.concatenate([moment, (alpha / (1 + alpha)) ** K], axis=-1)
        if not rv_size_is_none(size):
            moment_size = pt.concatenate(
                [
                    size,
                    [
                        K + 1,
                    ],
                ]
            )
            moment = pt.full(moment_size, moment)

        return moment

    def logp(value, alpha, K):
        """
        Calculate log-probability of the distribution induced from the stick-breaking process
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        logp = -pt.sum(
            pt.log(
                pt.cumsum(
                    value[..., ::-1],
                    axis=-1,
                )
            ),
            axis=-1,
        )
        logp += -K * betaln(1, alpha)
        logp += alpha * pt.log(value[..., -1])

        logp = pt.switch(
            pt.or_(
                pt.any(
                    pt.and_(pt.le(value, 0), pt.ge(value, 1)),
                    axis=-1,
                ),
                pt.or_(
                    pt.bitwise_not(pt.allclose(value.sum(-1), 1)),
                    pt.neq(value.shape[-1], K + 1),
                ),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            alpha > 0,
            K > 0,
            msg="alpha > 0, K > 0",
        )


class ZeroSumNormalRV(SymbolicRandomVariable):
    """ZeroSumNormal random variable"""

    _print_name = ("ZeroSumNormal", "\\operatorname{ZeroSumNormal}")
    default_output = 0


class ZeroSumNormal(Distribution):
    r"""
    ZeroSumNormal distribution, i.e Normal distribution where one or
    several axes are constrained to sum to zero.
    By default, the last axis is constrained to sum to zero.
    See `n_zerosum_axes` kwarg for more details.

    .. math::

        \begin{align*}
            ZSN(\sigma) = N \Big( 0, \sigma^2 (I - \tfrac{1}{n}J) \Big) \\
            \text{where} \ ~ J_{ij} = 1 \ ~ \text{and} \\
            n = \text{nbr of zero-sum axes}
        \end{align*}

    Parameters
    ----------
    sigma : tensor_like of float
        Scale parameter (sigma > 0).
        It's actually the standard deviation of the underlying, unconstrained Normal distribution.
        Defaults to 1 if not specified.
        For now, ``sigma`` has to be a scalar, to ensure the zero-sum constraint.
    n_zerosum_axes: int, defaults to 1
        Number of axes along which the zero-sum constraint is enforced, starting from the rightmost position.
        Defaults to 1, i.e the rightmost axis.
    zerosum_axes: int, deprecated please use n_zerosum_axes as its successor
    dims: sequence of strings, optional
        Dimension names of the distribution. Works the same as for other PyMC distributions.
        Necessary if ``shape`` is not passed.
    shape: tuple of integers, optional
        Shape of the distribution. Works the same as for other PyMC distributions.
        Necessary if ``dims`` or ``observed`` is not passed.

    Warnings
    --------
    ``sigma`` has to be a scalar, to ensure the zero-sum constraint.
    The ability to specify a vector of ``sigma`` may be added in future versions.

    ``n_zerosum_axes`` has to be > 0. If you want the behavior of ``n_zerosum_axes = 0``,
    just use ``pm.Normal``.

    Examples
    --------
    Define a `ZeroSumNormal` variable, with `sigma=1` and
    `n_zerosum_axes=1`  by default::

        COORDS = {
            "regions": ["a", "b", "c"],
            "answers": ["yes", "no", "whatever", "don't understand question"],
        }
        with pm.Model(coords=COORDS) as m:
            # the zero sum axis will be 'answers'
            v = pm.ZeroSumNormal("v", dims=("regions", "answers"))

        with pm.Model(coords=COORDS) as m:
            # the zero sum axes will be 'answers' and 'regions'
            v = pm.ZeroSumNormal("v", dims=("regions", "answers"), n_zerosum_axes=2)

        with pm.Model(coords=COORDS) as m:
            # the zero sum axes will be the last two
            v = pm.ZeroSumNormal("v", shape=(3, 4, 5), n_zerosum_axes=2)
    """
    rv_type = ZeroSumNormalRV

    def __new__(
        cls, *args, zerosum_axes=None, n_zerosum_axes=None, support_shape=None, dims=None, **kwargs
    ):
        if zerosum_axes is not None:
            n_zerosum_axes = zerosum_axes
            warnings.warn(
                "The 'zerosum_axes' parameter is deprecated. Use 'n_zerosum_axes' instead.",
                DeprecationWarning,
            )
        if dims is not None or kwargs.get("observed") is not None:
            n_zerosum_axes = cls.check_zerosum_axes(n_zerosum_axes)

            support_shape = get_support_shape(
                support_shape=support_shape,
                shape=None,  # Shape will be checked in `cls.dist`
                dims=dims,
                observed=kwargs.get("observed", None),
                ndim_supp=n_zerosum_axes,
            )

        return super().__new__(
            cls,
            *args,
            n_zerosum_axes=n_zerosum_axes,
            support_shape=support_shape,
            dims=dims,
            **kwargs,
        )

    @classmethod
    def dist(cls, sigma=1, n_zerosum_axes=None, support_shape=None, **kwargs):
        n_zerosum_axes = cls.check_zerosum_axes(n_zerosum_axes)

        sigma = pt.as_tensor_variable(floatX(sigma))
        if sigma.ndim > 0:
            raise ValueError("sigma has to be a scalar")

        support_shape = get_support_shape(
            support_shape=support_shape,
            shape=kwargs.get("shape"),
            ndim_supp=n_zerosum_axes,
        )

        if support_shape is None:
            if n_zerosum_axes > 0:
                raise ValueError("You must specify dims, shape or support_shape parameter")
            # TODO: edge-case doesn't work for now, because pt.stack in get_support_shape fails
            # else:
            #     support_shape = () # because it's just a Normal in that case
        support_shape = pt.as_tensor_variable(intX(support_shape))

        assert n_zerosum_axes == pt.get_vector_length(
            support_shape
        ), "support_shape has to be as long as n_zerosum_axes"

        return super().dist(
            [sigma], n_zerosum_axes=n_zerosum_axes, support_shape=support_shape, **kwargs
        )

    @classmethod
    def check_zerosum_axes(cls, n_zerosum_axes: Optional[int]) -> int:
        if n_zerosum_axes is None:
            n_zerosum_axes = 1
        if not isinstance(n_zerosum_axes, int):
            raise TypeError("n_zerosum_axes has to be an integer")
        if not n_zerosum_axes > 0:
            raise ValueError("n_zerosum_axes has to be > 0")
        return n_zerosum_axes

    @classmethod
    def rv_op(cls, sigma, n_zerosum_axes, support_shape, size=None):
        shape = to_tuple(size) + tuple(support_shape)
        normal_dist = pm.Normal.dist(sigma=sigma, shape=shape)

        if n_zerosum_axes > normal_dist.ndim:
            raise ValueError("Shape of distribution is too small for the number of zerosum axes")

        normal_dist_, sigma_, support_shape_ = (
            normal_dist.type(),
            sigma.type(),
            support_shape.type(),
        )

        # Zerosum-normaling is achieved by subtracting the mean along the given n_zerosum_axes
        zerosum_rv_ = normal_dist_
        for axis in range(n_zerosum_axes):
            zerosum_rv_ -= zerosum_rv_.mean(axis=-axis - 1, keepdims=True)

        return ZeroSumNormalRV(
            inputs=[normal_dist_, sigma_, support_shape_],
            outputs=[zerosum_rv_, support_shape_],
            ndim_supp=n_zerosum_axes,
        )(normal_dist, sigma, support_shape)


@_change_dist_size.register(ZeroSumNormalRV)
def change_zerosum_size(op, normal_dist, new_size, expand=False):
    normal_dist, sigma, support_shape = normal_dist.owner.inputs

    if expand:
        original_shape = tuple(normal_dist.shape)
        old_size = original_shape[: len(original_shape) - op.ndim_supp]
        new_size = tuple(new_size) + old_size

    return ZeroSumNormal.rv_op(
        sigma=sigma, n_zerosum_axes=op.ndim_supp, support_shape=support_shape, size=new_size
    )


@_moment.register(ZeroSumNormalRV)
def zerosumnormal_moment(op, rv, *rv_inputs):
    return pt.zeros_like(rv)


@_default_transform.register(ZeroSumNormalRV)
def zerosum_default_transform(op, rv):
    n_zerosum_axes = tuple(np.arange(-op.ndim_supp, 0))
    return ZeroSumTransform(n_zerosum_axes)


@_logprob.register(ZeroSumNormalRV)
def zerosumnormal_logp(op, values, normal_dist, sigma, support_shape, **kwargs):
    (value,) = values
    shape = value.shape
    n_zerosum_axes = op.ndim_supp
    *_, sigma = normal_dist.owner.inputs

    _deg_free_support_shape = pt.inc_subtensor(shape[-n_zerosum_axes:], -1)
    _full_size = pm.floatX(pt.prod(shape))
    _degrees_of_freedom = pm.floatX(pt.prod(_deg_free_support_shape))

    zerosums = [
        pt.all(pt.isclose(pt.mean(value, axis=-axis - 1), 0, atol=1e-9))
        for axis in range(n_zerosum_axes)
    ]

    out = pt.sum(
        -0.5 * pt.pow(value / sigma, 2)
        - (pt.log(pt.sqrt(2.0 * np.pi)) + pt.log(sigma)) * _degrees_of_freedom / _full_size,
        axis=tuple(np.arange(-n_zerosum_axes, 0)),
    )

    return check_parameters(out, *zerosums, msg="mean(value, axis=n_zerosum_axes) = 0")
