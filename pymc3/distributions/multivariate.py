#   Copyright 2020 The PyMC Developers
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

import numpy as np
import scipy
import theano
import theano.tensor as tt

from scipy import linalg, stats
from theano.graph.basic import Apply
from theano.graph.op import Op, get_test_value
from theano.graph.utils import TestValueError
from theano.tensor.nlinalg import det, eigh, matrix_inverse, trace
from theano.tensor.slinalg import Cholesky

import pymc3 as pm

from pymc3.distributions import transforms
from pymc3.distributions.continuous import ChiSquared, Normal
from pymc3.distributions.dist_math import bound, factln, logpow
from pymc3.distributions.distribution import (
    Continuous,
    Discrete,
    _DrawValuesContext,
    draw_values,
    generate_samples,
)
from pymc3.distributions.shape_utils import broadcast_dist_samples_to, to_tuple
from pymc3.distributions.special import gammaln, multigammaln
from pymc3.exceptions import ShapeError
from pymc3.math import kron_diag, kron_dot, kron_solve_lower, kronecker
from pymc3.model import Deterministic
from pymc3.theanof import floatX, intX

__all__ = [
    "MvNormal",
    "MvStudentT",
    "Dirichlet",
    "Multinomial",
    "DirichletMultinomial",
    "Wishart",
    "WishartBartlett",
    "LKJCorr",
    "LKJCholeskyCov",
    "MatrixNormal",
    "KroneckerNormal",
]


class _QuadFormBase(Continuous):
    def __init__(self, mu=None, cov=None, chol=None, tau=None, lower=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.shape) > 2:
            raise ValueError("Only 1 or 2 dimensions are allowed.")

        if chol is not None and not lower:
            chol = chol.T
        if len([i for i in [tau, cov, chol] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. Specify exactly one of tau, cov, or chol."
            )
        self.mu = mu = tt.as_tensor_variable(mu)
        self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        # Step methods and advi do not catch LinAlgErrors at the
        # moment. We work around that by using a cholesky op
        # that returns a nan as first entry instead of raising
        # an error.
        cholesky = Cholesky(lower=True, on_error="nan")

        if cov is not None:
            self.k = cov.shape[0]
            self._cov_type = "cov"
            cov = tt.as_tensor_variable(cov)
            if cov.ndim != 2:
                raise ValueError("cov must be two dimensional.")
            self.chol_cov = cholesky(cov)
            self.cov = cov
            self._n = self.cov.shape[-1]
        elif tau is not None:
            self.k = tau.shape[0]
            self._cov_type = "tau"
            tau = tt.as_tensor_variable(tau)
            if tau.ndim != 2:
                raise ValueError("tau must be two dimensional.")
            self.chol_tau = cholesky(tau)
            self.tau = tau
            self._n = self.tau.shape[-1]
        else:
            self.k = chol.shape[0]
            self._cov_type = "chol"
            if chol.ndim != 2:
                raise ValueError("chol must be two dimensional.")
            self.chol_cov = tt.as_tensor_variable(chol)
            self._n = self.chol_cov.shape[-1]

    def _quaddist(self, value):
        """Compute (x - mu).T @ Sigma^-1 @ (x - mu) and the logdet of Sigma."""
        mu = self.mu
        if value.ndim > 2 or value.ndim == 0:
            raise ValueError("Invalid dimension for value: %s" % value.ndim)
        if value.ndim == 1:
            onedim = True
            value = value[None, :]
        else:
            onedim = False

        delta = value - mu

        if self._cov_type == "cov":
            # Use this when Theano#5908 is released.
            # return MvNormalLogp()(self.cov, delta)
            dist, logdet, ok = self._quaddist_cov(delta)
        elif self._cov_type == "tau":
            dist, logdet, ok = self._quaddist_tau(delta)
        else:
            dist, logdet, ok = self._quaddist_chol(delta)

        if onedim:
            return dist[0], logdet, ok
        return dist, logdet, ok

    def _quaddist_chol(self, delta):
        chol_cov = self.chol_cov
        diag = tt.nlinalg.diag(chol_cov)
        # Check if the covariance matrix is positive definite.
        ok = tt.all(diag > 0)
        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        chol_cov = tt.switch(ok, chol_cov, 1)

        delta_trans = self.solve_lower(chol_cov, delta.T).T
        quaddist = (delta_trans ** 2).sum(axis=-1)
        logdet = tt.sum(tt.log(diag))
        return quaddist, logdet, ok

    def _quaddist_cov(self, delta):
        return self._quaddist_chol(delta)

    def _quaddist_tau(self, delta):
        chol_tau = self.chol_tau
        diag = tt.nlinalg.diag(chol_tau)
        # Check if the precision matrix is positive definite.
        ok = tt.all(diag > 0)
        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        chol_tau = tt.switch(ok, chol_tau, 1)

        delta_trans = tt.dot(delta, chol_tau)
        quaddist = (delta_trans ** 2).sum(axis=-1)
        logdet = -tt.sum(tt.log(diag))
        return quaddist, logdet, ok

    def _cov_param_for_repr(self):
        if self._cov_type == "chol":
            return "chol_cov"
        else:
            return self._cov_type


class MvNormal(_QuadFormBase):
    R"""
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
    mu: array
        Vector of means.
    cov: array
        Covariance matrix. Exactly one of cov, tau, or chol is needed.
    tau: array
        Precision matrix. Exactly one of cov, tau, or chol is needed.
    chol: array
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
        vals = pm.Deterministic('vals', tt.dot(chol, vals_raw.T).T)
    """

    def __init__(self, mu, cov=None, tau=None, chol=None, lower=True, *args, **kwargs):
        super().__init__(mu=mu, cov=cov, tau=tau, chol=chol, lower=lower, *args, **kwargs)
        self.mean = self.median = self.mode = self.mu = self.mu

    def random(self, point=None, size=None):
        """
        Draw random values from Multivariate Normal distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        size = to_tuple(size)

        param_attribute = getattr(self, "chol_cov" if self._cov_type == "chol" else self._cov_type)
        mu, param = draw_values([self.mu, param_attribute], point=point, size=size)

        dist_shape = to_tuple(self.shape)
        output_shape = size + dist_shape

        # Simple, there can be only be 1 batch dimension, only available from `mu`.
        # Insert it into `param` before events, if there is a sample shape in front.
        if param.ndim > 2 and dist_shape[:-1]:
            param = param.reshape(size + (1,) + param.shape[-2:])

        mu = broadcast_dist_samples_to(to_shape=output_shape, samples=[mu], size=size)[0]
        param = np.broadcast_to(param, shape=output_shape + dist_shape[-1:])

        assert mu.shape == output_shape
        assert param.shape == output_shape + dist_shape[-1:]

        if self._cov_type == "cov":
            chol = np.linalg.cholesky(param)
        elif self._cov_type == "chol":
            chol = param
        else:  # tau -> chol -> swapaxes (chol, -1, -2) -> inv ...
            lower_chol = np.linalg.cholesky(param)
            upper_chol = np.swapaxes(lower_chol, -1, -2)
            chol = np.linalg.inv(upper_chol)

        standard_normal = np.random.standard_normal(output_shape)
        return mu + np.einsum("...ij,...j->...i", chol, standard_normal)

    def logp(self, value):
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
        quaddist, logdet, ok = self._quaddist(value)
        k = floatX(value.shape[-1])
        norm = -0.5 * k * pm.floatX(np.log(2 * np.pi))
        return bound(norm - 0.5 * quaddist - logdet, ok)

    def _distr_parameters_for_repr(self):
        return ["mu", self._cov_param_for_repr()]


class MvStudentT(_QuadFormBase):
    R"""
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
    nu: int
        Degrees of freedom.
    Sigma: matrix
        Covariance matrix. Use `cov` in new code.
    mu: array
        Vector of means.
    cov: matrix
        The covariance matrix.
    tau: matrix
        The precision matrix.
    chol: matrix
        The cholesky factor of the covariance matrix.
    lower: bool, default=True
        Whether the cholesky fatcor is given as a lower triangular matrix.
    """

    def __init__(
        self, nu, Sigma=None, mu=None, cov=None, tau=None, chol=None, lower=True, *args, **kwargs
    ):
        if Sigma is not None:
            if cov is not None:
                raise ValueError("Specify only one of cov and Sigma")
            cov = Sigma
        super().__init__(mu=mu, cov=cov, tau=tau, chol=chol, lower=lower, *args, **kwargs)
        self.nu = nu = tt.as_tensor_variable(nu)
        self.mean = self.median = self.mode = self.mu = self.mu

    def random(self, point=None, size=None):
        """
        Draw random values from Multivariate Student's T distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        with _DrawValuesContext():
            nu, mu = draw_values([self.nu, self.mu], point=point, size=size)
            if self._cov_type == "cov":
                (cov,) = draw_values([self.cov], point=point, size=size)
                dist = MvNormal.dist(mu=np.zeros_like(mu), cov=cov, shape=self.shape)
            elif self._cov_type == "tau":
                (tau,) = draw_values([self.tau], point=point, size=size)
                dist = MvNormal.dist(mu=np.zeros_like(mu), tau=tau, shape=self.shape)
            else:
                (chol,) = draw_values([self.chol_cov], point=point, size=size)
                dist = MvNormal.dist(mu=np.zeros_like(mu), chol=chol, shape=self.shape)

            samples = dist.random(point, size)

        chi2_samples = np.random.chisquare(nu, size)
        # Add distribution shape to chi2 samples
        chi2_samples = chi2_samples.reshape(chi2_samples.shape + (1,) * len(self.shape))
        return (samples / np.sqrt(chi2_samples / nu)) + mu

    def logp(self, value):
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
        quaddist, logdet, ok = self._quaddist(value)
        k = floatX(value.shape[-1])

        norm = (
            gammaln((self.nu + k) / 2.0)
            - gammaln(self.nu / 2.0)
            - 0.5 * k * floatX(np.log(self.nu * np.pi))
        )
        inner = -(self.nu + k) / 2.0 * tt.log1p(quaddist / self.nu)
        return bound(norm + inner - logdet, ok)

    def _distr_parameters_for_repr(self):
        return ["mu", "nu", self._cov_param_for_repr()]


class Dirichlet(Continuous):
    R"""
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
    a: array
        Concentration parameters (a > 0).
    """

    def __init__(self, a, transform=transforms.stick_breaking, *args, **kwargs):

        if kwargs.get("shape") is None:
            warnings.warn(
                (
                    "Shape not explicitly set. "
                    "Please, set the value using the `shape` keyword argument. "
                    "Using the test value to infer the shape."
                ),
                DeprecationWarning,
            )
            try:
                kwargs["shape"] = np.shape(get_test_value(a))
            except TestValueError:
                pass

        super().__init__(transform=transform, *args, **kwargs)

        self.a = a = tt.as_tensor_variable(a)
        self.mean = a / tt.sum(a)

        self.mode = tt.switch(tt.all(a > 1), (a - 1) / tt.sum(a - 1), np.nan)

    def random(self, point=None, size=None):
        """
        Draw random values from Dirichlet distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        a = draw_values([self.a], point=point, size=size)[0]
        output_shape = to_tuple(size) + to_tuple(self.shape)
        a = broadcast_dist_samples_to(to_shape=output_shape, samples=[a], size=size)[0]
        samples = stats.gamma.rvs(a=a, size=output_shape)
        samples = samples / samples.sum(-1, keepdims=True)
        return samples

    def logp(self, value):
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
        a = self.a

        # only defined for sum(value) == 1
        return bound(
            tt.sum(logpow(value, a - 1) - gammaln(a), axis=-1) + gammaln(tt.sum(a, axis=-1)),
            tt.all(value >= 0),
            tt.all(value <= 1),
            tt.all(a > 0),
            broadcast_conditions=False,
        )

    def _distr_parameters_for_repr(self):
        return ["a"]


class Multinomial(Discrete):
    R"""
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
    n: int or array
        Number of trials (n > 0). If n is an array its shape must be (N,) with
        N = p.shape[0]
    p: one- or two-dimensional array
        Probability of each one of the different outcomes. Elements must
        be non-negative and sum to 1 along the last axis. They will be
        automatically rescaled otherwise.
    """

    def __init__(self, n, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        p = p / tt.sum(p, axis=-1, keepdims=True)

        if len(self.shape) > 1:
            self.n = tt.shape_padright(n)
            self.p = p if p.ndim > 1 else tt.shape_padleft(p)
        else:
            # n is a scalar, p is a 1d array
            self.n = tt.as_tensor_variable(n)
            self.p = tt.as_tensor_variable(p)

        self.mean = self.n * self.p
        mode = tt.cast(tt.round(self.mean), "int32")
        diff = self.n - tt.sum(mode, axis=-1, keepdims=True)
        inc_bool_arr = tt.abs_(diff) > 0
        mode = tt.inc_subtensor(mode[inc_bool_arr.nonzero()], diff[inc_bool_arr.nonzero()])
        self.mode = mode

    def _random(self, n, p, size=None, raw_size=None):
        original_dtype = p.dtype
        # Set float type to float64 for numpy. This change is related to numpy issue #8317 (https://github.com/numpy/numpy/issues/8317)
        p = p.astype("float64")
        # Now, re-normalize all of the values in float64 precision. This is done inside the conditionals
        p /= np.sum(p, axis=-1, keepdims=True)

        # Thanks to the default shape handling done in generate_values, the last
        # axis of n is a dummy axis that allows it to broadcast well with p
        n = np.broadcast_to(n, size)
        p = np.broadcast_to(p, size)
        n = n[..., 0]

        # np.random.multinomial needs `n` to be a scalar int and `p` a
        # sequence so we semi flatten them and iterate over them
        size_ = to_tuple(raw_size)
        if p.ndim > len(size_) and p.shape[: len(size_)] == size_:
            # p and n have the size_ prepend so we don't need it in np.random
            n_ = n.reshape([-1])
            p_ = p.reshape([-1, p.shape[-1]])
            samples = np.array([np.random.multinomial(nn, pp) for nn, pp in zip(n_, p_)])
            samples = samples.reshape(p.shape)
        else:
            # p and n don't have the size prepend
            n_ = n.reshape([-1])
            p_ = p.reshape([-1, p.shape[-1]])
            samples = np.array(
                [np.random.multinomial(nn, pp, size=size_) for nn, pp in zip(n_, p_)]
            )
            samples = np.moveaxis(samples, 0, -1)
            samples = samples.reshape(size + p.shape)
        # We cast back to the original dtype
        return samples.astype(original_dtype)

    def random(self, point=None, size=None):
        """
        Draw random values from Multinomial distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        n, p = draw_values([self.n, self.p], point=point, size=size)
        samples = generate_samples(
            self._random,
            n,
            p,
            dist_shape=self.shape,
            not_broadcast_kwargs={"raw_size": size},
            size=size,
        )
        return samples

    def logp(self, x):
        """
        Calculate log-probability of Multinomial distribution
        at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        n = self.n
        p = self.p

        return bound(
            factln(n) + tt.sum(-factln(x) + logpow(p, x), axis=-1, keepdims=True),
            tt.all(x >= 0),
            tt.all(tt.eq(tt.sum(x, axis=-1, keepdims=True), n)),
            tt.all(p <= 1),
            tt.all(tt.eq(tt.sum(p, axis=-1), 1)),
            tt.all(tt.ge(n, 0)),
            broadcast_conditions=False,
        )


class DirichletMultinomial(Discrete):
    R"""Dirichlet Multinomial log-likelihood.

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
    n : int or array
        Total counts in each replicate. If n is an array its shape must be (N,)
        with N = a.shape[0]

    a : one- or two-dimensional array
        Dirichlet parameter. Elements must be strictly positive.
        The number of categories is given by the length of the last axis.

    shape : integer tuple
        Describes shape of distribution. For example if n=array([5, 10]), and
        a=array([1, 1, 1]), shape should be (2, 3).
    """

    def __init__(self, n, a, shape, *args, **kwargs):

        super().__init__(shape=shape, defaults=("_defaultval",), *args, **kwargs)

        n = intX(n)
        a = floatX(a)
        if len(self.shape) > 1:
            self.n = tt.shape_padright(n)
            self.a = tt.as_tensor_variable(a) if a.ndim > 1 else tt.shape_padleft(a)
        else:
            # n is a scalar, p is a 1d array
            self.n = tt.as_tensor_variable(n)
            self.a = tt.as_tensor_variable(a)

        p = self.a / self.a.sum(-1, keepdims=True)

        self.mean = self.n * p
        # Mode is only an approximation. Exact computation requires a complex
        # iterative algorithm as described in https://doi.org/10.1016/j.spl.2009.09.013
        mode = tt.cast(tt.round(self.mean), "int32")
        diff = self.n - tt.sum(mode, axis=-1, keepdims=True)
        inc_bool_arr = tt.abs_(diff) > 0
        mode = tt.inc_subtensor(mode[inc_bool_arr.nonzero()], diff[inc_bool_arr.nonzero()])
        self._defaultval = mode

    def _random(self, n, a, size=None):
        # numpy will cast dirichlet and multinomial samples to float64 by default
        original_dtype = a.dtype

        # Thanks to the default shape handling done in generate_values, the last
        # axis of n is a dummy axis that allows it to broadcast well with `a`
        n = np.broadcast_to(n, size)
        a = np.broadcast_to(a, size)
        n = n[..., 0]

        # np.random.multinomial needs `n` to be a scalar int and `a` a
        # sequence so we semi flatten them and iterate over them
        n_ = n.reshape([-1])
        a_ = a.reshape([-1, a.shape[-1]])
        p_ = np.array([np.random.dirichlet(aa) for aa in a_])
        samples = np.array([np.random.multinomial(nn, pp) for nn, pp in zip(n_, p_)])
        samples = samples.reshape(a.shape)

        # We cast back to the original dtype
        return samples.astype(original_dtype)

    def random(self, point=None, size=None):
        """
        Draw random values from Dirichlet-Multinomial distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        n, a = draw_values([self.n, self.a], point=point, size=size)
        samples = generate_samples(
            self._random,
            n,
            a,
            dist_shape=self.shape,
            size=size,
        )

        # If distribution is initialized with .dist(), valid init shape is not asserted.
        # Under normal use in a model context valid init shape is asserted at start.
        expected_shape = to_tuple(size) + to_tuple(self.shape)
        sample_shape = tuple(samples.shape)
        if sample_shape != expected_shape:
            raise ShapeError(
                f"Expected sample shape was {expected_shape} but got {sample_shape}. "
                "This may reflect an invalid initialization shape."
            )

        return samples

    def logp(self, value):
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
        a = self.a
        n = self.n
        sum_a = a.sum(axis=-1, keepdims=True)

        const = (gammaln(n + 1) + gammaln(sum_a)) - gammaln(n + sum_a)
        series = gammaln(value + a) - (gammaln(value + 1) + gammaln(a))
        result = const + series.sum(axis=-1, keepdims=True)
        # Bounds checking to confirm parameters and data meet all constraints
        # and that each observation value_i sums to n_i.
        return bound(
            result,
            tt.all(tt.ge(value, 0)),
            tt.all(tt.gt(a, 0)),
            tt.all(tt.ge(n, 0)),
            tt.all(tt.eq(value.sum(axis=-1, keepdims=True), n)),
            broadcast_conditions=False,
        )

    def _distr_parameters_for_repr(self):
        return ["n", "a"]


def posdef(AA):
    try:
        linalg.cholesky(AA)
        return 1
    except linalg.LinAlgError:
        return 0


class PosDefMatrix(Op):
    """
    Check if input is positive definite. Input should be a square matrix.

    """

    # Properties attribute
    __props__ = ()

    # Compulsory if itypes and otypes are not defined

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        o = tt.TensorType(dtype="int8", broadcastable=[])()
        return Apply(self, [x], [o])

    # Python implementation:
    def perform(self, node, inputs, outputs):

        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.array(posdef(x), dtype="int8")
        except Exception:
            pm._log.exception("Failed to check if %s positive definite", x)
            raise

    def infer_shape(self, fgraph, node, shapes):
        return [[]]

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(theano.config.floatX)]

    def __str__(self):
        return "MatrixIsPositiveDefinite"


matrix_pos_def = PosDefMatrix()


class Wishart(Continuous):
    R"""
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
    nu: int
        Degrees of freedom, > 0.
    V: array
        p x p positive definite matrix.

    Notes
    -----
    This distribution is unusable in a PyMC3 model. You should instead
    use LKJCholeskyCov or LKJCorr.
    """

    def __init__(self, nu, V, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "The Wishart distribution can currently not be used "
            "for MCMC sampling. The probability of sampling a "
            "symmetric matrix is basically zero. Instead, please "
            "use LKJCholeskyCov or LKJCorr. For more information "
            "on the issues surrounding the Wishart see here: "
            "https://github.com/pymc-devs/pymc3/issues/538.",
            UserWarning,
        )
        self.nu = nu = tt.as_tensor_variable(nu)
        self.p = p = tt.as_tensor_variable(V.shape[0])
        self.V = V = tt.as_tensor_variable(V)
        self.mean = nu * V
        self.mode = tt.switch(tt.ge(nu, p + 1), (nu - p - 1) * V, np.nan)

    def random(self, point=None, size=None):
        """
        Draw random values from Wishart distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        nu, V = draw_values([self.nu, self.V], point=point, size=size)
        size = 1 if size is None else size
        return generate_samples(stats.wishart.rvs, nu.item(), V, broadcast_shape=(size,))

    def logp(self, X):
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
        nu = self.nu
        p = self.p
        V = self.V

        IVI = det(V)
        IXI = det(X)

        return bound(
            (
                (nu - p - 1) * tt.log(IXI)
                - trace(matrix_inverse(V).dot(X))
                - nu * p * tt.log(2)
                - nu * tt.log(IVI)
                - 2 * multigammaln(nu / 2.0, p)
            )
            / 2,
            matrix_pos_def(X),
            tt.eq(X, X.T),
            nu > (p - 1),
            broadcast_conditions=False,
        )


def WishartBartlett(name, S, nu, is_cholesky=False, return_cholesky=False, testval=None):
    R"""
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
    S: ndarray
        p x p positive definite matrix
        Or:
        p x p lower-triangular matrix that is the Cholesky factor
        of the covariance matrix.
    nu: int
        Degrees of freedom, > dim(S).
    is_cholesky: bool (default=False)
        Input matrix S is already Cholesky decomposed as S.T * S
    return_cholesky: bool (default=False)
        Only return the Cholesky decomposed matrix.
    testval: ndarray
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

    if testval is not None:
        # Inverse transform
        testval = np.dot(np.dot(np.linalg.inv(L), testval), np.linalg.inv(L.T))
        testval = linalg.cholesky(testval, lower=True)
        diag_testval = testval[diag_idx] ** 2
        tril_testval = testval[tril_idx]
    else:
        diag_testval = None
        tril_testval = None

    c = tt.sqrt(
        ChiSquared("%s_c" % name, nu - np.arange(2, 2 + n_diag), shape=n_diag, testval=diag_testval)
    )
    pm._log.info("Added new variable %s_c to model diagonal of Wishart." % name)
    z = Normal("%s_z" % name, 0.0, 1.0, shape=n_tril, testval=tril_testval)
    pm._log.info("Added new variable %s_z to model off-diagonals of Wishart." % name)
    # Construct A matrix
    A = tt.zeros(S.shape, dtype=np.float32)
    A = tt.set_subtensor(A[diag_idx], c)
    A = tt.set_subtensor(A[tril_idx], z)

    # L * A * A.T * L.T ~ Wishart(L*L.T, nu)
    if return_cholesky:
        return Deterministic(name, tt.dot(L, A))
    else:
        return Deterministic(name, tt.dot(tt.dot(tt.dot(L, A), A.T), L.T))


def _lkj_normalizing_constant(eta, n):
    if eta == 1:
        result = gammaln(2.0 * tt.arange(1, int((n - 1) / 2) + 1)).sum()
        if n % 2 == 1:
            result += (
                0.25 * (n ** 2 - 1) * tt.log(np.pi)
                - 0.25 * (n - 1) ** 2 * tt.log(2.0)
                - (n - 1) * gammaln(int((n + 1) / 2))
            )
        else:
            result += (
                0.25 * n * (n - 2) * tt.log(np.pi)
                + 0.25 * (3 * n ** 2 - 4 * n) * tt.log(2.0)
                + n * gammaln(n / 2)
                - (n - 1) * gammaln(n)
            )
    else:
        result = -(n - 1) * gammaln(eta + 0.5 * (n - 1))
        k = tt.arange(1, n)
        result += (0.5 * k * tt.log(np.pi) + gammaln(eta + 0.5 * (n - 1 - k))).sum()
    return result


class _LKJCholeskyCov(Continuous):
    R"""Underlying class for covariance matrix with LKJ distributed correlations.
    See docs for LKJCholeskyCov function for more details on how to use it in models.
    """

    def __init__(self, eta, n, sd_dist, *args, **kwargs):
        self.n = tt.as_tensor_variable(n)
        self.eta = tt.as_tensor_variable(eta)

        if "transform" in kwargs and kwargs["transform"] is not None:
            raise ValueError("Invalid parameter: transform.")
        if "shape" in kwargs:
            raise ValueError("Invalid parameter: shape.")

        shape = n * (n + 1) // 2

        if sd_dist.shape.ndim not in [0, 1]:
            raise ValueError("Invalid shape for sd_dist.")

        transform = transforms.CholeskyCovPacked(n)

        kwargs["shape"] = shape
        kwargs["transform"] = transform
        super().__init__(*args, **kwargs)

        self.sd_dist = sd_dist
        self.diag_idxs = transform.diag_idxs

        self.mode = floatX(np.zeros(shape))
        self.mode[self.diag_idxs] = 1

    def logp(self, x):
        """
        Calculate log-probability of Covariance matrix with LKJ
        distributed correlations at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        n = self.n
        eta = self.eta

        diag_idxs = self.diag_idxs
        cumsum = tt.cumsum(x ** 2)
        variance = tt.zeros(n)
        variance = tt.inc_subtensor(variance[0], x[0] ** 2)
        variance = tt.inc_subtensor(variance[1:], cumsum[diag_idxs[1:]] - cumsum[diag_idxs[:-1]])
        sd_vals = tt.sqrt(variance)

        logp_sd = self.sd_dist.logp(sd_vals).sum()
        corr_diag = x[diag_idxs] / sd_vals

        logp_lkj = (2 * eta - 3 + n - tt.arange(n)) * tt.log(corr_diag)
        logp_lkj = tt.sum(logp_lkj)

        # Compute the log det jacobian of the second transformation
        # described in the docstring.
        idx = tt.arange(n)
        det_invjac = tt.log(corr_diag) - idx * tt.log(sd_vals)
        det_invjac = det_invjac.sum()

        norm = _lkj_normalizing_constant(eta, n)

        return norm + logp_lkj + logp_sd + det_invjac

    def _random(self, n, eta, size=1):
        eta_sample_shape = (size,) + eta.shape
        P = np.eye(n) * np.ones(eta_sample_shape + (n, n))
        # original implementation in R see:
        # https://github.com/rmcelreath/rethinking/blob/master/R/distributions.r
        beta = eta - 1.0 + n / 2.0
        r12 = 2.0 * stats.beta.rvs(a=beta, b=beta, size=eta_sample_shape) - 1.0
        P[..., 0, 1] = r12
        P[..., 1, 1] = np.sqrt(1.0 - r12 ** 2)
        for mp1 in range(2, n):
            beta -= 0.5
            y = stats.beta.rvs(a=mp1 / 2.0, b=beta, size=eta_sample_shape)
            z = stats.norm.rvs(loc=0, scale=1, size=eta_sample_shape + (mp1,))
            z = z / np.sqrt(np.einsum("ij,ij->j", z, z))
            P[..., 0:mp1, mp1] = np.sqrt(y[..., np.newaxis]) * z
            P[..., mp1, mp1] = np.sqrt(1.0 - y)
        C = np.einsum("...ji,...jk->...ik", P, P)
        D = np.atleast_1d(self.sd_dist.random(size=P.shape[:-2]))
        if D.shape in [tuple(), (1,)]:
            D = self.sd_dist.random(size=P.shape[:-1])
        elif D.ndim < C.ndim - 1:
            D = [D] + [self.sd_dist.random(size=P.shape[:-2]) for _ in range(n - 1)]
            D = np.moveaxis(np.array(D), 0, C.ndim - 2)
        elif D.ndim == C.ndim - 1:
            if D.shape[-1] == 1:
                D = [D] + [self.sd_dist.random(size=P.shape[:-2]) for _ in range(n - 1)]
                D = np.concatenate(D, axis=-1)
            elif D.shape[-1] != n:
                raise ValueError(
                    "The size of the samples drawn from the "
                    "supplied sd_dist.random have the wrong "
                    "size. Expected {} but got {} instead.".format(n, D.shape[-1])
                )
        else:
            raise ValueError(
                "Supplied sd_dist.random generates samples with "
                "too many dimensions. It must yield samples "
                "with 0 or 1 dimensions. Got {} instead".format(D.ndim - C.ndim - 2)
            )
        C *= D[..., :, np.newaxis] * D[..., np.newaxis, :]
        tril_idx = np.tril_indices(n, k=0)
        return np.linalg.cholesky(C)[..., tril_idx[0], tril_idx[1]]

    def random(self, point=None, size=None):
        """
        Draw random values from Covariance matrix with LKJ
        distributed correlations.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        # Get parameters and broadcast them
        n, eta = draw_values([self.n, self.eta], point=point, size=size)
        broadcast_shape = np.broadcast(n, eta).shape
        # We can only handle cov matrices with a constant n per random call
        n = np.unique(n)
        if len(n) > 1:
            raise RuntimeError("Varying n is not supported for LKJCholeskyCov")
        n = int(n[0])
        dist_shape = ((n * (n + 1)) // 2,)
        # We make sure that eta and the drawn n get their shapes broadcasted
        eta = np.broadcast_to(eta, broadcast_shape)
        # We change the size of the draw depending on the broadcast shape
        sample_shape = broadcast_shape + dist_shape
        if size is not None:
            if not isinstance(size, tuple):
                try:
                    size = tuple(size)
                except TypeError:
                    size = (size,)
            if size == sample_shape:
                size = None
            elif size == broadcast_shape:
                size = None
            elif size[-len(sample_shape) :] == sample_shape:
                size = size[: len(size) - len(sample_shape)]
            elif size[-len(broadcast_shape) :] == broadcast_shape:
                size = size[: len(size) - len(broadcast_shape)]
        # We will always provide _random with an integer size and then reshape
        # the output to get the correct size
        if size is not None:
            _size = np.prod(size)
        else:
            _size = 1
        samples = self._random(n, eta, size=_size)
        if size is None:
            samples = samples[0]
        else:
            samples = np.reshape(samples, size + sample_shape)
        return samples

    def _distr_parameters_for_repr(self):
        return ["eta", "n"]


def LKJCholeskyCov(name, eta, n, sd_dist, compute_corr=False, store_in_trace=True, *args, **kwargs):
    R"""Wrapper function for covariance matrix with LKJ distributed correlations.

    This defines a distribution over Cholesky decomposed covariance
    matrices, such that the underlying correlation matrices follow an
    LKJ distribution [1] and the standard deviations follow an arbitray
    distribution specified by the user.

    Parameters
    ----------
    name: str
        The name given to the variable in the model.
    eta: float
        The shape parameter (eta > 0) of the LKJ distribution. eta = 1
        implies a uniform distribution of the correlation matrices;
        larger values put more weight on matrices with few correlations.
    n: int
        Dimension of the covariance matrix (n > 1).
    sd_dist: pm.Distribution
        A distribution for the standard deviations.
    compute_corr: bool, default=False
        If `True`, returns three values: the Cholesky decomposition, the correlations
        and the standard deviations of the covariance matrix. Otherwise, only returns
        the packed Cholesky decomposition. Defaults to `False` to ensure backwards
        compatibility.
    store_in_trace: bool, default=True
        Whether to store the correlations and standard deviations of the covariance
        matrix in the posterior trace. If `True`, they will automatically be named as
        `{name}_corr` and `{name}_stds` respectively. Effective only when
        `compute_corr=True`.

    Returns
    -------
    packed_chol: TensorVariable
        If `compute_corr=False` (default). The packed Cholesky covariance decomposition.
    chol:  TensorVariable
        If `compute_corr=True`. The unpacked Cholesky covariance decomposition.
    corr: TensorVariable
        If `compute_corr=True`. The correlations of the covariance matrix.
    stds: TensorVariable
        If `compute_corr=True`. The standard deviations of the covariance matrix.

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
            sd_dist = pm.Exponential.dist(1.0)
            chol, corr, sigmas = pm.LKJCholeskyCov('chol_cov', eta=4, n=10,
            sd_dist=sd_dist, compute_corr=True)

            # if you only want the packed Cholesky (default behavior):
            # packed_chol = pm.LKJCholeskyCov('chol_cov', eta=4, n=10, sd_dist=sd_dist)
            # chol = pm.expand_packed_triangular(10, packed_chol, lower=True)

            # Define a new MvNormal with the given covariance
            vals = pm.MvNormal('vals', mu=np.zeros(10), chol=chol, shape=10)

            # Or transform an uncorrelated normal:
            vals_raw = pm.Normal('vals_raw', mu=0, sigma=1, shape=10)
            vals = tt.dot(chol, vals_raw)

            # Or compute the covariance matrix
            cov = tt.dot(chol, chol.T)

    **Implementation** In the unconstrained space all values of the cholesky factor
    are stored untransformed, except for the diagonal entries, where
    we use a log-transform to restrict them to positive values.

    To correctly compute log-likelihoods for the standard deviations
    and the correlation matrix seperatly, we need to consider a
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
    # compute Cholesky decomposition
    packed_chol = _LKJCholeskyCov(name, eta=eta, n=n, sd_dist=sd_dist)
    if not compute_corr:
        return packed_chol

    else:
        chol = pm.expand_packed_triangular(n, packed_chol, lower=True)
        # compute covariance matrix
        cov = tt.dot(chol, chol.T)
        # extract standard deviations and rho
        stds = tt.sqrt(tt.diag(cov))
        inv_stds = 1 / stds
        corr = inv_stds[None, :] * cov * inv_stds[:, None]
        if store_in_trace:
            stds = pm.Deterministic(f"{name}_stds", stds)
            corr = pm.Deterministic(f"{name}_corr", corr)

        return chol, corr, stds


class LKJCorr(Continuous):
    R"""
    The LKJ (Lewandowski, Kurowicka and Joe) log-likelihood.

    The LKJ distribution is a prior distribution for correlation matrices.
    If eta = 1 this corresponds to the uniform distribution over correlation
    matrices. For eta -> oo the LKJ prior approaches the identity matrix.

    ========  ==============================================
    Support   Upper triangular matrix with values in [-1, 1]
    ========  ==============================================

    Parameters
    ----------
    n: int
        Dimension of the covariance matrix (n > 1).
    eta: float
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

    def __init__(self, eta=None, n=None, p=None, transform="interval", *args, **kwargs):
        if (p is not None) and (n is not None) and (eta is None):
            warnings.warn(
                "Parameters to LKJCorr have changed: shape parameter n -> eta "
                "dimension parameter p -> n. Please update your code. "
                "Automatically re-assigning parameters for backwards compatibility.",
                DeprecationWarning,
            )
            self.n = p
            self.eta = n
            eta = self.eta
            n = self.n
        elif (n is not None) and (eta is not None) and (p is None):
            self.n = n
            self.eta = eta
        else:
            raise ValueError(
                "Invalid parameter: please use eta as the shape parameter and "
                "n as the dimension parameter."
            )

        shape = n * (n - 1) // 2
        self.mean = floatX(np.zeros(shape))

        if transform == "interval":
            transform = transforms.interval(-1, 1)

        super().__init__(shape=shape, transform=transform, *args, **kwargs)
        warnings.warn(
            "Parameters in LKJCorr have been rename: shape parameter n -> eta "
            "dimension parameter p -> n. Please double check your initialization.",
            DeprecationWarning,
        )
        self.tri_index = np.zeros([n, n], dtype="int32")
        self.tri_index[np.triu_indices(n, k=1)] = np.arange(shape)
        self.tri_index[np.triu_indices(n, k=1)[::-1]] = np.arange(shape)

    def _random(self, n, eta, size=None):
        size = size if isinstance(size, tuple) else (size,)
        # original implementation in R see:
        # https://github.com/rmcelreath/rethinking/blob/master/R/distributions.r
        beta = eta - 1.0 + n / 2.0
        r12 = 2.0 * stats.beta.rvs(a=beta, b=beta, size=size) - 1.0
        P = np.eye(n)[:, :, np.newaxis] * np.ones(size)
        P[0, 1] = r12
        P[1, 1] = np.sqrt(1.0 - r12 ** 2)
        for mp1 in range(2, n):
            beta -= 0.5
            y = stats.beta.rvs(a=mp1 / 2.0, b=beta, size=size)
            z = stats.norm.rvs(loc=0, scale=1, size=(mp1,) + size)
            z = z / np.sqrt(np.einsum("ij,ij->j", z, z))
            P[0:mp1, mp1] = np.sqrt(y) * z
            P[mp1, mp1] = np.sqrt(1.0 - y)
        C = np.einsum("ji...,jk...->...ik", P, P)
        triu_idx = np.triu_indices(n, k=1)
        return C[..., triu_idx[0], triu_idx[1]]

    def random(self, point=None, size=None):
        """
        Draw random values from LKJ distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        n, eta = draw_values([self.n, self.eta], point=point, size=size)
        size = 1 if size is None else size
        samples = generate_samples(self._random, n, eta, broadcast_shape=to_tuple(size))
        return samples

    def logp(self, x):
        """
        Calculate log-probability of LKJ distribution at specified
        value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        n = self.n
        eta = self.eta

        X = x[self.tri_index]
        X = tt.fill_diagonal(X, 1)

        result = _lkj_normalizing_constant(eta, n)
        result += (eta - 1.0) * tt.log(det(X))
        return bound(
            result,
            tt.all(X <= 1),
            tt.all(X >= -1),
            matrix_pos_def(X),
            eta > 0,
            broadcast_conditions=False,
        )

    def _distr_parameters_for_repr(self):
        return ["eta", "n"]


class MatrixNormal(Continuous):
    R"""
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
    mu: array
        Array of means. Must be broadcastable with the random variable X such
        that the shape of mu + X is (m,n).
    rowcov: mxm array
        Among-row covariance matrix. Defines variance within
        columns. Exactly one of rowcov or rowchol is needed.
    rowchol: mxm array
        Cholesky decomposition of among-row covariance matrix. Exactly one of
        rowcov or rowchol is needed.
    colcov: nxn array
        Among-column covariance matrix. If rowcov is the identity matrix,
        this functions as `cov` in MvNormal.
        Exactly one of colcov or colchol is needed.
    colchol: nxn array
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
                               rowcov=rowcov, shape=(m, n))

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
            scale = pm.Lognormal('scale', mu=np.log(true_scale), sigma=0.5)
            rowcov = tt.nlinalg.diag([scale**(2*i) for i in range(m)])

            vals = pm.MatrixNormal('vals', mu=mu, colchol=colchol, rowcov=rowcov,
                                   observed=data, shape=(m, n))
    """

    def __init__(
        self,
        mu=0,
        rowcov=None,
        rowchol=None,
        rowtau=None,
        colcov=None,
        colchol=None,
        coltau=None,
        shape=None,
        *args,
        **kwargs,
    ):
        self._setup_matrices(colcov, colchol, coltau, rowcov, rowchol, rowtau)
        if shape is None:
            raise TypeError("shape is a required argument")
        assert len(shape) == 2, "shape must have length 2: mxn"
        self.shape = shape
        super().__init__(shape=shape, *args, **kwargs)
        self.mu = tt.as_tensor_variable(mu)
        self.mean = self.median = self.mode = self.mu
        self.solve_lower = tt.slinalg.solve_lower_triangular
        self.solve_upper = tt.slinalg.solve_upper_triangular

    def _setup_matrices(self, colcov, colchol, coltau, rowcov, rowchol, rowtau):
        cholesky = Cholesky(lower=True, on_error="raise")

        # Among-row matrices
        if len([i for i in [rowtau, rowcov, rowchol] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. "
                "Specify exactly one of rowtau, rowcov, "
                "or rowchol."
            )
        if rowcov is not None:
            self.m = rowcov.shape[0]
            self._rowcov_type = "cov"
            rowcov = tt.as_tensor_variable(rowcov)
            if rowcov.ndim != 2:
                raise ValueError("rowcov must be two dimensional.")
            self.rowchol_cov = cholesky(rowcov)
            self.rowcov = rowcov
        elif rowtau is not None:
            raise ValueError("rowtau not supported at this time")
            self.m = rowtau.shape[0]
            self._rowcov_type = "tau"
            rowtau = tt.as_tensor_variable(rowtau)
            if rowtau.ndim != 2:
                raise ValueError("rowtau must be two dimensional.")
            self.rowchol_tau = cholesky(rowtau)
            self.rowtau = rowtau
        else:
            self.m = rowchol.shape[0]
            self._rowcov_type = "chol"
            if rowchol.ndim != 2:
                raise ValueError("rowchol must be two dimensional.")
            self.rowchol_cov = tt.as_tensor_variable(rowchol)

        # Among-column matrices
        if len([i for i in [coltau, colcov, colchol] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. "
                "Specify exactly one of coltau, colcov, "
                "or colchol."
            )
        if colcov is not None:
            self.n = colcov.shape[0]
            self._colcov_type = "cov"
            colcov = tt.as_tensor_variable(colcov)
            if colcov.ndim != 2:
                raise ValueError("colcov must be two dimensional.")
            self.colchol_cov = cholesky(colcov)
            self.colcov = colcov
        elif coltau is not None:
            raise ValueError("coltau not supported at this time")
            self.n = coltau.shape[0]
            self._colcov_type = "tau"
            coltau = tt.as_tensor_variable(coltau)
            if coltau.ndim != 2:
                raise ValueError("coltau must be two dimensional.")
            self.colchol_tau = cholesky(coltau)
            self.coltau = coltau
        else:
            self.n = colchol.shape[0]
            self._colcov_type = "chol"
            if colchol.ndim != 2:
                raise ValueError("colchol must be two dimensional.")
            self.colchol_cov = tt.as_tensor_variable(colchol)

    def random(self, point=None, size=None):
        """
        Draw random values from Matrix-valued Normal distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, colchol, rowchol = draw_values(
            [self.mu, self.colchol_cov, self.rowchol_cov], point=point, size=size
        )
        size = to_tuple(size)
        dist_shape = to_tuple(self.shape)
        output_shape = size + dist_shape

        # Broadcasting all parameters
        (mu,) = broadcast_dist_samples_to(to_shape=output_shape, samples=[mu], size=size)
        rowchol = np.broadcast_to(rowchol, shape=size + rowchol.shape[-2:])

        colchol = np.broadcast_to(colchol, shape=size + colchol.shape[-2:])
        colchol = np.swapaxes(colchol, -1, -2)  # Take transpose

        standard_normal = np.random.standard_normal(output_shape)
        samples = mu + np.matmul(rowchol, np.matmul(standard_normal, colchol))
        return samples

    def _trquaddist(self, value):
        """Compute Tr[colcov^-1 @ (x - mu).T @ rowcov^-1 @ (x - mu)] and
        the logdet of colcov and rowcov."""

        delta = value - self.mu
        rowchol_cov = self.rowchol_cov
        colchol_cov = self.colchol_cov

        # Find exponent piece by piece
        right_quaddist = self.solve_lower(rowchol_cov, delta)
        quaddist = tt.nlinalg.matrix_dot(right_quaddist.T, right_quaddist)
        quaddist = self.solve_lower(colchol_cov, quaddist)
        quaddist = self.solve_upper(colchol_cov.T, quaddist)
        trquaddist = tt.nlinalg.trace(quaddist)

        coldiag = tt.nlinalg.diag(colchol_cov)
        rowdiag = tt.nlinalg.diag(rowchol_cov)
        half_collogdet = tt.sum(tt.log(coldiag))  # logdet(M) = 2*Tr(log(L))
        half_rowlogdet = tt.sum(tt.log(rowdiag))  # Using Cholesky: M = L L^T
        return trquaddist, half_collogdet, half_rowlogdet

    def logp(self, value):
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
        trquaddist, half_collogdet, half_rowlogdet = self._trquaddist(value)
        m = self.m
        n = self.n
        norm = -0.5 * m * n * pm.floatX(np.log(2 * np.pi))
        return norm - 0.5 * trquaddist - m * half_collogdet - n * half_rowlogdet

    def _distr_parameters_for_repr(self):
        mapping = {"tau": "tau", "cov": "cov", "chol": "chol_cov"}
        return ["mu", "row" + mapping[self._rowcov_type], "col" + mapping[self._colcov_type]]


class KroneckerNormal(Continuous):
    R"""
    Multivariate normal log-likelihood with Kronecker-structured covariance.

    .. math::

       f(x \mid \mu, K) =
           \frac{1}{(2\pi |K|)^{1/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} K^{-1} (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^N`
    Mean      :math:`\mu`
    Variance  :math:`K = \bigotimes K_i` + \sigma^2 I_N
    ========  ==========================

    Parameters
    ----------
    mu: array
        Vector of means, just as in `MvNormal`.
    covs: list of arrays
        The set of covariance matrices :math:`[K_1, K_2, ...]` to be
        Kroneckered in the order provided :math:`\bigotimes K_i`.
    chols: list of arrays
        The set of lower cholesky matrices :math:`[L_1, L_2, ...]` such that
        :math:`K_i = L_i L_i'`.
    evds: list of tuples
        The set of eigenvalue-vector, eigenvector-matrix pairs
        :math:`[(v_1, Q_1), (v_2, Q_2), ...]` such that
        :math:`K_i = Q_i \text{diag}(v_i) Q_i'`. For example::

            v_i, Q_i = tt.nlinalg.eigh(K_i)

    sigma: scalar, variable
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

    Effeciency gains are made by cholesky decomposing :math:`K_1` and
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

    def __init__(self, mu, covs=None, chols=None, evds=None, sigma=None, *args, **kwargs):
        self._setup(covs, chols, evds, sigma)
        super().__init__(*args, **kwargs)
        self.mu = tt.as_tensor_variable(mu)
        self.mean = self.median = self.mode = self.mu

    def _setup(self, covs, chols, evds, sigma):
        self.cholesky = Cholesky(lower=True, on_error="raise")
        if len([i for i in [covs, chols, evds] if i is not None]) != 1:
            raise ValueError(
                "Incompatible parameterization. Specify exactly one of covs, chols, or evds."
            )
        self._isEVD = False
        self.sigma = sigma
        self.is_noisy = self.sigma is not None and self.sigma != 0
        if covs is not None:
            self._cov_type = "cov"
            self.covs = covs
            if self.is_noisy:
                # Noise requires eigendecomposition
                eigh_map = map(eigh, covs)
                self._setup_evd(eigh_map)
            else:
                # Otherwise use cholesky as usual
                self.chols = list(map(self.cholesky, self.covs))
                self.chol_diags = list(map(tt.nlinalg.diag, self.chols))
                self.sizes = tt.as_tensor_variable([chol.shape[0] for chol in self.chols])
                self.N = tt.prod(self.sizes)
        elif chols is not None:
            self._cov_type = "chol"
            if self.is_noisy:  # A strange case...
                # Noise requires eigendecomposition
                covs = [tt.dot(chol, chol.T) for chol in chols]
                eigh_map = map(eigh, covs)
                self._setup_evd(eigh_map)
            else:
                self.chols = chols
                self.chol_diags = list(map(tt.nlinalg.diag, self.chols))
                self.sizes = tt.as_tensor_variable([chol.shape[0] for chol in self.chols])
                self.N = tt.prod(self.sizes)
        else:
            self._cov_type = "evd"
            self._setup_evd(evds)

    def _setup_evd(self, eigh_iterable):
        self._isEVD = True
        eigs_sep, Qs = zip(*eigh_iterable)  # Unzip
        self.Qs = list(map(tt.as_tensor_variable, Qs))
        self.QTs = list(map(tt.transpose, self.Qs))

        self.eigs_sep = list(map(tt.as_tensor_variable, eigs_sep))
        self.eigs = kron_diag(*self.eigs_sep)  # Combine separate eigs
        if self.is_noisy:
            self.eigs += self.sigma ** 2
        self.N = self.eigs.shape[0]

    def _setup_random(self):
        if not hasattr(self, "mv_params"):
            self.mv_params = {"mu": self.mu}
            if self._cov_type == "cov":
                cov = kronecker(*self.covs)
                if self.is_noisy:
                    cov = cov + self.sigma ** 2 * tt.identity_like(cov)
                self.mv_params["cov"] = cov
            elif self._cov_type == "chol":
                if self.is_noisy:
                    covs = []
                    for eig, Q in zip(self.eigs_sep, self.Qs):
                        cov_i = tt.dot(Q, tt.dot(tt.diag(eig), Q.T))
                        covs.append(cov_i)
                    cov = kronecker(*covs)
                    if self.is_noisy:
                        cov = cov + self.sigma ** 2 * tt.identity_like(cov)
                    self.mv_params["chol"] = self.cholesky(cov)
                else:
                    self.mv_params["chol"] = kronecker(*self.chols)
            elif self._cov_type == "evd":
                covs = []
                for eig, Q in zip(self.eigs_sep, self.Qs):
                    cov_i = tt.dot(Q, tt.dot(tt.diag(eig), Q.T))
                    covs.append(cov_i)
                cov = kronecker(*covs)
                if self.is_noisy:
                    cov = cov + self.sigma ** 2 * tt.identity_like(cov)
                self.mv_params["cov"] = cov

    def random(self, point=None, size=None):
        """
        Draw random values from Multivariate Normal distribution
        with Kronecker-structured covariance.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        # Expand params into terms MvNormal can understand to force consistency
        self._setup_random()
        self.mv_params["shape"] = self.shape
        dist = MvNormal.dist(**self.mv_params)
        return dist.random(point, size)

    def _quaddist(self, value):
        """Computes the quadratic (x-mu)^T @ K^-1 @ (x-mu) and log(det(K))"""
        if value.ndim > 2 or value.ndim == 0:
            raise ValueError("Invalid dimension for value: %s" % value.ndim)
        if value.ndim == 1:
            onedim = True
            value = value[None, :]
        else:
            onedim = False

        delta = value - self.mu
        if self._isEVD:
            sqrt_quad = kron_dot(self.QTs, delta.T)
            sqrt_quad = sqrt_quad / tt.sqrt(self.eigs[:, None])
            logdet = tt.sum(tt.log(self.eigs))
        else:
            sqrt_quad = kron_solve_lower(self.chols, delta.T)
            logdet = 0
            for chol_size, chol_diag in zip(self.sizes, self.chol_diags):
                logchol = tt.log(chol_diag) * self.N / chol_size
                logdet += tt.sum(2 * logchol)
        # Square each sample
        quad = tt.batched_dot(sqrt_quad.T, sqrt_quad.T)
        if onedim:
            quad = quad[0]
        return quad, logdet

    def logp(self, value):
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
        quad, logdet = self._quaddist(value)
        return -(quad + logdet + self.N * tt.log(2 * np.pi)) / 2.0

    def _distr_parameters_for_repr(self):
        return ["mu"]
