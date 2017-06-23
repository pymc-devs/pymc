#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import scipy
import theano
import theano.tensor as tt

from scipy import stats, linalg

from theano.tensor.nlinalg import det, matrix_inverse, trace

import pymc3 as pm

from pymc3.math import tround
from pymc3.theanof import floatX
from . import transforms
from pymc3.util import get_variable_name
from .distribution import Continuous, Discrete, draw_values, generate_samples
from ..model import Deterministic
from .continuous import ChiSquared, Normal
from .special import gammaln, multigammaln
from .dist_math import bound, logpow, factln, Cholesky


__all__ = ['MvNormal', 'MvStudentT', 'Dirichlet',
           'Multinomial', 'Wishart', 'WishartBartlett',
           'LKJCorr', 'LKJCholeskyCov']


class MvNormal(Continuous):
    R"""
    Multivariate normal log-likelihood.

    .. math::

       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{1/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : array
        Vector of means.
    cov : array
        Covariance matrix. Exactly one of cov, tau, or chol is needed.
    tau : array
        Precision matrix. Exactly one of cov, tau, or chol is needed.
    chol : array
        Cholesky decomposition of covariance matrix. Exactly one of cov,
        tau, or chol is needed.
    lower : bool, default=True
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

        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
            n=3, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(3, chol_packed)
        vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)

    For unobserved values it can be better to use a non-centered
    parametrization::

        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
            n=3, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(3, chol_packed)
        vals_raw = pm.Normal('vals_raw', mu=0, sd=1, shape=(5, 3))
        vals = pm.Deterministic('vals', tt.dot(chol, vals_raw.T).T)
    """

    def __init__(self, mu, cov=None, tau=None, chol=None, lower=True,
                 *args, **kwargs):
        super(MvNormal, self).__init__(*args, **kwargs)
        if len(self.shape) > 2:
            raise ValueError("Only 1 or 2 dimensions are allowed.")

        if not lower:
            chol = chol.T
        if len([i for i in [tau, cov, chol] if i is not None]) != 1:
            raise ValueError('Incompatible parameterization. '
                             'Specify exactly one of tau, cov, '
                             'or chol.')
        mu = tt.as_tensor_variable(mu)
        self.mean = self.median = self.mode = self.mu = mu
        self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        # Step methods and advi do not catch LinAlgErrors at the
        # moment. We work around that by using a cholesky op
        # that returns a nan as first entry instead of raising
        # an error.
        cholesky = Cholesky(nofail=True, lower=True)

        if cov is not None:
            self.k = cov.shape[0]
            self._cov_type = 'cov'
            cov = tt.as_tensor_variable(cov)
            if cov.ndim != 2:
                raise ValueError('cov must be two dimensional.')
            self.chol_cov = cholesky(cov)
            self.cov = cov
            self._n = self.cov.shape[-1]
        elif tau is not None:
            self.k = tau.shape[0]
            self._cov_type = 'tau'
            tau = tt.as_tensor_variable(tau)
            if tau.ndim != 2:
                raise ValueError('tau must be two dimensional.')
            self.chol_tau = cholesky(tau)
            self.tau = tau
            self._n = self.tau.shape[-1]
        else:
            self.k = chol.shape[0]
            self._cov_type = 'chol'
            if chol.ndim != 2:
                raise ValueError('chol must be two dimensional.')
            self.chol_cov = tt.as_tensor_variable(chol)
            self._n = self.chol_cov.shape[-1]

    def random(self, point=None, size=None):
        if size is None:
            size = []
        else:
            try:
                size = list(size)
            except TypeError:
                size = [size]

        if self._cov_type == 'cov':
            mu, cov = draw_values([self.mu, self.cov], point=point)
            if mu.shape != cov[0].shape:
                raise ValueError("Shapes for mu an cov don't match")

            try:
                dist = stats.multivariate_normal(
                    mean=mu, cov=cov, allow_singular=True)
            except ValueError as error:
                size.append(mu.shape[0])
                return np.nan * np.zeros(size)
            return dist.rvs(size)
        elif self._cov_type == 'chol':
            mu, chol = draw_values([self.mu, self.chol_cov], point=point)
            if mu.shape != chol[0].shape:
                raise ValueError("Shapes for mu an chol don't match")

            size.append(mu.shape[0])
            standard_normal = np.random.standard_normal(size)
            return mu + np.dot(standard_normal, chol.T)
        else:
            mu, tau = draw_values([self.mu, self.tau], point=point)
            if mu.shape != tau[0].shape:
                raise ValueError("Shapes for mu an tau don't match")

            size.append(mu.shape[0])
            standard_normal = np.random.standard_normal(size)
            try:
                chol = linalg.cholesky(tau, lower=True)
            except linalg.LinAlgError:
                return np.nan * np.zeros(size)
            transformed = linalg.solve_triangular(
                chol, standard_normal.T, lower=True)
            return mu + transformed.T

    def logp(self, value):
        mu = self.mu
        if value.ndim > 2 or value.ndim == 0:
            raise ValueError('Invalid dimension for value: %s' % value.ndim)
        if value.ndim == 1:
            onedim = True
            value = value[None, :]
        else:
            onedim = False

        delta = value - mu

        if self._cov_type == 'cov':
            # Use this when Theano#5908 is released.
            # return MvNormalLogp()(self.cov, delta)
            logp = self._logp_cov(delta)
        elif self._cov_type == 'tau':
            logp = self._logp_tau(delta)
        else:
            logp = self._logp_chol(delta)

        if onedim:
            return logp[0]
        return logp

    def _logp_chol(self, delta):
        chol_cov = self.chol_cov
        _, k = delta.shape
        k = pm.floatX(k)
        diag = tt.nlinalg.diag(chol_cov)
        # Check if the covariance matrix is positive definite.
        ok = tt.all(diag > 0)
        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        chol_cov = tt.switch(ok, chol_cov, 1)

        delta_trans = self.solve_lower(chol_cov, delta.T)

        result = k * pm.floatX(np.log(2. * np.pi))
        result += 2.0 * tt.sum(tt.log(diag))
        result += (delta_trans ** 2).sum(axis=0)
        result = -0.5 * result
        return bound(result, ok)

    def _logp_cov(self, delta):
        chol_cov = self.chol_cov
        _, k = delta.shape
        k = pm.floatX(k)

        diag = tt.nlinalg.diag(chol_cov)
        ok = tt.all(diag > 0)

        chol_cov = tt.switch(ok, chol_cov, 1)
        diag = tt.nlinalg.diag(chol_cov)
        delta_trans = self.solve_lower(chol_cov, delta.T)

        result = k * pm.floatX(np.log(2. * np.pi))
        result += 2.0 * tt.sum(tt.log(diag))
        result += (delta_trans ** 2).sum(axis=0)
        result = -0.5 * result
        return bound(result, ok)

    def _logp_tau(self, delta):
        chol_tau = self.chol_tau
        _, k = delta.shape
        k = pm.floatX(k)

        diag = tt.nlinalg.diag(chol_tau)
        ok = tt.all(diag > 0)

        chol_tau = tt.switch(ok, chol_tau, 1)
        diag = tt.nlinalg.diag(chol_tau)
        delta_trans = tt.dot(chol_tau.T, delta.T)

        result = k * pm.floatX(np.log(2. * np.pi))
        result -= 2.0 * tt.sum(tt.log(diag))
        result += (delta_trans ** 2).sum(axis=0)
        result = -0.5 * result
        return bound(result, ok)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        try:
            cov = dist.cov
        except AttributeError:
            cov = dist.chol_cov
        name_mu = get_variable_name(mu)
        name_cov = get_variable_name(cov)
        return (r'${} \sim \text{{MvNormal}}'
                r'(\mathit{{mu}}={}, \mathit{{cov}}={})$'
                .format(name, name_mu, name_cov))


class MvStudentT(Continuous):
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
             \right]^{(\nu+p)/2}}


    ========  =============================================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu` if :math:`\nu > 1` else undefined
    Variance  :math:`\frac{\nu}{\mu-2}\Sigma`
                  if :math:`\nu>2` else undefined
    ========  =============================================


    Parameters
    ----------
    nu : int
        Degrees of freedom.
    Sigma : matrix
        Covariance matrix.
    mu : array
        Vector of means.
    """

    def __init__(self, nu, Sigma, mu=None, *args, **kwargs):
        super(MvStudentT, self).__init__(*args, **kwargs)
        self.nu = nu = tt.as_tensor_variable(nu)
        mu = tt.zeros(Sigma.shape[0]) if mu is None else tt.as_tensor_variable(mu)
        self.Sigma = Sigma = tt.as_tensor_variable(Sigma)

        self.mean = self.median = self.mode = self.mu = mu

    def random(self, point=None, size=None):
        chi2 = np.random.chisquare
        mvn = np.random.multivariate_normal

        nu, S, mu = draw_values([self.nu, self.Sigma, self.mu], point=point)

        return (np.sqrt(nu) * (mvn(np.zeros(len(S)), S, size).T
                               / chi2(nu, size))).T + mu

    def logp(self, value):

        S = self.Sigma
        nu = self.nu
        mu = self.mu

        d = S.shape[0]

        X = value - mu

        Q = X.dot(matrix_inverse(S)).dot(X.T).sum()
        log_det = tt.log(det(S))
        log_pdf = gammaln((nu + d) / 2.) - 0.5 * \
            (d * tt.log(np.pi * nu) + log_det) - gammaln(nu / 2.)
        log_pdf -= 0.5 * (nu + d) * tt.log(1 + Q / nu)

        return log_pdf

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        nu = dist.nu
        Sigma = dist.Sigma
        name_nu = get_variable_name(nu)
        name_mu = get_variable_name(mu)
        name_sigma = get_variable_name(Sigma)
        return (r'${} \sim \text{{MvStudentT}}'
                r'(\mathit{{nu}}={}, \mathit{{mu}}={}, '
                r'\mathit{{Sigma}}={})$'
                .format(name, name_nu, name_mu, name_sigma))


class Dirichlet(Continuous):
    R"""
    Dirichlet log-likelihood.

    .. math::

       f(\mathbf{x}) =
           \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}
           \prod_{i=1}^{k-1} x_i^{\theta_i - 1}
           \left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

    ========  ===============================================
    Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum x_i = 1`
    Mean      :math:`\dfrac{a_i}{\sum a_i}`
    Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}`
              where :math:`a_0 = \sum a_i`
    ========  ===============================================

    Parameters
    ----------
    a : array
        Concentration parameters (a > 0).

    Notes
    -----
    Only the first `k-1` elements of `x` are expected. Can be used
    as a parent of Multinomial and Categorical nevertheless.
    """

    def __init__(self, a, transform=transforms.stick_breaking,
                 *args, **kwargs):
        shape = a.shape[-1]
        kwargs.setdefault("shape", shape)
        super(Dirichlet, self).__init__(transform=transform, *args, **kwargs)

        self.k = tt.as_tensor_variable(shape)
        self.a = a = tt.as_tensor_variable(a)
        self.mean = a / tt.sum(a)

        self.mode = tt.switch(tt.all(a > 1),
                              (a - 1) / tt.sum(a - 1),
                              np.nan)

    def random(self, point=None, size=None):
        a = draw_values([self.a], point=point)[0]

        def _random(a, size=None):
            return stats.dirichlet.rvs(a, None if size == a.shape else size)

        samples = generate_samples(_random, a,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, value):
        k = self.k
        a = self.a

        # only defined for sum(value) == 1
        return bound(tt.sum(logpow(value, a - 1) - gammaln(a), axis=-1)
                     + gammaln(tt.sum(a, axis=-1)),
                     tt.all(value >= 0), tt.all(value <= 1),
                     k > 1, tt.all(a > 0),
                     broadcast_conditions=False)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        return r'${} \sim \text{{Dirichlet}}(\mathit{{a}}={})$'.format(name,
                                                get_variable_name(a))


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
    n : int or array
        Number of trials (n > 0).
    p : one- or two-dimensional array
        Probability of each one of the different outcomes. Elements must
        be non-negative and sum to 1 along the last axis. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, n, p, *args, **kwargs):
        super(Multinomial, self).__init__(*args, **kwargs)

        p = p / tt.sum(p, axis=-1, keepdims=True)

        if len(self.shape) == 2:
            try:
                assert n.shape == (self.shape[0],)
            except AttributeError:
                # this occurs when n is a scalar Python int or float
                n *= tt.ones(self.shape[0])

            self.n = tt.shape_padright(n)
            self.p = p if p.ndim == 2 else tt.shape_padleft(p)
        else:
            self.n = tt.as_tensor_variable(n)
            self.p = tt.as_tensor_variable(p)

        self.mean = self.n * self.p
        self.mode = tt.cast(tround(self.mean), 'int32')

    def _random(self, n, p, size=None):
        if size == p.shape:
            size = None
        return np.random.multinomial(n, p, size=size)

    def random(self, point=None, size=None):
        n, p = draw_values([self.n, self.p], point=point)
        samples = generate_samples(self._random, n, p,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, x):
        n = self.n
        p = self.p

        return bound(
            tt.sum(factln(n)) - tt.sum(factln(x)) + tt.sum(x * tt.log(p)),
            tt.all(x >= 0),
            tt.all(tt.eq(tt.sum(x, axis=-1, keepdims=True), n)),
            tt.all(p <= 1),
            tt.all(tt.eq(tt.sum(p, axis=-1), 1)),
            tt.all(tt.ge(n, 0)),
            broadcast_conditions=False
        )

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        n = dist.n
        p = dist.p
        return r'${} \sim \text{{Multinomial}}(\mathit{{n}}={}, \mathit{{p}}={})$'.format(name,
                                                get_variable_name(n),
                                                get_variable_name(p))


def posdef(AA):
    try:
        np.linalg.cholesky(AA)
        return 1
    except np.linalg.LinAlgError:
        return 0


class PosDefMatrix(theano.Op):
    """
    Check if input is positive definite. Input should be a square matrix.

    """

    # Properties attribute
    __props__ = ()

    # Compulsory if itypes and otypes are not defined

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        o = tt.TensorType(dtype='int8', broadcastable=[])()
        return theano.Apply(self, [x], [o])

    # Python implementation:
    def perform(self, node, inputs, outputs):

        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.array(posdef(x), dtype='int8')
        except Exception:
            pm._log.exception('Failed to check if positive definite', x)
            raise

    def infer_shape(self, node, shapes):
        return [[]]

    def grad(self, inp, grads):
        x, = inp
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
    nu : int
        Degrees of freedom, > 0.
    V : array
        p x p positive definite matrix.

    Note
    ----
    This distribution is unusable in a PyMC3 model. You should instead
    use LKJCholeskyCov or LKJCorr.
    """

    def __init__(self, nu, V, *args, **kwargs):
        super(Wishart, self).__init__(*args, **kwargs)
        warnings.warn('The Wishart distribution can currently not be used '
                      'for MCMC sampling. The probability of sampling a '
                      'symmetric matrix is basically zero. Instead, please '
                      'use LKJCholeskyCov or LKJCorr. For more information '
                      'on the issues surrounding the Wishart see here: '
                      'https://github.com/pymc-devs/pymc3/issues/538.',
                      UserWarning)
        self.nu = nu = tt.as_tensor_variable(nu)
        self.p = p = tt.as_tensor_variable(V.shape[0])
        self.V = V = tt.as_tensor_variable(V)
        self.mean = nu * V
        self.mode = tt.switch(1 * (nu >= p + 1),
                              (nu - p - 1) * V,
                              np.nan)

    def logp(self, X):
        nu = self.nu
        p = self.p
        V = self.V

        IVI = det(V)
        IXI = det(X)

        return bound(((nu - p - 1) * tt.log(IXI)
                      - trace(matrix_inverse(V).dot(X))
                      - nu * p * tt.log(2) - nu * tt.log(IVI)
                      - 2 * multigammaln(nu / 2., p)) / 2,
                     matrix_pos_def(X),
                     tt.eq(X, X.T),
                     nu > (p - 1),
                     broadcast_conditions=False
        )

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.nu
        V = dist.V
        return r'${} \sim \text{{Wishart}}(\mathit{{nu}}={}, \mathit{{V}}={})$'.format(name,
                                                get_variable_name(nu),
                                                get_variable_name(V))

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
    S : ndarray
        p x p positive definite matrix
        Or:
        p x p lower-triangular matrix that is the Cholesky factor
        of the covariance matrix.
    nu : int
        Degrees of freedom, > dim(S).
    is_cholesky : bool (default=False)
        Input matrix S is already Cholesky decomposed as S.T * S
    return_cholesky : bool (default=False)
        Only return the Cholesky decomposed matrix.
    testval : ndarray
        p x p positive definite matrix used to initialize

    Note
    ----
    This is not a standard Distribution class but follows a similar
    interface. Besides the Wishart distribution, it will add RVs
    c and z to your model which make up the matrix.

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
        testval = scipy.linalg.cholesky(testval, lower=True)
        diag_testval = testval[diag_idx]**2
        tril_testval = testval[tril_idx]
    else:
        diag_testval = None
        tril_testval = None

    c = tt.sqrt(ChiSquared('c', nu - np.arange(2, 2 + n_diag), shape=n_diag,
                           testval=diag_testval))
    pm._log.info('Added new variable c to model diagonal of Wishart.')
    z = Normal('z', 0., 1., shape=n_tril, testval=tril_testval)
    pm._log.info('Added new variable z to model off-diagonals of Wishart.')
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
        result = gammaln(2. * tt.arange(1, int((n - 1) / 2) + 1)).sum()
        if n % 2 == 1:
            result += (0.25 * (n ** 2 - 1) * tt.log(np.pi)
                       - 0.25 * (n - 1) ** 2 * tt.log(2.)
                       - (n - 1) * gammaln(int((n + 1) / 2)))
        else:
            result += (0.25 * n * (n - 2) * tt.log(np.pi)
                       + 0.25 * (3 * n ** 2 - 4 * n) * tt.log(2.)
                       + n * gammaln(n / 2) - (n - 1) * gammaln(n))
    else:
        result = -(n - 1) * gammaln(eta + 0.5 * (n - 1))
        k = tt.arange(1, n)
        result += (0.5 * k * tt.log(np.pi)
                   + gammaln(eta + 0.5 * (n - 1 - k))).sum()
    return result


class LKJCholeskyCov(Continuous):
    R"""Covariance matrix with LKJ distributed correlations.

    This defines a distribution over cholesky decomposed covariance
    matrices, such that the underlying correlation matrices follow an
    LKJ distribution [1] and the standard deviations follow an arbitray
    distribution specified by the user.

    Parameters
    ----------
    n : int
        Dimension of the covariance matrix (n > 0).
    eta : float
        The shape parameter (eta > 0) of the LKJ distribution. eta = 1
        implies a uniform distribution of the correlation matrices;
        larger values put more weight on matrices with few correlations.
    sd_dist : pm.Distribution
        A distribution for the standard deviations.

    Notes
    -----
    Since the cholesky factor is a lower triangular matrix, we use
    packed storge for the matrix: We store and return the values of
    the lower triangular matrix in a one-dimensional array, numbered
    by row::

        [[0 - - -]
         [1 2 - -]
         [3 4 5 -]
         [6 7 8 9]]

    You can use `pm.expand_packed_triangular(packed_cov, lower=True)`
    to convert this to a regular two-dimensional array.

    Examples
    --------
    .. code:: python

        with pm.Model() as model:
            # Note that we access the distribution for the standard
            # deviations, and do not create a new random variable.
            sd_dist = pm.HalfCauchy.dist(beta=2.5)
            packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=10, sd_dist=sd_dist)
            chol = pm.expand_packed_triangular(10, packed_chol, lower=True)

            # Define a new MvNormal with the given covariance
            vals = pm.MvNormal('vals', mu=np.zeros(10), chol=chol, shape=10)

            # Or transform an uncorrelated normal:
            vals_raw = pm.Normal('vals_raw', mu=np.zeros(10), sd=1)
            vals = tt.dot(chol, vals_raw)

            # Or compute the covariance matrix
            cov = tt.dot(chol, chol.T)

            # Extract the standard deviations
            stds = tt.sqrt(tt.diag(cov))

    Implementation
    --------------
    In the unconstrained space all values of the cholesky factor
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
    compute the different likelihoods seperatly, as the likelihood
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
    def __init__(self, eta, n, sd_dist, *args, **kwargs):
        self.n = n
        self.eta = eta

        if 'transform' in kwargs:
            raise ValueError('Invalid parameter: transform.')
        if 'shape' in kwargs:
            raise ValueError('Invalid parameter: shape.')

        shape = n * (n + 1) // 2

        if sd_dist.shape.ndim not in [0, 1]:
            raise ValueError('Invalid shape for sd_dist.')

        transform = transforms.CholeskyCovPacked(n)

        kwargs['shape'] = shape
        kwargs['transform'] = transform
        super(LKJCholeskyCov, self).__init__(*args, **kwargs)

        self.sd_dist = sd_dist
        self.diag_idxs = transform.diag_idxs

        self.mode = floatX(np.zeros(shape))
        self.mode[self.diag_idxs] = 1

    def logp(self, x):
        n = self.n
        eta = self.eta

        diag_idxs = self.diag_idxs
        cumsum = tt.cumsum(x ** 2)
        variance = tt.zeros(n)
        variance = tt.inc_subtensor(variance[0], x[0] ** 2)
        variance = tt.inc_subtensor(
            variance[1:],
            cumsum[diag_idxs[1:]] - cumsum[diag_idxs[:-1]])
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
    n : int
        Dimension of the covariance matrix (n > 0).
    eta : float
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

    def __init__(self, eta=None, n=None, p=None, transform='interval', *args, **kwargs):
        if (p is not None) and (n is not None) and (eta is None):
            warnings.warn('Parameters to LKJCorr have changed: shape parameter n -> eta '
                          'dimension parameter p -> n. Please update your code. '
                          'Automatically re-assigning parameters for backwards compatibility.',
                          DeprecationWarning)
            self.n = p
            self.eta = n
            eta = self.eta
            n = self.n
        elif (n is not None) and (eta is not None) and (p is None):
            self.n = n
            self.eta = eta
        else:
            raise ValueError('Invalid parameter: please use eta as the shape parameter and '
                             'n as the dimension parameter.')

        n_elem = int(n * (n - 1) / 2)
        self.mean = np.zeros(n_elem, dtype=theano.config.floatX)

        if transform == 'interval':
            transform = transforms.interval(-1, 1)

        super(LKJCorr, self).__init__(shape=n_elem, transform=transform,
                                      *args, **kwargs)
        warnings.warn('Parameters in LKJCorr have been rename: shape parameter n -> eta '
                      'dimension parameter p -> n. Please double check your initialization.',
                      DeprecationWarning)
        self.tri_index = np.zeros([n, n], dtype='int32')
        self.tri_index[np.triu_indices(n, k=1)] = np.arange(n_elem)
        self.tri_index[np.triu_indices(n, k=1)[::-1]] = np.arange(n_elem)

    def logp(self, x):
        n = self.n
        eta = self.eta

        X = x[self.tri_index]
        X = tt.fill_diagonal(X, 1)

        result = _lkj_normalizing_constant(eta, n)
        result += (eta - 1.) * tt.log(det(X))
        return bound(result,
                     tt.all(X <= 1), tt.all(X >= -1),
                     matrix_pos_def(X),
                     eta > 0,
                     broadcast_conditions=False
        )
