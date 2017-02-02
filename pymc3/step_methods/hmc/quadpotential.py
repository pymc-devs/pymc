from numpy import dot
from numpy.random import normal
import scipy.linalg
from scipy import linalg
from theano.tensor import slinalg
import theano.tensor as tt
import theano
from scipy.sparse import issparse
import itertools

from pymc3.theanof import floatX

import numpy as np

__all__ = ['quad_potential', 'ElemWiseQuadPotential', 'QuadPotential',
           'QuadPotential_Inv', 'isquadpotential', 'QuadPotentialSample']


def quad_potential(C, is_cov, as_cov):
    """
    Parameters
    ----------
    C : arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
        If C is already a potential, it is returned unchanged.
    is_cov : Boolean
        whether C is provided as a covariance matrix or hessian
    as_cov : Boolean
        whether the random draws should come from the normal dist
        using the covariance matrix above or the inverse

    Returns
    -------
    q : Quadpotential
    """
    if isquadpotential(C):
        return C

    if issparse(C):
        if not chol_available:
            raise ImportError("Sparse mass matrices require scikits.sparse")
        if is_cov != as_cov:
            return QuadPotential_Sparse(C)
        else:
            raise ValueError("Sparse precission matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov != as_cov:
            return ElemWiseQuadPotential(C)
        else:
            return ElemWiseQuadPotential(1. / C)
    else:
        if is_cov != as_cov:
            return QuadPotential(C)
        else:
            return QuadPotential_Inv(C)


def partial_check_positive_definite(C):
    """Simple but partial check for Positive Definiteness"""
    if C.ndim == 1:
        d = C
    else:
        d = np.diag(C)
    i, = np.nonzero(np.logical_or(np.isnan(d), d <= 0))

    if len(i):
        raise PositiveDefiniteError(
            "Simple check failed. Diagonal contains negatives", i)


class PositiveDefiniteError(ValueError):

    def __init__(self, msg, idx):
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return "Scaling is not positive definite. " + self.msg + ". Check indexes " + str(self.idx)


def isquadpotential(o):
    return all(hasattr(o, attr) for attr in ["velocity", "random", "energy"])


class ElemWiseQuadPotential(object):

    def __init__(self, v):
        v = floatX(v)
        s = v ** .5

        self.s = s
        self.inv_s = 1. / s
        self.v = v

    def velocity(self, x):
        return self.v * x

    def random(self):
        return floatX(normal(size=self.s.shape)) * self.inv_s

    def energy(self, x):
        return .5 * x.dot(self.v * x)


class QuadPotential_Inv(object):

    def __init__(self, A):
        self.L = floatX(scipy.linalg.cholesky(A, lower=True))

    def velocity(self, x):
        solve = slinalg.Solve(lower=True)
        y = solve(self.L, x)
        return solve(self.L.T, y)

    def random(self):
        n = floatX(normal(size=self.L.shape[0]))
        return dot(self.L, n)

    def energy(self, x):
        L1x = slinalg.Solve(lower=True)(self.L, x)
        return .5 * L1x.T.dot(L1x)


class QuadPotential(object):

    def __init__(self, A):
        self.A = floatX(A)
        self.L = scipy.linalg.cholesky(A, lower=True)

    def velocity(self, x):
        return x.T.dot(self.A.T)

    def random(self):
        n = floatX(normal(size=self.L.shape[0]))
        return scipy.linalg.solve_triangular(self.L.T, n)

    def energy(self, x):
        return .5 * x.dot(self.A).dot(x)

    __call__ = random


def _solve_lanczos(V, alphas, betas, estimator=None):
    e = np.zeros(len(alphas))
    e[0] = 1
    T = [alphas, betas]
    # TODO Is there a way to avoid computing all eigenvectors?
    eig, eigv = linalg.eig_banded(T, lower=True)
    if estimator is None:
        if np.any(eig < 0):
            raise ValueError("Matrix is not positive definite")
        eig = 1 / np.sqrt(eig)
    else:
        eig = estimator(eig)
    f_T = eigv @ (eig * np.linalg.solve(eigv, e))
    return np.dot(np.transpose(V), f_T)


def _sample_mvn_lanczos(matprot, z, epsilon, maxiter=None,
                       verbose=False, estimator=None):
    """Sample from a multivariate normal using matrix-free lanczos.

    Given a matrix :math:`A`, sample from :math:`N(0, A^{-1})`, using
    only matrix vector products :math:`Ax`. This is done by approximating
    :math:`A^{-1/2}z` for a :math:`z \sim N(0, I)` using a Lanczos process
    as described in [1].

    Parameters
    ----------
    matprot : function(x) -> Ax
        A function that computes the matrix-vector product :math:`Ax`.
    z : ndarray
        A standard normal distributed vector.
    epsilon : float
        Stop iteration, if the relative change of the estimate is below
        `epsilon`.
    maxiter : int, optional
        Maximum number of iterations.
    estimator : function, optional
        A function that maps the eigenvalues of the krylov subspace to
        what is supposed to be estimated. By default this is
        `lambda eig: 1 / np.sqrt(eig)`. If you want to sample from
        :math:`N(0, A)` instead of from :math:`N(0, A^{-1/2})`, you can
        set this to `lambda eig: np.sqrt(eig)`.

    References
    ----------
    [1] Chow, E., and Y. Saad. “Preconditioned Krylov Subspace Methods for
        Sampling Multivariate Gaussian Distributions.” SIAM Journal on
        Scientific Computing 36, no. 2 (January 1, 2014): A588–608.
        doi:10.1137/130920587.
    """
    n = len(z)
    z_norm = np.linalg.norm(z)
    z = z / z_norm
    q = z.copy()
    beta = 1

    alphas = []
    betas = []
    V = []

    for n in itertools.count():
        if maxiter is not None and n > maxiter:
            raise ArithmeticError("Maximum number of iterations reached.")
        v = q / beta
        if n > 0:
            q = matprot(v) - beta * V[-1]
        else:
            q = matprot(v)

        alpha = v @ q
        q = q - alpha * v
        beta = np.linalg.norm(q)

        alphas.append(alpha)
        betas.append(beta)
        V.append(v)

        y_new = _solve_lanczos(V, alphas, betas, estimator)

        if beta == 0:
            return z_norm * y_new

        if n > 0:
            error = np.linalg.norm(y_new - y_old) / np.linalg.norm(y_new)
            if np.isnan(error):
                raise ArithmeticError("Current error is nan.")
            if verbose and n % 100 == 0:
                print(error)
            if error < epsilon:
                return z_norm * y_new
        y_old = y_new


class QuadPotentialSample:
    def __init__(self, samples, lam=0.01, epsilon=1e-8, maxiter=None):
        """Use posterior samples to estimate the covariance.

        Parameters
        ----------
        samples : ndarray
            An array of posterior samples with shape `(s, n)`, where
            `s` is the number of samples and `n` the number of parameters.
        lam : float
            Regularization parameter. Use :math:`(1 - \lambda)\Sigma +
            \lambda D` as covariance, where `\Sigma` is the sample covariance
            and D is the diagonal matrix containing the sample variances.
        epsilon : float, optional
            Error tolerance used in the convergence criterion of the
            lanczos process.
        maxiter : int, optional
            The maximum number of iterations in the lanczos process.
        """
        self.k, self.n = samples.shape
        self.lam = lam
        self.samples = samples.astype(np.float64).copy()
        self.samples[:] -= self.samples.mean(axis=0)
        self.stds = self.samples.std(axis=0)
        self.vars = self.stds ** 2
        self.epsilon = epsilon
        self.maxiter = maxiter

    def velocity(self, x):
        samples = theano.shared(floatX(np.copy(self.samples, 'C')))
        D = theano.shared(floatX(self.vars))
        Ax = tt.dot(samples, x)
        Ax = tt.dot(samples.T, Ax) / (self.k - 1)
        return floatX(1 - self.lam) * Ax + floatX(self.lam) * D * x

    def matvec(self, x):
        Ax = self.samples.T.dot(self.samples.dot(x)) / (self.k - 1)
        return (1 - self.lam) * Ax + self.lam * x * self.vars

    def energy(self, x):
        return floatX(0.5) * x.dot(self.velocity(x))

    def random(self):
        b = normal(size=self.n)
        # Use the diagonal matrix as preconditioner
        matvec = lambda x: self.matvec(x / self.stds) / self.stds
        sample = _sample_mvn_lanczos(matvec, b, self.epsilon, self.maxiter)
        return floatX(sample / self.stds)


try:
    import sksparse.cholmod as cholmod
    chol_available = True
except ImportError:
    chol_available = False

if chol_available:
    __all__ += ['QuadPotential_Sparse']

    import theano
    import theano.sparse

    class QuadPotential_Sparse(object):
        def __init__(self, A):
            self.A = A
            self.size = A.shape[0]
            self.factor = factor = cholmod.cholesky(A)
            self.d_sqrt = np.sqrt(factor.D())

        def velocity(self, x):
            A = theano.sparse.as_sparse(self.A)
            return theano.sparse.dot(A, x)

        def random(self):
            n = floatX(normal(size=self.size))
            n /= self.d_sqrt
            n = self.factor.solve_Lt(n)
            n = self.factor.apply_Pt(n)
            return n

        def energy(self, x):
            return 0.5 * x.T.dot(self.velocity(x))
